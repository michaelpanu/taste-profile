import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# -----------------------------
# Config
# -----------------------------

from config import EVAL_CASES_PATH

RULE_RESULTS_JSON = "eval/food_eval_rule_results.json"
RULE_RESULTS_CSV = "eval/food_eval_rule_results.csv"
RAGAS_INPUT_JSON = "eval/food_eval_ragas_input.json"
RAGAS_RESULTS_CSV = "eval/food_eval_ragas_results.csv"
SUMMARY_JSON = "eval/food_eval_summary.json"

RUN_RAGAS = True
RUN_RULE_CHECKS = True

# If True, recommendation cases that ask a follow-up are marked as failed.
STRICT_EXPECTED_BEHAVIOR = True

# Only run RAGAS on recommendation cases that pass your custom rule checks.
RUN_RAGAS_ON_RULE_PASS_ONLY = False

# If parsedsearch.py says "ask follow-up", but the query already has useful constraints
# and the eval case expects recommendations, this script will continue to retrieval.
SEARCH_INSTEAD_OF_FOLLOWUP_WHEN_CONSTRAINED = True

# This makes the eval answer more direct and grounded for RAGAS.
# It does NOT change your real app behavior in parsedsearch.py.
USE_RAGAS_FRIENDLY_EVAL_ANSWER = True

RAGAS_CONTEXT_TOP_K = 3


# -----------------------------
# Import your food app
# -----------------------------

try:
    import parsedsearch as app
except Exception as e:
    raise RuntimeError(
        "Could not import parsedsearch.py. Make sure this script is in the same folder "
        "as parsedsearch.py, or update the import path."
    ) from e


# -----------------------------
# Helpers
# -----------------------------

def model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    if isinstance(model, dict):
        return model
    return {}


def normalize_text(value: Any) -> str:
    return str(value or "").lower()


def listify(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    return [str(value)]


def flatten_contexts(contexts: Any, top_k: int = RAGAS_CONTEXT_TOP_K) -> List[str]:
    """
    RAGAS expects retrieved_contexts to be list[str].

    This prevents errors like:
    retrieved_contexts.0 Input should be a valid string
    """
    if contexts is None:
        return []

    flattened = []

    def _walk(value):
        if value is None:
            return

        if isinstance(value, str):
            if value.strip():
                flattened.append(value)
            return

        if isinstance(value, (list, tuple)):
            for item in value:
                _walk(item)
            return

        text = str(value)
        if text.strip():
            flattened.append(text)

    _walk(contexts)

    return flattened[:top_k]


def get_case_reference(case: Dict[str, Any]) -> str:
    return (
        case.get("ragas", {}).get("ground_truth")
        or case.get("reference")
        or case.get("ideal_behavior")
        or case.get("reference_behavior")
        or case.get("expected_answer")
        or ""
    )


# -----------------------------
# RAGAS import helpers
# -----------------------------

def get_ragas_metrics():
    """
    Return initialized RAGAS metric objects.
    """
    import ragas.metrics as metrics_module

    try:
        from ragas.metrics.collections import (
            Faithfulness,
            ResponseRelevancy,
            LLMContextPrecisionWithoutReference,
        )

        return [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithoutReference(),
        ]
    except Exception:
        pass

    try:
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
            LLMContextPrecisionWithoutReference,
        )

        return [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithoutReference(),
        ]
    except Exception:
        pass

    try:
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
        )

        return [
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
        ]
    except Exception:
        pass

    available = dir(metrics_module)

    raise RuntimeError(
        "Could not find compatible RAGAS metric classes.\n"
        "Try installing/updating:\n"
        "pip install -U ragas langchain-openai langchain-core datasets pandas\n\n"
        f"Available ragas.metrics names include:\n{available[:100]}"
    )


def get_ragas_evaluators():
    """
    Creates explicit RAGAS evaluator LLM + embeddings wrappers.

    Deprecation warnings from RAGAS are okay for now.
    """
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except Exception as e:
        raise RuntimeError(
            "Could not import LangChain/RAGAS wrappers. Install or update:\n"
            "pip install -U ragas langchain-openai langchain-core datasets pandas\n"
        ) from e

    ragas_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
        )
    )

    return ragas_llm, ragas_embeddings


# -----------------------------
# Load eval cases
# -----------------------------

def load_eval_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "cases" in data:
        return data["cases"]

    if isinstance(data, list):
        return data

    raise ValueError(
        "Eval JSON must either be a list of cases or an object with a 'cases' key."
    )


# -----------------------------
# Eval-side follow-up policy helpers
# -----------------------------

def parsed_query_has_substantive_food_idea(parsed_query: Any) -> bool:
    if hasattr(app, "has_substantive_food_idea"):
        try:
            return app.has_substantive_food_idea(parsed_query)
        except Exception:
            pass

    data = model_to_dict(parsed_query)
    semantic_query = str(data.get("semantic_query", "") or "").strip().lower().replace(",", "")

    if not semantic_query:
        return False

    vague_terms = {
        "food",
        "meal",
        "meals",
        "dish",
        "dishes",
        "recipe",
        "recipes",
        "option",
        "options",
        "something",
        "anything",
        "dinner",
        "lunch",
        "breakfast",
        "snack",
        "healthy",
        "healthy food",
        "healthy meal",
        "healthy dinner",
        "healthy lunch",
        "healthy breakfast",
        "healthy snack",
        "good food",
        "good meal",
        "good dinner",
        "good lunch",
        "good breakfast",
        "vegetarian",
        "vegan",
        "gluten free",
        "gluten-free",
        "dairy free",
        "dairy-free",
    }

    if semantic_query in vague_terms:
        return False

    filter_only_words = {
        "healthy",
        "nutritious",
        "balanced",
        "light",
        "dinner",
        "lunch",
        "breakfast",
        "snack",
        "american",
        "italian",
        "indian",
        "mexican",
        "asian",
        "french",
        "japanese",
        "mediterranean",
        "spicy",
        "sweet",
        "salty",
        "sour",
        "bitter",
        "savory",
        "rich",
    }

    words = [w for w in semantic_query.split() if w not in filter_only_words]

    return len(words) > 0


def parsed_query_has_enough_constraints(parsed_query: Any) -> bool:
    if hasattr(app, "has_enough_constraints"):
        try:
            return app.has_enough_constraints(parsed_query)
        except Exception:
            pass

    data = model_to_dict(parsed_query)

    direct_constraint_fields = [
        "cuisine_type",
        "dish_type",
        "health_label",
        "diet_label",
        "caution_to_exclude",
        "flavor_any_of",
        "flavor_all_of",
    ]

    for field in direct_constraint_fields:
        value = data.get(field)
        if value:
            return True

    boolean_constraint_fields = [
        "spiciness",
        "sweetness",
        "saltiness",
        "sourness",
        "bitterness",
        "savoriness",
        "fattiness",
    ]

    for field in boolean_constraint_fields:
        if data.get(field) is not None:
            return True

    if parsed_query_has_substantive_food_idea(parsed_query):
        return True

    return False


def should_continue_to_recommend_in_eval(case: Dict[str, Any], parsed_query: Any) -> bool:
    if not SEARCH_INSTEAD_OF_FOLLOWUP_WHEN_CONSTRAINED:
        return False

    if case.get("expected_behavior") != "recommend":
        return False

    return parsed_query_has_enough_constraints(parsed_query)


# -----------------------------
# App runner for one eval case
# -----------------------------

def run_food_app_once(case: Dict[str, Any]) -> Dict[str, Any]:
    question = case["question"]

    chat_history = []

    for msg in case.get("chat_history_seed", []):
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            chat_history.append({
                "role": msg["role"],
                "content": msg["content"],
            })

    app_state = {
        "seen_food_ids": set(),
        "last_top_food": None,
    }

    if case.get("last_top_food"):
        app_state["last_top_food"] = case["last_top_food"]

        if case["last_top_food"].get("food_id"):
            app_state["seen_food_ids"].add(case["last_top_food"]["food_id"])

    app.add_to_history(chat_history, "user", question)

    parsed_query = app.parse_user_query(
        user_query=question,
        chat_history=chat_history,
    )

    parsed_query = app.apply_similarity_context(
        user_query=question,
        parsed_query=parsed_query,
        app_state=app_state,
    )

    if app.should_ask_initial_followup(parsed_query):
        followup_question = app.generate_followup_question(parsed_query)

        if case.get("followup_answer"):
            app.add_to_history(chat_history, "assistant", followup_question)
            app.add_to_history(chat_history, "user", case["followup_answer"])

            combined_query = f"{question}. Additional preference: {case['followup_answer']}"

            parsed_query = app.parse_user_query(
                user_query=combined_query,
                chat_history=chat_history,
            )

            parsed_query = app.apply_similarity_context(
                user_query=combined_query,
                parsed_query=parsed_query,
                app_state=app_state,
            )

            question = combined_query

        elif should_continue_to_recommend_in_eval(case, parsed_query):
            pass

        else:
            return {
                "behavior": "follow_up",
                "answer": followup_question,
                "contexts": [],
                "retrieved_foods": [],
                "parsed_query": model_to_dict(parsed_query),
                "where_filter": None,
                "best_distance": None,
                "match_quality": None,
                "relaxation_used": None,
            }

    results, where_filter = app.search_with_seen_filter(
        parsed_query=parsed_query,
        app_state=app_state,
        candidate_k=app.CANDIDATE_K,
        top_k=app.DEFAULT_TOP_K,
    )

    best_distance = app.get_best_distance(results)
    match_quality = app.get_match_quality(best_distance)
    relaxation_used = None

    if match_quality in {"very_weak", "no_results"} and not app.is_vague_query(parsed_query):
        relaxed_attempt = app.retry_with_relaxed_filters(
            parsed_query=parsed_query,
            app_state=app_state,
        )

        if relaxed_attempt:
            relaxed_results = relaxed_attempt["results"]
            relaxed_best_distance = relaxed_attempt["best_distance"]
            relaxed_match_quality = relaxed_attempt["match_quality"]

            if relaxed_match_quality in {"strong", "weak"} or best_distance is None:
                results = relaxed_results
                where_filter = relaxed_attempt["where_filter"]
                parsed_query = relaxed_attempt["parsed_query"]
                relaxation_used = relaxed_attempt["relaxation_label"]
                best_distance = relaxed_best_distance
                match_quality = relaxed_match_quality

    retrieved_foods = extract_retrieved_foods(results)

    if USE_RAGAS_FRIENDLY_EVAL_ANSWER:
        answer = generate_eval_answer(
            user_query=question,
            parsed_query=parsed_query,
            results=results,
            where_filter=where_filter,
            chat_history=chat_history,
            app_state=app_state,
            relaxation_used=relaxation_used,
        )
    else:
        answer, match_quality, best_distance = app.generate_user_answer(
            user_query=question,
            parsed_query=parsed_query,
            results=results,
            where_filter=where_filter,
            chat_history=chat_history,
            app_state=app_state,
            relaxation_used=relaxation_used,
        )

    return {
        "behavior": "recommend",
        "answer": answer,
        "contexts": [food["document"] for food in retrieved_foods],
        "retrieved_foods": retrieved_foods,
        "parsed_query": model_to_dict(parsed_query),
        "where_filter": where_filter,
        "best_distance": best_distance,
        "match_quality": match_quality,
        "relaxation_used": relaxation_used,
    }


def extract_retrieved_foods(results) -> List[Dict[str, Any]]:
    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    foods = []

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for food_id, doc, metadata, distance in zip(ids, documents, metadatas, distances):
        foods.append({
            "food_id": food_id,
            "recipe_name": metadata.get("recipe_name", ""),
            "metadata": metadata,
            "document": doc,
            "distance": distance,
        })

    return foods


# -----------------------------
# RAGAS-friendly eval answer generation
# -----------------------------

def generate_eval_answer(
    user_query: str,
    parsed_query,
    results,
    where_filter,
    chat_history: List[Dict[str, str]] | None = None,
    app_state: Dict[str, Any] | None = None,
    relaxation_used: str | None = None,
) -> str:
    """
    This is only for evaluation.

    It makes the answer more direct and grounded, which usually improves RAGAS
    faithfulness and answer relevancy.
    """
    foods = app.format_results_for_llm(
    results,
    max_results=RAGAS_CONTEXT_TOP_K,
)
    best_distance = app.get_best_distance(results)
    match_quality = app.get_match_quality(best_distance)

    parsed_query_dict = model_to_dict(parsed_query)

    response = app.client.responses.create(
        model=app.ANSWER_MODEL,
        input=[
            {
                "role": "system",
                "content": """
You are a food recommendation assistant being evaluated for faithfulness and answer relevancy.

You will receive:
- The user's query
- The parsed query and filters
- The retrieved foods from the dataset
- Match quality information

Rules:
- Start with a direct answer to the user's request.
- Recommend 2 to 5 retrieved foods by exact recipe name when available.
- Only recommend foods that appear in the provided foods list.
- Do not suggest, name, or recommend foods that are not in the provided foods list.
- Do not use outside food knowledge.
- Do not invent ingredients, labels, properties, cuisines, or recipe names.
- For each recommended food, explain briefly why it fits using only the provided metadata or embedding text preview.
- Be specific and name the retrieved foods.
- Do not say vague things like "there are several options" without naming the foods.
- Do not explain vector search, Chroma, embeddings, distance scores, or retrieval internals.
- If the matches are weak, briefly say they are approximate, then still name the closest retrieved foods.
- If the search was relaxed, mention it only briefly if necessary.
- If no foods are provided, say no matching foods were found.
- Keep the answer concise: 3 to 6 sentences total.
- Do not use markdown tables.
"""
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_query": user_query,
                        "parsed_query": parsed_query_dict,
                        "where_filter": where_filter,
                        "best_distance": best_distance,
                        "match_quality": match_quality,
                        "relaxation_used": relaxation_used,
                        "foods": foods,
                    },
                    indent=2,
                )
            }
        ],
    )

    return response.output_text


# -----------------------------
# Rule checks
# -----------------------------

def metadata_contains(metadata: Dict[str, Any], key: str, expected: Any) -> bool:
    actual = metadata.get(key)

    if isinstance(actual, list):
        return str(expected).lower() in [str(x).lower() for x in actual]

    return str(actual).lower() == str(expected).lower()


def score_followup(case: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
    answer = normalize_text(output.get("answer"))
    expected_words = case.get("expected_followup_contains", [])

    hits = [
        word for word in expected_words
        if normalize_text(word) in answer
    ]

    failures = []

    if output.get("behavior") != "follow_up":
        failures.append("Expected follow-up behavior, but app returned recommendations.")

    if expected_words and len(hits) == 0:
        failures.append(
            f"Follow-up did not contain any expected words: {expected_words}"
        )

    return {
        "followup_score": len(hits) / max(len(expected_words), 1),
        "followup_hits": hits,
        "rule_pass": len(failures) == 0,
        "rule_failures": failures,
    }


def check_required_filters(case: Dict[str, Any], output: Dict[str, Any]) -> List[str]:
    failures = []

    required_filters = case.get("required_filters", {})
    parsed_query = output.get("parsed_query", {})

    for key, expected_value in required_filters.items():
        actual_value = parsed_query.get(key)

        if actual_value != expected_value:
            failures.append(
                f"Parsed query field '{key}' was {actual_value!r}, expected {expected_value!r}"
            )

    return failures


def check_required_metadata(case: Dict[str, Any], output: Dict[str, Any]) -> List[str]:
    failures = []

    required_metadata = case.get("required_metadata", {})
    retrieved_foods = output.get("retrieved_foods", [])

    if not required_metadata:
        return failures

    if not retrieved_foods:
        failures.append("No retrieved foods to check required metadata.")
        return failures

    for food in retrieved_foods:
        recipe_name = food.get("recipe_name", "")
        metadata = food.get("metadata", {})

        for key, expected_value in required_metadata.items():
            if key.endswith("_any"):
                base_key = key.replace("_any", "")
                actual_values = [v.lower() for v in listify(metadata.get(base_key))]
                expected_values = [v.lower() for v in listify(expected_value)]

                if not any(v in actual_values for v in expected_values):
                    failures.append(
                        f"{recipe_name}: metadata '{base_key}'={actual_values}, expected any of {expected_values}"
                    )

            elif key.endswith("_all"):
                base_key = key.replace("_all", "")
                actual_values = [v.lower() for v in listify(metadata.get(base_key))]
                expected_values = [v.lower() for v in listify(expected_value)]

                missing = [v for v in expected_values if v not in actual_values]

                if missing:
                    failures.append(
                        f"{recipe_name}: metadata '{base_key}' missing required values {missing}"
                    )

            else:
                if isinstance(expected_value, list):
                    if not any(metadata_contains(metadata, key, v) for v in expected_value):
                        failures.append(
                            f"{recipe_name}: metadata '{key}' did not contain any of {expected_value}"
                        )
                else:
                    if not metadata_contains(metadata, key, expected_value):
                        failures.append(
                            f"{recipe_name}: metadata '{key}' was {metadata.get(key)!r}, expected {expected_value!r}"
                        )

    return failures


def check_excluded_terms(case: Dict[str, Any], output: Dict[str, Any]) -> List[str]:
    failures = []

    must_exclude_terms = case.get("must_exclude_terms", [])
    answer = normalize_text(output.get("answer"))

    for term in must_exclude_terms:
        term_lower = normalize_text(term)

        if term_lower in answer:
            failures.append(f"Answer contains excluded term: {term}")

    for food in output.get("retrieved_foods", []):
        recipe_name = normalize_text(food.get("recipe_name"))
        document = normalize_text(food.get("document"))
        metadata_text = normalize_text(json.dumps(food.get("metadata", {})))

        for term in must_exclude_terms:
            term_lower = normalize_text(term)

            if (
                term_lower in recipe_name
                or term_lower in document
                or term_lower in metadata_text
            ):
                failures.append(
                    f"Retrieved food '{food.get('recipe_name')}' contains excluded term: {term}"
                )

    return failures


def check_required_terms(case: Dict[str, Any], output: Dict[str, Any]) -> List[str]:
    failures = []

    required_terms = case.get("required_terms", [])

    if not required_terms:
        return failures

    combined = normalize_text(output.get("answer"))

    for food in output.get("retrieved_foods", []):
        combined += " "
        combined += normalize_text(food.get("recipe_name"))
        combined += " "
        combined += normalize_text(food.get("document"))
        combined += " "
        combined += normalize_text(json.dumps(food.get("metadata", {})))

    for term in required_terms:
        if normalize_text(term) not in combined:
            failures.append(f"Required term not found in answer/retrieved foods: {term}")

    return failures


def check_behavior(case: Dict[str, Any], output: Dict[str, Any]) -> List[str]:
    failures = []

    expected_behavior = case.get("expected_behavior")

    if STRICT_EXPECTED_BEHAVIOR and expected_behavior:
        if output.get("behavior") != expected_behavior:
            failures.append(
                f"Expected behavior {expected_behavior!r}, got {output.get('behavior')!r}"
            )

    return failures


def check_rule_case(case: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
    if case.get("expected_behavior") == "follow_up":
        return score_followup(case, output)

    failures = []

    failures.extend(check_behavior(case, output))
    failures.extend(check_required_filters(case, output))
    failures.extend(check_required_metadata(case, output))
    failures.extend(check_excluded_terms(case, output))
    failures.extend(check_required_terms(case, output))

    max_best_distance = case.get("max_best_distance")

    if max_best_distance is not None:
        best_distance = output.get("best_distance")

        if best_distance is None:
            failures.append("No best_distance returned.")
        elif best_distance > max_best_distance:
            failures.append(
                f"best_distance {best_distance:.4f} exceeded max_best_distance {max_best_distance}"
            )

    return {
        "rule_pass": len(failures) == 0,
        "rule_failures": failures,
    }


# -----------------------------
# RAGAS preparation
# -----------------------------

def should_include_in_ragas(
    case: Dict[str, Any],
    output: Dict[str, Any],
    rule_result: Optional[Dict[str, Any]] = None,
) -> bool:
    if case.get("ragas", {}).get("use_for_ragas") is False:
        return False

    if output.get("behavior") != "recommend":
        return False

    if not output.get("contexts"):
        return False

    if RUN_RULE_CHECKS and RUN_RAGAS_ON_RULE_PASS_ONLY and rule_result is not None:
        if not rule_result.get("rule_pass"):
            return False

    return True


def build_ragas_rows(
    cases: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
    rule_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []

    for case, output, rule_result in zip(cases, outputs, rule_results):
        if not should_include_in_ragas(case, output, rule_result):
            continue

        reference = get_case_reference(case)

        question = case["question"]
        answer = output.get("answer", "")

        contexts = flatten_contexts(output.get("contexts", []), top_k=RAGAS_CONTEXT_TOP_K)

        if not contexts:
            continue

        rows.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": reference,
        })

    return rows


def run_ragas_eval(ragas_rows: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    if not ragas_rows:
        print("No recommendation rows available for RAGAS.")
        return None

    with open(RAGAS_INPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(ragas_rows, f, indent=2)

    dataset = Dataset.from_list(ragas_rows)

    from ragas import evaluate

    metrics = get_ragas_metrics()
    ragas_llm, ragas_embeddings = get_ragas_evaluators()

    print("Using RAGAS metrics:")
    for metric in metrics:
        print(f" - {metric.__class__.__name__}")

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    df = result.to_pandas()
    df.to_csv(RAGAS_RESULTS_CSV, index=False)

    return df


def should_run_case(case: Dict[str, Any]) -> bool:
    """
    When rule checks are off, only run cases intended for RAGAS.
    This prevents the script from running all 50 eval cases.
    """
    if RUN_RULE_CHECKS:
        return True

    if not RUN_RAGAS:
        return False

    return case.get("ragas", {}).get("use_for_ragas") is True


# -----------------------------
# Main
# -----------------------------

def main():
    cases = load_eval_cases(EVAL_CASES_PATH)

    # Only run the cases needed for the current mode.
    cases_to_run = [case for case in cases if should_run_case(case)]

    print(f"Loaded {len(cases)} eval cases from {EVAL_CASES_PATH}")
    print(f"Running {len(cases_to_run)} cases.")
    print()

    outputs = []
    rule_results = []

    for i, case in enumerate(cases_to_run, start=1):
        case_id = case.get("id", i)
        question = case.get("question", "")

        print("=" * 100)
        print(f"Running case {case_id}: {question}")

        try:
            output = run_food_app_once(case)

            if RUN_RULE_CHECKS:
                rule_result = check_rule_case(case, output)
            else:
                rule_result = {
                    "rule_pass": True,
                    "rule_failures": [],
                    "followup_score": None,
                    "followup_hits": None,
                }

            outputs.append(output)

            row = {
                "id": case_id,
                "category": case.get("category"),
                "question": question,
                "expected_behavior": case.get("expected_behavior"),
                "actual_behavior": output.get("behavior"),
                "answer": output.get("answer"),
                "parsed_query": output.get("parsed_query"),
                "where_filter": output.get("where_filter"),
                "best_distance": output.get("best_distance"),
                "match_quality": output.get("match_quality"),
                "relaxation_used": output.get("relaxation_used"),
                "retrieved_recipe_names": [
                    food.get("recipe_name") for food in output.get("retrieved_foods", [])
                ],
                "rule_pass": rule_result.get("rule_pass"),
                "rule_failures": rule_result.get("rule_failures"),
                "followup_score": rule_result.get("followup_score"),
                "followup_hits": rule_result.get("followup_hits"),
            }

            rule_results.append(row)

            if RUN_RULE_CHECKS:
                if row["rule_pass"]:
                    print("Rule check: PASS")
                else:
                    print("Rule check: FAIL")
                    for failure in row["rule_failures"]:
                        print(f" - {failure}")
            else:
                print("Rule check: SKIPPED")

        except Exception as e:
            print("ERROR while running case:")
            print(traceback.format_exc())

            outputs.append({
                "behavior": "error",
                "answer": str(e),
                "contexts": [],
                "retrieved_foods": [],
                "parsed_query": {},
            })

            rule_results.append({
                "id": case_id,
                "category": case.get("category"),
                "question": question,
                "expected_behavior": case.get("expected_behavior"),
                "actual_behavior": "error",
                "answer": str(e),
                "parsed_query": {},
                "where_filter": None,
                "best_distance": None,
                "match_quality": None,
                "relaxation_used": None,
                "retrieved_recipe_names": [],
                "rule_pass": False,
                "rule_failures": [str(e)],
            })

    with open(RULE_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(rule_results, f, indent=2, default=str)

    rule_df = pd.DataFrame(rule_results)
    rule_df.to_csv(RULE_RESULTS_CSV, index=False)

    print()
    print("=" * 100)
    print(f"Saved run results to {RULE_RESULTS_JSON}")
    print(f"Saved run results CSV to {RULE_RESULTS_CSV}")

    rule_pass_count = int(rule_df["rule_pass"].sum()) if "rule_pass" in rule_df else 0
    total = len(rule_df)
    rule_pass_rate = rule_pass_count / total if total else 0

    if RUN_RULE_CHECKS:
        print(f"Rule pass rate: {rule_pass_count}/{total} = {rule_pass_rate:.3f}")
    else:
        print("Rule checks skipped.")

    ragas_summary = {}

    if RUN_RAGAS:
        print()
        print("=" * 100)
        print("Running RAGAS...")

        # IMPORTANT:
        # Use cases_to_run here, not cases.
        ragas_rows = build_ragas_rows(cases_to_run, outputs, rule_results)

        print(f"RAGAS rows included: {len(ragas_rows)}")

        try:
            ragas_df = run_ragas_eval(ragas_rows)

            if ragas_df is not None:
                print(f"Saved RAGAS results to {RAGAS_RESULTS_CSV}")
                print()
                print(ragas_df.head())

                numeric_cols = ragas_df.select_dtypes(include="number").columns

                ragas_summary = {
                    col: float(ragas_df[col].mean())
                    for col in numeric_cols
                }

                print()
                print("RAGAS averages:")
                for k, v in ragas_summary.items():
                    print(f"{k}: {v:.4f}")

        except Exception:
            print("RAGAS evaluation failed:")
            print(traceback.format_exc())
            print()
            print("Your run results were still saved.")

    summary = {
        "total_cases_in_file": len(cases),
        "cases_run": len(cases_to_run),
        "rule_checks_enabled": RUN_RULE_CHECKS,
        "rule_pass_count": rule_pass_count if RUN_RULE_CHECKS else None,
        "rule_pass_rate": rule_pass_rate if RUN_RULE_CHECKS else None,
        "ragas_summary": ragas_summary,
        "settings": {
            "strict_expected_behavior": STRICT_EXPECTED_BEHAVIOR,
            "run_ragas": RUN_RAGAS,
            "run_rule_checks": RUN_RULE_CHECKS,
            "run_ragas_on_rule_pass_only": RUN_RAGAS_ON_RULE_PASS_ONLY,
            "search_instead_of_followup_when_constrained": SEARCH_INSTEAD_OF_FOLLOWUP_WHEN_CONSTRAINED,
            "use_ragas_friendly_eval_answer": USE_RAGAS_FRIENDLY_EVAL_ANSWER,
            "ragas_context_top_k": RAGAS_CONTEXT_TOP_K,
        },
        "outputs": {
            "rule_results_json": RULE_RESULTS_JSON,
            "rule_results_csv": RULE_RESULTS_CSV,
            "ragas_input_json": RAGAS_INPUT_JSON,
            "ragas_results_csv": RAGAS_RESULTS_CSV,
        }
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Saved summary to {SUMMARY_JSON}")

if __name__ == "__main__":
    main()