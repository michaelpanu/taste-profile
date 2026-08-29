import json
import chromadb
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, Literal

from config import CHROMA_PATH, COLLECTION_NAME

EMBEDDING_MODEL = "text-embedding-3-small"
PARSER_MODEL = "gpt-4.1-mini"
ANSWER_MODEL = "gpt-4.1-mini"

# Distance thresholds.
# Lower distance = better match.
STRONG_MATCH_DISTANCE = 0.45
WEAK_MATCH_DISTANCE = 0.60

DEFAULT_TOP_K = 5
CANDIDATE_K = 25

client = OpenAI()

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)


# -----------------------------
# LLM parser schema
# -----------------------------

class ParsedFoodQuery(BaseModel):
    semantic_query: str = Field(
        description=(
            "The part of the user query that should be embedded for semantic search. "
            "Remove words that are already represented by hard filters, but keep core foods, "
            "ingredients, cuisines, dishes, and preparation styles. "
            "Do not include healthy/nutritious/balanced/light unless it is part of a specific dish name."
        )
    )

    health_label: Optional[
        Literal[
            "Vegan",
            "Vegetarian",
            "Dairy-Free",
            "Gluten-Free",
            "Egg-Free",
            "Peanut-Free",
            "Tree-Nut-Free",
            "Soy-Free",
            "Fish-Free",
            "Shellfish-Free",
        ]
    ] = None

    diet_label: Optional[
        Literal[
            "Balanced",
            "High-Fiber",
            "High-Protein",
            "Low-Carb",
            "Low-Fat",
            "Low-Sodium",
        ]
    ] = None

    caution_to_exclude: Optional[
        Literal[
            "Gluten",
            "Wheat",
            "Sulfites",
            "FODMAP",
            "Peanut",
            "Tree Nuts",
            "Soy",
            "Shellfish",
            "Fish",
            "Dairy",
            "Egg",
        ]
    ] = None

    meal_type: Optional[
        Literal[
            "breakfast",
            "lunch/dinner",
            "snack",
            "teatime",
        ]
    ] = None

    cuisine_type: Optional[str] = Field(
        default=None,
        description=(
            "Cuisine type if explicitly requested, such as asian, italian, mexican, "
            "american, mediterranean, indian, french, japanese, etc. Use lowercase."
        )
    )

    cuisine_was_explicit: bool = Field(
        default=False,
        description=(
            "True if the user explicitly requested a cuisine in the current message, "
            "such as Indian, Mexican, Italian, Asian, etc."
        )
    )

    dish_type: Optional[str] = Field(
        default=None,
        description=(
            "Dish type if explicitly requested, such as main course, salad, soup, "
            "dessert, starter, side dish, sandwich, drinks, etc. Use lowercase."
        )
    )

    spiciness: Optional[bool] = None
    sweetness: Optional[bool] = None
    saltiness: Optional[bool] = None
    sourness: Optional[bool] = None
    bitterness: Optional[bool] = None
    savoriness: Optional[bool] = None
    fattiness: Optional[bool] = None

    flavor_any_of: Optional[
        list[
            Literal[
                "spicy",
                "sweet",
                "salty",
                "sour",
                "bitter",
                "savory",
                "rich_or_fatty",
            ]
        ]
    ] = Field(
        default=None,
        description=(
            "Use this when the user asks for one of several possible flavor filters. "
            "This means any one of the listed flavors is acceptable."
        )
    )

    flavor_all_of: Optional[
        list[
            Literal[
                "spicy",
                "sweet",
                "salty",
                "sour",
                "bitter",
                "savory",
                "rich_or_fatty",
            ]
        ]
    ] = Field(
        default=None,
        description=(
            "Use this when the user clearly requires multiple flavors together. "
            "This means all listed flavors are required."
        )
    )

    healthy: Optional[bool] = Field(
        default=None,
        description=(
            "True if the user asks for something healthy, nutritious, light, balanced, "
            "high protein, high fiber, low sodium, or generally better-for-you."
        )
    )

    possible_contradiction: bool = Field(
        default=False,
        description=(
            "True if the user query may contain a contradiction or difficult request, "
            "such as vegetarian chicken, vegan beef, vegan pork, dairy-free cheese, etc."
        )
    )

    contradiction_reason: Optional[str] = Field(
        default=None,
        description="Brief explanation of the possible contradiction."
    )


# -----------------------------
# Conversation memory helpers
# -----------------------------

def add_to_history(chat_history: list[dict], role: str, content: str):
    chat_history.append({
        "role": role,
        "content": content,
    })


def get_recent_chat_context(chat_history: list[dict], max_messages: int = 10) -> list[dict]:
    return chat_history[-max_messages:]


def parse_user_query(user_query: str, chat_history: list[dict] | None = None) -> ParsedFoodQuery:
    recent_context = get_recent_chat_context(chat_history or [])

    response = client.responses.parse(
        model=PARSER_MODEL,
        input=[
            {
                "role": "system",
                "content": """
You parse food search queries for a recipe recommendation app.

Return only structured data that fits the schema.

Conversation context:
- Use the recent chat history to understand follow-up messages.
- If the user previously said they are vegetarian, vegan, dairy-free, gluten-free, etc., keep that constraint for later follow-up queries unless they explicitly change it.
- If the user says things like "make it spicy", "something with tofu", "what about pasta", or "for dinner", interpret that using the prior conversation.
- If the user says "similar to that", "other variations", "more like that", or "I liked it", use the previously recommended food from the chat as context.
- Do not forget dietary restrictions or allergies from earlier in the conversation.
- If the current message clearly overrides an earlier preference, use the newest preference.

Rules:
- Use hard filters only when the user clearly asks for them.
- Do not guess dietary restrictions.
- If the user says vegan, set health_label to Vegan.
- If the user says vegetarian, set health_label to Vegetarian.
- If the user says dairy-free or no dairy, use Dairy-Free.
- If the user says gluten-free or no gluten, use Gluten-Free or caution_to_exclude Gluten.
- If the user says peanut-free or no peanuts, use Peanut-Free or caution_to_exclude Peanut.
- If the user says dinner, lunch, or meal, use meal_type lunch/dinner.
- If the user says breakfast, use meal_type breakfast.
- If the user asks for something healthy, nutritious, balanced, light, better-for-you, or generally good for them, set healthy true.
- If the user specifically says high protein, high fiber, low sodium, low fat, low carb, or balanced, set the matching diet_label.
- For vague "healthy", set healthy true instead of choosing one exact diet_label.
- Do NOT put "healthy", "nutritious", "balanced", "light", or "better-for-you" in semantic_query.
- If the user only says something like "healthy dinner", semantic_query should be empty or very minimal, because the app should ask a follow-up.
- If the user says "healthy American spicy dinner" but gives no ingredient, dish, or food type, semantic_query should still be minimal; cuisine, meal, healthy, and spicy should be filters.

Cuisine logic:
- If the user explicitly asks for a cuisine, such as Indian food, Mexican food, Italian food, French food, Japanese food, Asian food, etc., set cuisine_type and set cuisine_was_explicit true.
- If the cuisine comes only from previous context and was not requested in the current user message, cuisine_was_explicit may be false.
- If the user asks for "Indian foods with similar taste profiles", cuisine_type should be "indian" and cuisine_was_explicit should be true.
- Explicit cuisine is a hard constraint.

Flavor logic:
- The available hard flavor filters are spicy, sweet, salty, sour, bitter, savory, and rich_or_fatty.
- Map "spicy" to spicy.
- Map "sweet" to sweet.
- Map "salty" to salty.
- Map "sour" or "tangy" to sour.
- Map "bitter" to bitter.
- Map "savory", "umami", or "hearty" to savory.
- Map "rich", "fatty", "creamy", or "buttery" to rich_or_fatty when used as a flavor/texture preference.
- If the user says "or" between flavor words, use flavor_any_of.
- If the user says "either", "any of", "one of", or "or", use flavor_any_of.
- If the user says "and" between flavor words, use flavor_all_of.
- If the user says "both", "all of", or clearly combines flavors with "and", use flavor_all_of.
- Do not set direct flavor booleans when using flavor_any_of or flavor_all_of.
- Do not use flavor_all_of or flavor_any_of for only one flavor.
- If there is only one flavor, use the direct boolean field instead.
- Example: "spicy or sweet" -> flavor_any_of ["spicy", "sweet"].
- Example: "spicy and savory" -> flavor_all_of ["spicy", "savory"].
- Example: "sweet and sour" -> flavor_all_of ["sweet", "sour"].
- Example: "salty or savory" -> flavor_any_of ["salty", "savory"].
- Example: "rich and spicy" -> flavor_all_of ["rich_or_fatty", "spicy"].
- Example: "sour or spicy soup" -> flavor_any_of ["sour", "spicy"].
- Example: "savory ramen" -> savoriness true.
- Example: "spicy American dinner" -> spiciness true, cuisine_type "american", meal_type "lunch/dinner".
- Words like fresh, comforting, light, crispy, crunchy, warm, cozy, and refreshing should usually stay in semantic_query unless there is a matching metadata field.

Contradiction handling:
- If the user asks for something like vegetarian chicken, vegan beef, vegan pork, vegan fish, or dairy-free cheese, set possible_contradiction to true.
- Still keep the food concept in semantic_query.
- Example: "vegetarian chicken" should become semantic_query "chicken", health_label Vegetarian, possible_contradiction true.
- Example: "vegan beef tacos" should become semantic_query "beef tacos", health_label Vegan, possible_contradiction true.
- Example: "dairy-free cheese pasta" should become semantic_query "cheese pasta", health_label Dairy-Free, possible_contradiction true.

Important:
- The semantic_query should keep the food/ingredient/dish idea.
- Remove filter-only words strictly. Do not leave any of them in.
- Filter-only words include: healthy, nutritious, balanced, high protein, high fiber, low sodium, low fat, low carb, vegan, vegetarian, gluten-free, dairy-free, spicy, sweet, salty, sour, bitter, savory, breakfast, dinner, lunch, snack, and cuisine names.
- Example: "give me some good healthy dinner options" becomes semantic_query "", meal_type "lunch/dinner", healthy true.
- Example: "healthy chicken dinner" becomes semantic_query "chicken", meal_type "lunch/dinner", healthy true.
- Example: "healthy vegetarian dinner" becomes semantic_query "", health_label "Vegetarian", meal_type "lunch/dinner", healthy true.
- Example: "healthy vegan dinner" becomes semantic_query "", health_label "Vegan", meal_type "lunch/dinner", healthy true.
- Example: "healthy spicy American dinner" becomes semantic_query "", cuisine_type "american", meal_type "lunch/dinner", spiciness true, healthy true.
- Example: "high protein breakfast" becomes semantic_query "", diet_label "High-Protein", meal_type "breakfast".
- Example: "spicy vegan tofu dinner" becomes semantic_query "tofu", health_label "Vegan", meal_type "lunch/dinner", spiciness true.
- Example: "vegetarian chicken" becomes semantic_query "chicken", health_label Vegetarian.
- Example: "gluten free pasta dinner" becomes semantic_query "pasta", health_label Gluten-Free, meal_type lunch/dinner.
- Example: "not spicy chicken soup" becomes semantic_query "chicken soup", spiciness false.
"""
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "recent_chat_context": recent_context,
                        "current_user_query": user_query,
                    },
                    indent=2,
                ),
            }
        ],
        text_format=ParsedFoodQuery,
    )

    return response.output_parsed


# -----------------------------
# Embedding
# -----------------------------

def embed_query(query: str) -> list[float]:
    # Chroma/OpenAI embeddings need some text.
    # If the parser returns an empty semantic query, use a neutral fallback.
    safe_query = query.strip() or "food recommendation"

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=safe_query,
    )
    return response.data[0].embedding


# -----------------------------
# Chroma filter builder
# -----------------------------

FLAVOR_TO_METADATA = {
    "spicy": "spiciness",
    "sweet": "sweetness",
    "salty": "saltiness",
    "sour": "sourness",
    "bitter": "bitterness",
    "savory": "savoriness",
    "rich_or_fatty": "fattiness",
}


def build_flavor_any_filter(flavor_any_of):
    if not flavor_any_of:
        return None

    flavor_conditions = []

    for flavor in flavor_any_of:
        metadata_field = FLAVOR_TO_METADATA.get(flavor)
        if metadata_field:
            flavor_conditions.append({metadata_field: True})

    if not flavor_conditions:
        return None

    if len(flavor_conditions) == 1:
        return flavor_conditions[0]

    return {"$or": flavor_conditions}


def build_flavor_all_filters(flavor_all_of):
    if not flavor_all_of:
        return []

    flavor_conditions = []

    for flavor in flavor_all_of:
        metadata_field = FLAVOR_TO_METADATA.get(flavor)
        if metadata_field:
            flavor_conditions.append({metadata_field: True})

    return flavor_conditions


def build_where_filter(
    health_label: str | None = None,
    diet_label: str | None = None,
    caution_to_exclude: str | None = None,
    meal_type: str | None = None,
    cuisine_type: str | None = None,
    dish_type: str | None = None,
    spiciness: bool | None = None,
    sweetness: bool | None = None,
    saltiness: bool | None = None,
    sourness: bool | None = None,
    bitterness: bool | None = None,
    savoriness: bool | None = None,
    fattiness: bool | None = None,
    flavor_any_of=None,
    flavor_all_of=None,
    healthy: bool | None = None,
):
    conditions = []

    if health_label:
        conditions.append({
            "health_labels": {
                "$contains": health_label
            }
        })

    if diet_label:
        conditions.append({
            "diet_labels": {
                "$contains": diet_label
            }
        })

    if caution_to_exclude:
        conditions.append({
            "cautions": {
                "$not_contains": caution_to_exclude
            }
        })

    if meal_type:
        conditions.append({
            "meal_type": {
                "$contains": meal_type
            }
        })

    if cuisine_type:
        conditions.append({
            "cuisine_type": {
                "$contains": cuisine_type
            }
        })

    if dish_type:
        conditions.append({
            "dish_type": {
                "$contains": dish_type
            }
        })

    flavor_any_filter = build_flavor_any_filter(flavor_any_of)
    flavor_all_filters = build_flavor_all_filters(flavor_all_of)

    if flavor_any_filter:
        conditions.append(flavor_any_filter)

    if flavor_all_filters:
        conditions.extend(flavor_all_filters)

    if not flavor_any_filter and not flavor_all_filters:
        if spiciness is not None:
            conditions.append({"spiciness": spiciness})

        if sweetness is not None:
            conditions.append({"sweetness": sweetness})

        if saltiness is not None:
            conditions.append({"saltiness": saltiness})

        if sourness is not None:
            conditions.append({"sourness": sourness})

        if bitterness is not None:
            conditions.append({"bitterness": bitterness})

        if savoriness is not None:
            conditions.append({"savoriness": savoriness})

        if fattiness is not None:
            conditions.append({"fattiness": fattiness})

    if healthy:
        conditions.append({
            "$or": [
                {"diet_labels": {"$contains": "Balanced"}},
                {"diet_labels": {"$contains": "High-Fiber"}},
                {"diet_labels": {"$contains": "High-Protein"}},
                {"diet_labels": {"$contains": "Low-Sodium"}},
                {"diet_labels": {"$contains": "Low-Fat"}},
            ]
        })

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


# -----------------------------
# Search
# -----------------------------

def search_foods_from_parsed_query(
    parsed_query: ParsedFoodQuery,
    top_k: int = DEFAULT_TOP_K,
):
    query_embedding = embed_query(parsed_query.semantic_query)

    where_filter = build_where_filter(
        health_label=parsed_query.health_label,
        diet_label=parsed_query.diet_label,
        caution_to_exclude=parsed_query.caution_to_exclude,
        meal_type=parsed_query.meal_type,
        cuisine_type=parsed_query.cuisine_type,
        dish_type=parsed_query.dish_type,
        spiciness=parsed_query.spiciness,
        sweetness=parsed_query.sweetness,
        saltiness=parsed_query.saltiness,
        sourness=parsed_query.sourness,
        bitterness=parsed_query.bitterness,
        savoriness=parsed_query.savoriness,
        fattiness=parsed_query.fattiness,
        flavor_any_of=parsed_query.flavor_any_of,
        flavor_all_of=parsed_query.flavor_all_of,
        healthy=parsed_query.healthy,
    )

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if where_filter:
        query_args["where"] = where_filter

    results = collection.query(**query_args)

    return results, where_filter


# -----------------------------
# Result filtering / memory helpers
# -----------------------------

def empty_results():
    return {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }


def filter_out_seen_results(results, seen_food_ids: set, top_k: int = DEFAULT_TOP_K):
    if not results.get("ids") or not results["ids"][0]:
        return empty_results()

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    filtered_ids = []
    filtered_documents = []
    filtered_metadatas = []
    filtered_distances = []

    for food_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        if food_id in seen_food_ids:
            continue

        filtered_ids.append(food_id)
        filtered_documents.append(document)
        filtered_metadatas.append(metadata)
        filtered_distances.append(distance)

        if len(filtered_ids) >= top_k:
            break

    return {
        "ids": [filtered_ids],
        "documents": [filtered_documents],
        "metadatas": [filtered_metadatas],
        "distances": [filtered_distances],
    }


def update_seen_and_last_food(results, app_state: dict):
    if not results.get("ids") or not results["ids"][0]:
        return

    for food_id in results["ids"][0]:
        app_state["seen_food_ids"].add(food_id)

    top_id = results["ids"][0][0]
    top_doc = results["documents"][0][0]
    top_metadata = results["metadatas"][0][0]

    app_state["last_top_food"] = {
        "food_id": top_id,
        "recipe_name": top_metadata.get("recipe_name", "Unknown"),
        "document": top_doc,
        "metadata": top_metadata,
    }


def user_wants_similar(user_query: str) -> bool:
    q = user_query.lower()

    triggers = [
        "similar",
        "like that",
        "more like that",
        "other variations",
        "variations",
        "something like that",
        "i liked it",
        "i really liked it",
        "more options like",
        "other options like",
        "taste similar",
        "similar taste",
    ]

    return any(trigger in q for trigger in triggers)


def build_similarity_query_from_last_food(last_top_food: dict) -> str:
    metadata = last_top_food.get("metadata", {})
    document = last_top_food.get("document", "")

    recipe_name = metadata.get("recipe_name", "")
    dish_type = metadata.get("dish_type", [])
    diet_labels = metadata.get("diet_labels", [])

    flavor_words = []

    if metadata.get("spiciness"):
        flavor_words.append("spicy")
    if metadata.get("sweetness"):
        flavor_words.append("sweet")
    if metadata.get("saltiness"):
        flavor_words.append("salty")
    if metadata.get("sourness"):
        flavor_words.append("sour")
    if metadata.get("bitterness"):
        flavor_words.append("bitter")
    if metadata.get("savoriness"):
        flavor_words.append("savory")
    if metadata.get("fattiness"):
        flavor_words.append("rich or fatty")

    return (
        "Find foods with a similar ingredient and flavor profile. "
        f"Original recipe: {recipe_name}. "
        f"Original dish type: {dish_type}. "
        f"Original diet labels: {diet_labels}. "
        f"Original flavor profile: {', '.join(flavor_words)}. "
        f"Original details: {document[:1200]}"
    )


def apply_similarity_context(
    user_query: str,
    parsed_query: ParsedFoodQuery,
    app_state: dict,
):
    last_top_food = app_state.get("last_top_food")

    if not last_top_food:
        return parsed_query

    if not user_wants_similar(user_query):
        return parsed_query

    updated_query = parsed_query.model_copy(deep=True)
    updated_query.semantic_query = build_similarity_query_from_last_food(last_top_food)

    return updated_query


# -----------------------------
# Match quality helpers
# -----------------------------

def get_best_distance(results):
    if not results.get("distances"):
        return None

    if not results["distances"][0]:
        return None

    return results["distances"][0][0]


def get_match_quality(best_distance):
    if best_distance is None:
        return "no_results"

    if best_distance <= STRONG_MATCH_DISTANCE:
        return "strong"

    if best_distance <= WEAK_MATCH_DISTANCE:
        return "weak"

    return "very_weak"


def format_results_for_llm(results, max_results: int = DEFAULT_TOP_K):
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    foods = []

    for i in range(min(len(ids), max_results)):
        metadata = metadatas[i]

        foods.append({
            "rank": i + 1,
            "food_id": ids[i],
            "distance": distances[i],
            "recipe_name": metadata.get("recipe_name", "Unknown"),
            "health_labels": metadata.get("health_labels", []),
            "diet_labels": metadata.get("diet_labels", []),
            "cautions": metadata.get("cautions", []),
            "meal_type": metadata.get("meal_type", []),
            "cuisine_type": metadata.get("cuisine_type", []),
            "dish_type": metadata.get("dish_type", []),
            "spiciness": metadata.get("spiciness", None),
            "sweetness": metadata.get("sweetness", None),
            "saltiness": metadata.get("saltiness", None),
            "sourness": metadata.get("sourness", None),
            "bitterness": metadata.get("bitterness", None),
            "savoriness": metadata.get("savoriness", None),
            "fattiness": metadata.get("fattiness", None),
            "embedding_text_preview": documents[i][:700],
        })

    return foods


# -----------------------------
# Relaxed fallback search
# -----------------------------

def build_relaxed_versions(parsed_query: ParsedFoodQuery):
    relaxed_queries = []

    q1 = parsed_query.model_copy(deep=True)
    q1.spiciness = None
    q1.sweetness = None
    q1.saltiness = None
    q1.sourness = None
    q1.bitterness = None
    q1.savoriness = None
    q1.fattiness = None
    q1.flavor_any_of = None
    q1.flavor_all_of = None
    relaxed_queries.append(("without strict flavor filters", q1))

    q2 = parsed_query.model_copy(deep=True)
    q2.healthy = None
    relaxed_queries.append(("without the broad healthy filter", q2))

    q3 = parsed_query.model_copy(deep=True)
    q3.dish_type = None
    relaxed_queries.append(("without the dish type filter", q3))

    if not parsed_query.cuisine_was_explicit:
        q4 = parsed_query.model_copy(deep=True)
        q4.cuisine_type = None
        relaxed_queries.append(("without the cuisine filter", q4))

    q5 = parsed_query.model_copy(deep=True)
    q5.dish_type = None
    q5.spiciness = None
    q5.sweetness = None
    q5.saltiness = None
    q5.sourness = None
    q5.bitterness = None
    q5.savoriness = None
    q5.fattiness = None
    q5.flavor_any_of = None
    q5.flavor_all_of = None
    q5.healthy = None

    if not parsed_query.cuisine_was_explicit:
        q5.cuisine_type = None

    relaxed_queries.append(("with only core dietary, cuisine, and meal constraints", q5))

    return relaxed_queries


def search_with_seen_filter(
    parsed_query: ParsedFoodQuery,
    app_state: dict,
    candidate_k: int = CANDIDATE_K,
    top_k: int = DEFAULT_TOP_K,
):
    raw_results, where_filter = search_foods_from_parsed_query(
        parsed_query=parsed_query,
        top_k=candidate_k,
    )

    filtered_results = filter_out_seen_results(
        results=raw_results,
        seen_food_ids=app_state["seen_food_ids"],
        top_k=top_k,
    )

    return filtered_results, where_filter


def retry_with_relaxed_filters(
    parsed_query: ParsedFoodQuery,
    app_state: dict,
):
    best_attempt = None

    for relaxation_label, relaxed_query in build_relaxed_versions(parsed_query):
        results, where_filter = search_with_seen_filter(
            parsed_query=relaxed_query,
            app_state=app_state,
            candidate_k=CANDIDATE_K,
            top_k=DEFAULT_TOP_K,
        )

        best_distance = get_best_distance(results)
        match_quality = get_match_quality(best_distance)

        attempt = {
            "results": results,
            "where_filter": where_filter,
            "parsed_query": relaxed_query,
            "relaxation_label": relaxation_label,
            "best_distance": best_distance,
            "match_quality": match_quality,
        }

        if best_attempt is None:
            best_attempt = attempt
        else:
            old_distance = best_attempt["best_distance"]
            if old_distance is None or (best_distance is not None and best_distance < old_distance):
                best_attempt = attempt

        if match_quality in {"strong", "weak"}:
            return attempt

    return best_attempt


# -----------------------------
# Follow-up helpers
# -----------------------------

VAGUE_SEMANTIC_QUERIES = {
    "",
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
    "american dinner",
    "italian dinner",
    "indian dinner",
    "mexican dinner",
    "asian dinner",
}


def normalized_semantic_query(parsed_query: ParsedFoodQuery) -> str:
    return parsed_query.semantic_query.strip().lower().replace(",", "")


def has_substantive_food_idea(parsed_query: ParsedFoodQuery) -> bool:
    """
    Returns True if the user gave an actual ingredient, food, dish, or meaningful search idea.

    Examples that SHOULD count:
    - chicken
    - tofu
    - pasta
    - ramen
    - curry
    - pumpkin turkey
    - mushroom soup

    Examples that should NOT count by themselves:
    - healthy
    - dinner
    - spicy
    - vegan
    - american dinner
    """
    semantic_query = normalized_semantic_query(parsed_query)

    if not semantic_query:
        return False

    if semantic_query in VAGUE_SEMANTIC_QUERIES:
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
        "meal",
        "food",
        "option",
        "options",
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
        "vegan",
        "vegetarian",
        "gluten-free",
        "gluten",
        "dairy-free",
        "dairy",
    }

    words = [
        word
        for word in semantic_query.split()
        if word not in filter_only_words
    ]

    return len(words) > 0


def has_enough_constraints(parsed_query: ParsedFoodQuery) -> bool:
    """
    Returns True when the query has enough useful constraints to search without asking
    another generic follow-up.

    This prevents cases like these from asking follow-ups:
    - spicy vegetarian dinner
    - gluten-free and tasty
    - nut allergy
    - shellfish allergy with Asian savory
    - mild non-spicy food
    - Mediterranean vegetarian dinners
    - Indian food for first time
    """
    if parsed_query.cuisine_type:
        return True

    if parsed_query.dish_type:
        return True

    if parsed_query.health_label:
        return True

    if parsed_query.diet_label:
        return True

    if parsed_query.caution_to_exclude:
        return True

    if parsed_query.spiciness is not None:
        return True

    if parsed_query.sweetness is not None:
        return True

    if parsed_query.saltiness is not None:
        return True

    if parsed_query.sourness is not None:
        return True

    if parsed_query.bitterness is not None:
        return True

    if parsed_query.savoriness is not None:
        return True

    if parsed_query.fattiness is not None:
        return True

    if parsed_query.flavor_any_of:
        return True

    if parsed_query.flavor_all_of:
        return True

    if has_substantive_food_idea(parsed_query):
        return True

    return False


def is_vague_query(parsed_query: ParsedFoodQuery) -> bool:
    """
    Vague means there is not enough to search with.

    Important:
    A query can have an empty semantic_query but still be searchable if it has enough
    filters, such as vegetarian + spicy + dinner.
    """
    if has_enough_constraints(parsed_query):
        return False

    return True


def should_ask_initial_followup(
    parsed_query: ParsedFoodQuery,
    already_asked_followup: bool = False,
) -> bool:
    """
    Ask an initial follow-up only when the user gave almost no useful search constraints.

    This should ask:
    - healthy dinner
    - something tasty for dinner
    - something quick and easy please
    - surprise me

    This should NOT ask:
    - spicy vegetarian dinner
    - gluten-free and tasty
    - nut allergy
    - mild non-spicy food
    - Mediterranean vegetarian dinners
    """
    if already_asked_followup:
        return False

    return is_vague_query(parsed_query)


def should_ask_followup_after_search(
    parsed_query: ParsedFoodQuery,
    match_quality,
    already_asked_followup: bool = False,
) -> bool:
    """
    Ask a second follow-up only if:
    - we have not already asked one,
    - the search was weak,
    - and the query still has almost no useful constraints.

    If the user already gave cuisine/flavor/diet/allergy/etc., do not ask again.
    Use relaxed search instead.
    """
    if already_asked_followup:
        return False

    if match_quality not in {"weak", "very_weak", "no_results"}:
        return False

    if has_enough_constraints(parsed_query):
        return False

    return is_vague_query(parsed_query)


def generate_followup_question(parsed_query: ParsedFoodQuery) -> str:
    meal_phrase = ""

    if parsed_query.meal_type == "breakfast":
        meal_phrase = " breakfast"
    elif parsed_query.meal_type == "lunch/dinner":
        meal_phrase = " lunch or dinner"
    elif parsed_query.meal_type == "snack":
        meal_phrase = " snack"

    if parsed_query.health_label == "Vegetarian":
        ingredient_examples = (
            "tofu, beans, lentils, chickpeas, eggs, cheese, rice, pasta, vegetables, "
            "soup, salad, or something else vegetarian"
        )
    elif parsed_query.health_label == "Vegan":
        ingredient_examples = (
            "tofu, tempeh, beans, lentils, chickpeas, rice, quinoa, vegetables, "
            "soup, salad, or something else vegan"
        )
    elif parsed_query.health_label == "Dairy-Free":
        ingredient_examples = (
            "chicken, tofu, beans, lentils, rice, quinoa, vegetables, soup, salad, "
            "or something else dairy-free"
        )
    elif parsed_query.health_label == "Gluten-Free":
        ingredient_examples = (
            "chicken, tofu, beans, lentils, rice, quinoa, potatoes, vegetables, soup, salad, "
            "or something else gluten-free"
        )
    else:
        ingredient_examples = (
            "chicken, tofu, salmon, beans, rice, pasta, vegetables, soup, salad, "
            "or something else"
        )

    flavor_examples = "spicy, sweet, savory, salty, sour, mild, rich, fresh, tangy, or comforting"

    return (
        f"What kind of{meal_phrase} are you looking for? "
        f"You can give me a main ingredient, cuisine, dish type, or flavor. "
        f"For example: {ingredient_examples}. "
        f"You can also say a flavor like {flavor_examples}."
    )


def apply_followup_answer(
    original_query: str,
    followup_answer: str,
    chat_history: list[dict],
) -> tuple[str, ParsedFoodQuery]:
    if not followup_answer:
        return original_query, parse_user_query(
            user_query=original_query,
            chat_history=chat_history,
        )

    combined_query = f"{original_query}. Additional preference: {followup_answer}"

    updated_parsed_query = parse_user_query(
        user_query=combined_query,
        chat_history=chat_history,
    )

    return combined_query, updated_parsed_query


# -----------------------------
# LLM answer generation
# -----------------------------

def generate_user_answer(
    user_query: str,
    parsed_query: ParsedFoodQuery,
    results,
    where_filter,
    chat_history: list[dict] | None = None,
    app_state: dict | None = None,
    relaxation_used: str | None = None,
):
    foods = format_results_for_llm(results)
    best_distance = get_best_distance(results)
    match_quality = get_match_quality(best_distance)

    app_state = app_state or {}

    response = client.responses.create(
        model=ANSWER_MODEL,
        input=[
            {
                "role": "system",
                "content": """
You are a food recommendation assistant.

You will receive:
- The user's original query
- The parsed semantic query and metadata filters
- A list of retrieved food results from vector search
- The best vector distance
- A match quality label


Your job:
- Start with a direct answer to the user's request.
- Recommend 2 to 3 retrieved foods by exact recipe name when available.
- Only recommend foods that appear in the provided foods list.
- Do not suggest, name, or recommend foods that are not in the provided foods list.
- Do not use general knowledge to invent extra recipe examples.
- Never invent labels, ingredients, properties, recipe names, or dish names that are not in the provided results.
- For each recommended food, explain briefly why it fits using only the provided metadata or embedding text preview.
- Mention the actual recipe names from the provided foods list.
- Do not say vague things like "there are several options" without naming the foods.
- Do not over-explain vector search, distance scores, filters, or retrieval.
- If the provided foods are weak matches, briefly say the matches are approximate, then still recommend the closest retrieved foods.
- If no foods are provided, say no matching foods were found. Do not invent examples.
- If match_quality is "very_weak", say that no strong match was found, then list the closest retrieved options if any exist.
- If match_quality is "weak", you may briefly say the matches are approximate, but still give the best retrieved options first.
- If relaxation_used is not null, mention it only if it is important for the user to understand why the results are approximate.
- If the query contains a contradiction, such as vegetarian chicken or vegan beef, explain that exact matches may not exist, then recommend retrieved alternatives only.
- Mention dietary/allergy constraints carefully when relevant.
- If a food has cautions, mention them when relevant.
- Keep the answer concise: 3 to 6 sentences total.
- Do not use markdown tables.
- If nothing relevant matches, say "I unfortunately do not have a good answer for that, sorry"

VERY IMPORTANT: If no foods are found in the search, just say "I unfortunately do not have a good answer for that, sorry"

"""
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "recent_chat_context": get_recent_chat_context(chat_history or []),
                        "user_query": user_query,
                        "parsed_query": parsed_query.model_dump(),
                        "where_filter": where_filter,
                        "best_distance": best_distance,
                        "match_quality": match_quality,
                        "relaxation_used": relaxation_used,
                        "seen_food_ids_excluded": list(app_state.get("seen_food_ids", set())),
                        "last_top_food": app_state.get("last_top_food"),
                        "must_only_use_retrieved_foods": True,
                        "requested_cuisine": parsed_query.cuisine_type,
                        "cuisine_was_explicit": parsed_query.cuisine_was_explicit,
                        "foods": foods,
                    },
                    indent=2,
                )
            }
        ],
    )

    return response.output_text, match_quality, best_distance


# -----------------------------
# Debug printing
# -----------------------------

def print_results(results):
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not ids:
        print("No matching foods found.")
        return

    for i in range(len(ids)):
        metadata = metadatas[i]

        print("=" * 80)
        print(f"Rank: {i + 1}")
        print(f"Food ID: {ids[i]}")
        print(f"Distance: {distances[i]:.4f}")
        print(f"Recipe Name: {metadata.get('recipe_name', 'Unknown')}")
        print(f"Health labels: {metadata.get('health_labels', [])}")
        print(f"Diet labels: {metadata.get('diet_labels', [])}")
        print(f"Cautions: {metadata.get('cautions', [])}")
        print(f"Meal type: {metadata.get('meal_type', [])}")
        print(f"Cuisine type: {metadata.get('cuisine_type', [])}")
        print(f"Dish type: {metadata.get('dish_type', [])}")
        print(f"Spicy: {metadata.get('spiciness', None)}")
        print(f"Sweet: {metadata.get('sweetness', None)}")
        print(f"Salty: {metadata.get('saltiness', None)}")
        print(f"Sour: {metadata.get('sourness', None)}")
        print(f"Bitter: {metadata.get('bitterness', None)}")
        print(f"Savory: {metadata.get('savoriness', None)}")
        print(f"Rich/Fatty: {metadata.get('fattiness', None)}")
        print()
        print("Embedding Text Preview:")
        print(documents[i][:400])
        print()


# -----------------------------
# One search turn
# -----------------------------

def run_search_turn(user_query: str, chat_history: list[dict], app_state: dict):
    pending_followup = app_state.get("pending_followup")

    if pending_followup:
        add_to_history(chat_history, "user", user_query)

        already_asked_followup = True
        app_state["pending_followup"] = None

        user_query, parsed_query = apply_followup_answer(
            original_query=pending_followup["original_query"],
            followup_answer=user_query,
            chat_history=chat_history,
        )

        parsed_query = apply_similarity_context(
            user_query=user_query,
            parsed_query=parsed_query,
            app_state=app_state,
        )

        print()
        print("Updated parsed query after follow-up:")
        print(parsed_query.model_dump_json(indent=2))
    else:
        add_to_history(chat_history, "user", user_query)

        already_asked_followup = False

        parsed_query = parse_user_query(
            user_query=user_query,
            chat_history=chat_history,
        )

        parsed_query = apply_similarity_context(
            user_query=user_query,
            parsed_query=parsed_query,
            app_state=app_state,
        )

        print()
        print("Parsed query:")
        print(parsed_query.model_dump_json(indent=2))

        if should_ask_initial_followup(parsed_query, already_asked_followup):
            followup_question = generate_followup_question(parsed_query)

            print()
            print("Follow-up question:")
            print(followup_question)

            add_to_history(chat_history, "assistant", followup_question)
            app_state["pending_followup"] = {"original_query": user_query}

            return followup_question, None, None, None

    results, where_filter = search_with_seen_filter(
        parsed_query=parsed_query,
        app_state=app_state,
        candidate_k=CANDIDATE_K,
        top_k=DEFAULT_TOP_K,
    )

    print_results(results)
    best_distance = get_best_distance(results)
    match_quality = get_match_quality(best_distance)
    relaxation_used = None

    if should_ask_followup_after_search(parsed_query, match_quality, already_asked_followup):
        print()
        print("The first search was not very strong.")
        print(f"Best distance: {best_distance}")
        print(f"Match quality: {match_quality}")

        followup_question = generate_followup_question(parsed_query)

        print()
        print("Follow-up question:")
        print(followup_question)

        add_to_history(chat_history, "assistant", followup_question)
        app_state["pending_followup"] = {"original_query": user_query}

        return followup_question, None, None, None

    if match_quality in {"very_weak", "no_results"} and not is_vague_query(parsed_query):
        relaxed_attempt = retry_with_relaxed_filters(
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

    answer, match_quality, best_distance = generate_user_answer(
        user_query=user_query,
        parsed_query=parsed_query,
        results=results,
        where_filter=where_filter,
        chat_history=chat_history,
        app_state=app_state,
        relaxation_used=relaxation_used,
    )

    print()
    print("LLM Answer:")
    print(answer)

    add_to_history(chat_history, "assistant", answer)

    update_seen_and_last_food(results, app_state)

    return answer, results, match_quality, best_distance


# -----------------------------
# Main chat loop
# -----------------------------

if __name__ == "__main__":
    chat_history = []

    app_state = {
        "seen_food_ids": set(),
        "last_top_food": None,
        "pending_followup": None,
    }

    print("Food recommendation chat started.")
    print("Type 'quit', 'exit', or 'q' to stop.")
    print()

    while True:
        user_query = input("Enter your food search: ").strip()

        if user_query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if not user_query:
            continue

        run_search_turn(
            user_query=user_query,
            chat_history=chat_history,
            app_state=app_state,
        )


        print()
        print("-" * 100)
        print()