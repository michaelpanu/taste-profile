from __future__ import annotations
"""
Standalone food recommendation script.

Goal:
Given liked food IDs and disliked food IDs, recommend 5 new foods that are
similar ingredient-wise and taste-profile-wise.

Assumptions:
1. You already have a Chroma collection with food documents and metadata.
2. Food IDs in liked/disliked arrays exist in the collection.
3. Each Chroma item has:
   - document text containing recipe/food details
   - metadata with fields like:
        recipe_name
        ingredients OR ingredient_names OR ingredients_text
        cuisine_type
        dish_type
        meal_type
        spiciness
        sweetness
        saltiness
        sourness
        bitterness
        savoriness
        fattiness
4. Embeddings are already stored in Chroma.
5. The user-facing app/LLM already maps a typed food to the closest food_id.
"""

import math
import re

import ast
import json

from collections import Counter
from dataclasses import dataclass
from typing import Any


import chromadb

from config import CHROMA_PATH, COLLECTION_NAME


# -----------------------------
# Config
# -----------------------------

NUM_RECOMMENDATIONS = 5
CANDIDATE_POOL_SIZE = 75

# Put the actual IDs from your app here.
LIKED_FOOD_IDS = ["11940",  "12787", "5184", "11677", "7238"]

DISLIKED_FOOD_IDS = ["5315", "2481", "12077"]

# Taste metadata fields from your current setup.
TASTE_FIELDS = [
    "spiciness",
    "sweetness",
    "saltiness",
    "sourness",
    "bitterness",
    "savoriness",
    "fattiness",
]

# Ranking weights.
# You can tweak these quickly during the hackathon.
VECTOR_WEIGHT = 0.55
INGREDIENT_WEIGHT = 0.25
TASTE_WEIGHT = 0.25
CUISINE_DISH_WEIGHT = 0.10
DISLIKED_INGREDIENT_PENALTY = 0.35
DISLIKED_TASTE_PENALTY = 0.30
POPULAR_INGREDIENT_REPEAT_BOOST = 0.10


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class FoodItem:
    food_id: str
    document: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None


@dataclass
class UserTasteProfile:
    liked_foods: list[FoodItem]
    disliked_foods: list[FoodItem]
    liked_ingredients: Counter
    disliked_ingredients: Counter
    liked_tastes: Counter
    disliked_tastes: Counter
    liked_cuisines: Counter
    liked_dish_types: Counter
    profile_vector: list[float] | None


# -----------------------------
# Basic helpers
# -----------------------------

def safe_list(value: Any) -> list[str]:
    """
    Converts metadata fields to a clean list of lowercase strings.

    Handles:
    - list[str]
    - comma-separated strings
    - None
    """

    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(x).strip().lower()
            for x in value
            if str(x).strip()
        ]

    if isinstance(value, str):
        # Handles strings like "chicken, rice, garlic"
        # or Chroma-style serialized-ish values.
        cleaned = value.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
        parts = re.split(r",|;", cleaned)
        return [
            x.strip().lower()
            for x in parts
            if x.strip()
        ]

    return [str(value).strip().lower()]


def cosine_similarity(a, b) -> float:
    if a is None or b is None:
        return 0.0

    a = list(a)
    b = list(b)

    if len(a) == 0 or len(b) == 0:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def average_vectors(vectors) -> list[float] | None:
    vectors = [list(v) for v in vectors if v is not None]

    if len(vectors) == 0:
        return None

    dim = len(vectors[0])
    averaged = []

    for i in range(dim):
        averaged.append(sum(vector[i] for vector in vectors) / len(vectors))

    return averaged


def subtract_vectors(
    positive_vector: list[float] | None,
    negative_vector: list[float] | None,
    negative_weight: float = 0.45,
) -> list[float] | None:
    """
    Creates a preference vector:

        user_vector = liked_average - negative_weight * disliked_average

    If there are no disliked foods, this is just liked_average.
    """

    if positive_vector is None:
        return None

    if negative_vector is None:
        return positive_vector

    return [
        pos - negative_weight * neg
        for pos, neg in zip(positive_vector, negative_vector)
    ]


# -----------------------------
# Ingredient extraction
# -----------------------------

def extract_food_name_from_ingredient_obj(obj):
    """
    Handles ingredient objects like:
    {"food": "kosher salt", "quantity": 1, "measure": "teaspoon"}
    {"text": "low sodium soy sauce", "quantity": 2, "measure": "tablespoon"}
    """

    if not isinstance(obj, dict):
        return None

    for key in ["food", "name", "ingredient", "text"]:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    return None


def parse_ingredient_field(value):
    """
    Converts messy ingredient metadata into clean ingredient names only.
    """

    if value is None:
        return []

    # Already a list.
    if isinstance(value, list):
        ingredients = []

        for item in value:
            if isinstance(item, dict):
                food_name = extract_food_name_from_ingredient_obj(item)
                if food_name:
                    ingredients.append(food_name)
            elif isinstance(item, str):
                ingredients.append(item.strip().lower())

        return [x for x in ingredients if x]

    # Single dict.
    if isinstance(value, dict):
        food_name = extract_food_name_from_ingredient_obj(value)
        return [food_name] if food_name else []

    # Stringified list/dict.
    if isinstance(value, str):
        raw = value.strip()

        # Try JSON first.
        try:
            parsed = json.loads(raw)
            return parse_ingredient_field(parsed)
        except Exception:
            pass

        # Try Python literal format.
        try:
            parsed = ast.literal_eval(raw)
            return parse_ingredient_field(parsed)
        except Exception:
            pass

        # Fallback for strings like:
        # "{food: kosher salt, quantity: 1, measure: teaspoon}"
        food_matches = re.findall(
            r"(?:food|name|ingredient|text)\s*:\s*([^,\}\]]+)",
            raw,
            flags=re.IGNORECASE,
        )

        if food_matches:
            return [
                match.strip().strip("'\"").lower()
                for match in food_matches
                if match.strip()
            ]

        # Last fallback: only use this if the string seems like a plain ingredient,
        # not a serialized object.
        if "quantity:" in raw.lower() or "measure:" in raw.lower():
            return []

        return [raw.lower()]

    return []

def extract_ingredients(food: FoodItem) -> list[str]:
    metadata = food.metadata

    possible_fields = [
        "ingredients",
        "ingredient_names",
        "ingredients_text",
        "ingredient_text",
    ]

    for field in possible_fields:
        if field in metadata and metadata[field]:
            parsed = parse_ingredient_field(metadata[field])
            if parsed:
                return parsed

    document = food.document or ""
    lower_doc = document.lower()

    if "ingredients:" in lower_doc:
        after = lower_doc.split("ingredients:", 1)[1]
        before_instructions = re.split(
            r"instructions:|directions:|method:|preparation:",
            after,
            maxsplit=1,
        )[0]

        rough_items = [
            token.strip()
            for token in re.split(r",|\n|;", before_instructions)
            if len(token.strip()) > 2
        ]

        cleaned_items = []

        for item in rough_items:
            parsed = parse_ingredient_field(item)
            cleaned_items.extend(parsed)

        return cleaned_items

    return []

def extract_tastes(food: FoodItem) -> list[str]:
    tastes = []

    for taste in TASTE_FIELDS:
        if food.metadata.get(taste) is True:
            tastes.append(taste)

    return tastes


def extract_cuisines(food: FoodItem) -> list[str]:
    return safe_list(food.metadata.get("cuisine_type"))


def extract_dish_types(food: FoodItem) -> list[str]:
    return safe_list(food.metadata.get("dish_type"))


# -----------------------------
# Chroma loading
# -----------------------------

def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma_client.get_collection(name=COLLECTION_NAME)


def fetch_foods_by_ids(collection, food_ids: list[str]) -> list[FoodItem]:
    """
    Fetches foods from Chroma by ID.

    Includes embeddings because we need them for profile vector creation.
    """

    if not food_ids:
        return []

    result = collection.get(
        ids=food_ids,
        include=["documents", "metadatas", "embeddings"],
    )

    foods = []

    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    embeddings = result.get("embeddings", [])

    for i, food_id in enumerate(ids):
        foods.append(
            FoodItem(
                food_id=food_id,
                document=documents[i] if i < len(documents) else "",
                metadata=metadatas[i] if i < len(metadatas) else {},
                embedding=embeddings[i] if i < len(embeddings) else None,
            )
        )

    return foods


# -----------------------------
# User profile building
# -----------------------------

def build_user_taste_profile(
    liked_foods: list[FoodItem],
    disliked_foods: list[FoodItem],
) -> UserTasteProfile:
    liked_ingredients = Counter()
    disliked_ingredients = Counter()
    liked_tastes = Counter()
    disliked_tastes = Counter()
    liked_cuisines = Counter()
    liked_dish_types = Counter()

    for food in liked_foods:
        liked_ingredients.update(extract_ingredients(food))
        liked_tastes.update(extract_tastes(food))
        liked_cuisines.update(extract_cuisines(food))
        liked_dish_types.update(extract_dish_types(food))

    for food in disliked_foods:
        disliked_ingredients.update(extract_ingredients(food))
        disliked_tastes.update(extract_tastes(food))

    liked_vectors = [
        food.embedding
        for food in liked_foods
        if food.embedding is not None
    ]

    disliked_vectors = [
        food.embedding
        for food in disliked_foods
        if food.embedding is not None
    ]

    liked_average = average_vectors(liked_vectors)
    disliked_average = average_vectors(disliked_vectors)

    profile_vector = subtract_vectors(
        positive_vector=liked_average,
        negative_vector=disliked_average,
        negative_weight=0.45,
    )

    return UserTasteProfile(
        liked_foods=liked_foods,
        disliked_foods=disliked_foods,
        liked_ingredients=liked_ingredients,
        disliked_ingredients=disliked_ingredients,
        liked_tastes=liked_tastes,
        disliked_tastes=disliked_tastes,
        liked_cuisines=liked_cuisines,
        liked_dish_types=liked_dish_types,
        profile_vector=profile_vector,
    )


# -----------------------------
# Candidate retrieval
# -----------------------------

def retrieve_candidates(collection, profile: UserTasteProfile, n_results: int) -> list[FoodItem]:
    """
    Retrieves candidate foods from Chroma using the user profile vector.

    If there are no liked foods/profile vector, this cannot make a meaningful
    personalized recommendation.
    """

    if profile.profile_vector is None:
        raise ValueError(
            "Cannot build recommendations without at least one liked food with an embedding."
        )

    result = collection.query(
        query_embeddings=[profile.profile_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    candidates = []

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    embeddings = result.get("embeddings", [[]])[0]

    for i, food_id in enumerate(ids):
        candidates.append(
            FoodItem(
                food_id=food_id,
                document=documents[i] if i < len(documents) else "",
                metadata=metadatas[i] if i < len(metadatas) else {},
                embedding=embeddings[i] if i < len(embeddings) else None,
            )
        )

    return candidates


# -----------------------------
# Scoring
# -----------------------------

def normalized_counter_overlap(candidate_values: list[str], preference_counter: Counter) -> float:
    """
    Scores how much a candidate overlaps with a weighted preference counter.

    Example:
    If user likes chicken 5 times and garlic 3 times, a candidate with chicken
    and garlic scores higher than a candidate with only garlic.
    """

    if not candidate_values or not preference_counter:
        return 0.0

    total_preference_weight = sum(preference_counter.values())

    if total_preference_weight == 0:
        return 0.0

    overlap = 0.0

    for value in candidate_values:
        overlap += preference_counter.get(value, 0)

    return min(overlap / total_preference_weight, 1.0)


def taste_match_score(candidate_tastes: list[str], liked_tastes: Counter) -> float:
    return normalized_counter_overlap(candidate_tastes, liked_tastes)


def ingredient_match_score(candidate_ingredients: list[str], liked_ingredients: Counter) -> float:
    return normalized_counter_overlap(candidate_ingredients, liked_ingredients)


def cuisine_dish_match_score(candidate: FoodItem, profile: UserTasteProfile) -> float:
    cuisines = extract_cuisines(candidate)
    dish_types = extract_dish_types(candidate)

    cuisine_score = normalized_counter_overlap(cuisines, profile.liked_cuisines)
    dish_score = normalized_counter_overlap(dish_types, profile.liked_dish_types)

    return 0.5 * cuisine_score + 0.5 * dish_score


def disliked_ingredient_penalty(candidate_ingredients: list[str], disliked_ingredients: Counter) -> float:
    return normalized_counter_overlap(candidate_ingredients, disliked_ingredients)


def disliked_taste_penalty(candidate_tastes: list[str], disliked_tastes: Counter) -> float:
    return normalized_counter_overlap(candidate_tastes, disliked_tastes)


def popular_ingredient_repeat_score(candidate_ingredients: list[str], liked_ingredients: Counter) -> float:
    """
    Gives a small extra boost if the candidate contains the user's most frequent liked ingredients.

    This is what makes things like repeated chicken preferences matter.
    """

    if not liked_ingredients:
        return 0.0

    top_liked = {ingredient for ingredient, _ in liked_ingredients.most_common(5)}

    if not top_liked:
        return 0.0

    candidate_set = set(candidate_ingredients)
    overlap_count = len(candidate_set.intersection(top_liked))

    return min(overlap_count / len(top_liked), 1.0)


def score_candidate(candidate: FoodItem, profile: UserTasteProfile) -> tuple[float, dict[str, float]]:
    candidate_ingredients = extract_ingredients(candidate)
    candidate_tastes = extract_tastes(candidate)

    vector_score = 0.0
    if profile.profile_vector is not None and candidate.embedding is not None:
        # cosine_similarity ranges roughly from -1 to 1.
        # Convert to 0 to 1.
        raw_cosine = cosine_similarity(profile.profile_vector, candidate.embedding)
        vector_score = (raw_cosine + 1) / 2

    ingredient_score = ingredient_match_score(
        candidate_ingredients,
        profile.liked_ingredients,
    )

    taste_score = taste_match_score(
        candidate_tastes,
        profile.liked_tastes,
    )

    cuisine_dish_score = cuisine_dish_match_score(
        candidate,
        profile,
    )

    disliked_ingredient_score = disliked_ingredient_penalty(
        candidate_ingredients,
        profile.disliked_ingredients,
    )

    disliked_taste_score = disliked_taste_penalty(
        candidate_tastes,
        profile.disliked_tastes,
    )

    repeat_boost = popular_ingredient_repeat_score(
        candidate_ingredients,
        profile.liked_ingredients,
    )

    final_score = (
        VECTOR_WEIGHT * vector_score
        + INGREDIENT_WEIGHT * ingredient_score
        + TASTE_WEIGHT * taste_score
        + CUISINE_DISH_WEIGHT * cuisine_dish_score
        + POPULAR_INGREDIENT_REPEAT_BOOST * repeat_boost
        - DISLIKED_INGREDIENT_PENALTY * disliked_ingredient_score
        - DISLIKED_TASTE_PENALTY * disliked_taste_score
    )

    details = {
        "vector_score": round(vector_score, 4),
        "ingredient_score": round(ingredient_score, 4),
        "taste_score": round(taste_score, 4),
        "cuisine_dish_score": round(cuisine_dish_score, 4),
        "repeat_boost": round(repeat_boost, 4),
        "disliked_ingredient_penalty": round(disliked_ingredient_score, 4),
        "disliked_taste_penalty": round(disliked_taste_score, 4),
        "final_score": round(final_score, 4),
    }

    return final_score, details


# -----------------------------
# Recommendation generation
# -----------------------------

def explain_recommendation(candidate: FoodItem, profile: UserTasteProfile, score_details: dict[str, float]) -> str:
    ingredients = extract_ingredients(candidate)
    tastes = extract_tastes(candidate)

    top_liked_ingredients = {
        ingredient
        for ingredient, _ in profile.liked_ingredients.most_common(8)
    }

    matching_ingredients = [
        ingredient
        for ingredient in ingredients
        if ingredient in top_liked_ingredients
    ][:5]

    top_liked_tastes = {
        taste
        for taste, _ in profile.liked_tastes.most_common()
    }

    matching_tastes = [
        taste
        for taste in tastes
        if taste in top_liked_tastes
    ]

    reasons = []

    if matching_ingredients:
        reasons.append(f"matches liked ingredients: {', '.join(matching_ingredients)}")

    if matching_tastes:
        pretty_tastes = [
            taste.replace("fattiness", "rich/fatty")
            for taste in matching_tastes
        ]
        reasons.append(f"matches taste profile: {', '.join(pretty_tastes)}")

    if score_details["vector_score"] >= 0.75:
        reasons.append("overall vector profile is similar")

    if not reasons:
        reasons.append("closest overall match from the available food database")

    return "; ".join(reasons)


def recommend_foods(
    liked_food_ids: list[str],
    disliked_food_ids: list[str],
    num_recommendations: int = NUM_RECOMMENDATIONS,
) -> list[dict[str, Any]]:
    collection = get_collection()

    liked_foods = fetch_foods_by_ids(collection, liked_food_ids)
    disliked_foods = fetch_foods_by_ids(collection, disliked_food_ids)

    if not liked_foods:
        raise ValueError("You need at least one liked food ID to generate recommendations.")

    profile = build_user_taste_profile(
        liked_foods=liked_foods,
        disliked_foods=disliked_foods,
    )

    candidates = retrieve_candidates(
        collection=collection,
        profile=profile,
        n_results=CANDIDATE_POOL_SIZE,
    )

    excluded_ids = set(liked_food_ids) | set(disliked_food_ids)

    scored_candidates = []

    for candidate in candidates:
        if candidate.food_id in excluded_ids:
            continue

        final_score, score_details = score_candidate(candidate, profile)

        scored_candidates.append(
            {
                "food": candidate,
                "score": final_score,
                "score_details": score_details,
            }
        )

    scored_candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    recommendations = []

    for item in scored_candidates[:num_recommendations]:
        food = item["food"]
        metadata = food.metadata
        score_details = item["score_details"]

        recommendations.append(
            {
                "food_id": food.food_id,
                "recipe_name": metadata.get("recipe_name", "Unknown"),
                "score": round(item["score"], 4),
                "score_details": score_details,
                "cuisine_type": metadata.get("cuisine_type", []),
                "dish_type": metadata.get("dish_type", []),
                "meal_type": metadata.get("meal_type", []),
                "tastes": extract_tastes(food),
                "top_matching_reason": explain_recommendation(
                    candidate=food,
                    profile=profile,
                    score_details=score_details,
                ),
                "document_preview": food.document[:350],
            }
        )

    return recommendations


# -----------------------------
# Debug/profile printing
# -----------------------------

def print_user_profile_summary(profile: UserTasteProfile) -> None:
    print("\nUser taste profile summary")
    print("=" * 80)

    print("\nTop liked ingredients:")
    for ingredient, count in profile.liked_ingredients.most_common(10):
        print(f"- {ingredient}: {count}")

    print("\nTop disliked ingredients:")
    for ingredient, count in profile.disliked_ingredients.most_common(10):
        print(f"- {ingredient}: {count}")

    print("\nLiked taste frequencies:")
    for taste, count in profile.liked_tastes.most_common():
        print(f"- {taste}: {count}")

    print("\nDisliked taste frequencies:")
    for taste, count in profile.disliked_tastes.most_common():
        print(f"- {taste}: {count}")

    print("\nLiked cuisines:")
    for cuisine, count in profile.liked_cuisines.most_common(10):
        print(f"- {cuisine}: {count}")

    print("\nLiked dish types:")
    for dish_type, count in profile.liked_dish_types.most_common(10):
        print(f"- {dish_type}: {count}")


def print_recommendations(recommendations: list[dict[str, Any]]) -> None:
    print("\nRecommended foods")
    print("=" * 80)

    if not recommendations:
        print("No recommendations found.")
        return

    for i, rec in enumerate(recommendations, start=1):
        print(f"\n{i}. {rec['recipe_name']}")
        print(f"   Food ID: {rec['food_id']}")
        print(f"   Score: {rec['score']}")
        print(f"   Reason: {rec['top_matching_reason']}")
        print(f"   Tastes: {rec['tastes']}")
        print(f"   Cuisine: {rec['cuisine_type']}")
        print(f"   Dish type: {rec['dish_type']}")
        print(f"   Meal type: {rec['meal_type']}")
        print(f"   Score details: {rec['score_details']}")
        print(f"   Preview: {rec['document_preview']}")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    collection = get_collection()

    liked_foods = fetch_foods_by_ids(collection, LIKED_FOOD_IDS)
    disliked_foods = fetch_foods_by_ids(collection, DISLIKED_FOOD_IDS)

    profile = build_user_taste_profile(
        liked_foods=liked_foods,
        disliked_foods=disliked_foods,
    )

    print_user_profile_summary(profile)

    recommendations = recommend_foods(
        liked_food_ids=LIKED_FOOD_IDS,
        disliked_food_ids=DISLIKED_FOOD_IDS,
        num_recommendations=5,
    )

    print_recommendations(recommendations)