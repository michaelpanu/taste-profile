import ast
import json
import pandas as pd


INPUT_CSV = "data/cleanfood.csv"
OUTPUT_CSV = "data/datafood.csv"


TASTE_COLUMNS = {
    "sweetness": "sweet",
    "saltiness": "salty",
    "sourness": "sour",
    "bitterness": "bitter",
    "savoriness": "savory",
    "fattiness": "rich or fatty",
    "spiciness": "spicy",
}


def parse_list(value):
    """
    Safely parses list-like strings such as:
    '["Vegan", "Vegetarian"]'
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def parse_ingredients(value):
    """
    Extracts ingredient names from strings like:
    '[{"food":"tofu","text":"1 pound tofu"}]'
    """
    items = parse_list(value)

    ingredients = []

    for item in items:
        if isinstance(item, dict):
            food = item.get("food") or item.get("text")
            if food:
                ingredients.append(str(food).strip())
        elif isinstance(item, str):
            ingredients.append(item.strip())

    # Remove duplicates while preserving order
    seen = set()
    cleaned = []

    for ingredient in ingredients:
        key = ingredient.lower()
        if ingredient and key not in seen:
            seen.add(key)
            cleaned.append(ingredient)

    return cleaned


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def bool_is_true(value):
    """
    Handles booleans stored as True/False, strings, or 1/0.
    """
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def join_list(items):
    items = [str(x).strip() for x in items if str(x).strip()]
    return ", ".join(items)


def build_embedding_text(row):
    recipe_name = clean_text(row.get("recipe_name", ""))
    description = clean_text(row.get("short_description", ""))
    tags = clean_text(row.get("tags", ""))

    ingredients = parse_ingredients(row.get("ingredients", ""))

    diet_labels = parse_list(row.get("diet_labels", ""))
    health_labels = parse_list(row.get("health_labels", ""))
    cautions = parse_list(row.get("cautions", ""))
    cuisine_type = parse_list(row.get("cuisine_type", ""))
    meal_type = parse_list(row.get("meal_type", ""))
    dish_type = parse_list(row.get("dish_type", ""))

    taste_words = [
        taste_word
        for column, taste_word in TASTE_COLUMNS.items()
        if bool_is_true(row.get(column, False))
    ]

    parts = []

    if recipe_name:
        parts.append(f"Recipe name: {recipe_name}.")

    if description:
        parts.append(f"Description: {description}")

    if ingredients:
        parts.append(f"Ingredients: {join_list(ingredients)}.")

    if cuisine_type:
        parts.append(f"Cuisine type: {join_list(cuisine_type)}.")

    if meal_type:
        parts.append(f"Meal type: {join_list(meal_type)}.")

    if dish_type:
        parts.append(f"Dish type: {join_list(dish_type)}.")

    if diet_labels:
        parts.append(f"Diet labels: {join_list(diet_labels)}.")

    if health_labels:
        parts.append(f"Health labels: {join_list(health_labels)}.")

    if cautions:
        parts.append(f"Cautions or possible allergens: {join_list(cautions)}.")

    if taste_words:
        parts.append(f"Flavor profile: {join_list(taste_words)}.")

    if tags:
        parts.append(f"Tags: {tags}.")

    return " ".join(parts)


df = pd.read_csv(INPUT_CSV)

# This overwrites the existing embedding_text column.
# It does not create a second embedding_text column.
df["embedding_text"] = df.apply(build_embedding_text, axis=1)

df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved updated file to: {OUTPUT_CSV}")
print()
print("Example updated embedding text:")
print(df.loc[0, "embedding_text"])