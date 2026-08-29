import ast
import json
import re
import pandas as pd


# -----------------------------
# Config
# -----------------------------

INPUT_CSV = "data/foodsfactsfinal.csv"
OUTPUT_CSV = "data/cleanfood.csv"
AUDIT_CSV = "data/label_cleanup_audit.csv"

ID_COLUMN = "food_id"


# Only check fields that describe the actual recipe/ingredients.
# Do NOT include tags, short_description, or embedding_text here because they may contain
# words like "vegan", "dairy-free", "egg-free", etc.
TEXT_COLUMNS_TO_CHECK = [
    "recipe_name",
    "ingredients",
]


# -----------------------------
# Conflict terms
# -----------------------------

MEAT_AND_SEAFOOD_TERMS = [
    "chicken", "beef", "pork", "bacon", "turkey", "ham", "sausage",
    "lamb", "veal", "duck", "goose", "steak", "meatball", "meatballs",
    "salmon", "tuna", "shrimp", "prawn", "prawns", "crab", "lobster",
    "fish", "shellfish", "anchovy", "anchovies", "prosciutto",
    "pepperoni", "chorizo", "brisket", "ribs", "ribeye", "sirloin",
    "oyster", "oysters", "clam", "clams", "mussel", "mussels",
    "scallop", "scallops", "cod", "tilapia", "trout", "halibut",
    "mahi mahi", "mahimahi", "catfish", "sardine", "sardines",
]

DAIRY_TERMS = [
    "milk", "cheese", "butter", "cream", "yogurt", "yoghurt",
    "parmesan", "mozzarella", "cheddar", "feta", "ricotta",
    "sour cream", "half-and-half", "half and half", "buttermilk",
    "ghee", "whey", "casein", "mascarpone", "goat cheese",
    "cream cheese", "heavy cream", "whipping cream",
]

EGG_TERMS = [
    "egg", "eggs", "egg yolk", "egg yolks", "egg white", "egg whites",
    "mayonnaise", "mayo",
]

GLUTEN_WHEAT_TERMS = [
    "wheat", "flour", "bread", "breadcrumbs", "panko", "pasta",
    "spaghetti", "linguine", "fettuccine", "noodles", "barley",
    "rye", "seitan", "couscous",
]

PEANUT_TERMS = [
    "peanut", "peanuts", "peanut butter",
]

TREE_NUT_TERMS = [
    "almond", "almonds", "walnut", "walnuts", "cashew", "cashews",
    "pecan", "pecans", "pistachio", "pistachios", "hazelnut",
    "hazelnuts", "macadamia", "pine nut", "pine nuts",
]

SOY_TERMS = [
    "soy", "soy sauce", "tofu", "tempeh", "miso", "edamame",
]


# If these terms are found in recipe_name or ingredients,
# remove these labels from health_labels.
CONFLICT_RULES = [
    {
        "name": "meat_or_seafood",
        "terms": MEAT_AND_SEAFOOD_TERMS,
        "remove_labels": ["Vegan", "Vegetarian"],
    },
    {
        "name": "dairy",
        "terms": DAIRY_TERMS,
        "remove_labels": ["Vegan", "Dairy-Free"],
    },
    {
        "name": "egg",
        "terms": EGG_TERMS,
        "remove_labels": ["Vegan", "Egg-Free"],
    },
    {
        "name": "gluten_or_wheat",
        "terms": GLUTEN_WHEAT_TERMS,
        "remove_labels": ["Gluten-Free", "Wheat-Free"],
    },
    {
        "name": "peanut",
        "terms": PEANUT_TERMS,
        "remove_labels": ["Peanut-Free"],
    },
    {
        "name": "tree_nut",
        "terms": TREE_NUT_TERMS,
        "remove_labels": ["Tree-Nut-Free"],
    },
    {
        "name": "soy",
        "terms": SOY_TERMS,
        "remove_labels": ["Soy-Free"],
    },
]


# When a health label is removed, remove matching tag words too.
LABEL_TO_TAGS = {
    "Vegan": ["vegan"],
    "Vegetarian": ["vegetarian"],
    "Dairy-Free": ["dairy-free", "dairy free"],
    "Egg-Free": ["egg-free", "egg free"],
    "Gluten-Free": ["gluten-free", "gluten free"],
    "Wheat-Free": ["wheat-free", "wheat free"],
    "Fish-Free": ["fish-free", "fish free"],
    "Shellfish-Free": ["shellfish-free", "shellfish free"],
    "Peanut-Free": ["peanut-free", "peanut free"],
    "Tree-Nut-Free": ["tree-nut-free", "tree nut free", "tree-nut free"],
    "Soy-Free": ["soy-free", "soy free"],
}


TASTE_COLUMNS = {
    "sweetness": "sweet",
    "saltiness": "salty",
    "sourness": "sour",
    "bitterness": "bitter",
    "savoriness": "savory",
    "fattiness": "rich or fatty",
    "spiciness": "spicy",
}


# -----------------------------
# Helpers
# -----------------------------

def is_missing(value) -> bool:
    return value is None or pd.isna(value)


def clean_text(value) -> str:
    if is_missing(value):
        return ""
    return str(value).strip()


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if is_missing(value):
        return False

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_list(value) -> list[str]:
    if is_missing(value):
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    return [x.strip() for x in text.split(",") if x.strip()]


def json_list(items: list[str]) -> str:
    return json.dumps(items, ensure_ascii=False)


def remove_labels(labels: list[str], labels_to_remove: list[str]) -> list[str]:
    return [label for label in labels if label not in labels_to_remove]


def row_text_for_validation(row: pd.Series) -> str:
    parts = []

    for col in TEXT_COLUMNS_TO_CHECK:
        if col in row and not is_missing(row[col]):
            parts.append(str(row[col]))

    return " ".join(parts).lower()


def term_matches(text: str, term: str) -> bool:
    """
    Uses word boundaries so 'ham' does not match 'shampoo'.
    Also allows spaces/hyphens for multiword terms like 'half-and-half'.
    """
    escaped = re.escape(term.lower())
    escaped = escaped.replace(r"\ ", r"[\s\-]+")
    pattern = r"\b" + escaped + r"\b"
    return re.search(pattern, text) is not None


def find_matching_terms(text: str, terms: list[str]) -> list[str]:
    matches = []

    for term in terms:
        if term_matches(text, term):
            matches.append(term)

    return matches


def clean_tags(tags_value, removed_labels: list[str]) -> str:
    if is_missing(tags_value):
        return ""

    tags = [tag.strip() for tag in str(tags_value).split(",") if tag.strip()]

    remove_terms = set()

    for label in removed_labels:
        for tag_term in LABEL_TO_TAGS.get(label, []):
            remove_terms.add(tag_term.lower())

    cleaned_tags = [
        tag for tag in tags
        if tag.lower() not in remove_terms
    ]

    return ", ".join(cleaned_tags)


def parse_ingredients_for_embedding(value) -> list[str]:
    """
    Attempts to make ingredients readable in embedding_text.
    Handles strings, JSON list of strings, or JSON list of dicts with 'food'/'text'.
    """
    if is_missing(value):
        return []

    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip()

        if not text:
            return []

        try:
            items = json.loads(text)
        except Exception:
            try:
                items = ast.literal_eval(text)
            except Exception:
                return [text]

    ingredients = []

    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                food = item.get("food") or item.get("text")
                if food:
                    ingredients.append(str(food).strip())
            elif isinstance(item, str):
                ingredients.append(item.strip())
    elif isinstance(items, dict):
        food = items.get("food") or items.get("text")
        if food:
            ingredients.append(str(food).strip())

    # Deduplicate while preserving order
    seen = set()
    cleaned = []

    for ingredient in ingredients:
        key = ingredient.lower()
        if ingredient and key not in seen:
            seen.add(key)
            cleaned.append(ingredient)

    return cleaned


def join_list(items: list[str]) -> str:
    return ", ".join([str(x).strip() for x in items if str(x).strip()])


def build_embedding_text(row: pd.Series) -> str:
    recipe_name = clean_text(row.get("recipe_name"))
    description = clean_text(row.get("short_description"))
    tags = clean_text(row.get("tags"))

    ingredients = parse_ingredients_for_embedding(row.get("ingredients"))

    diet_labels = parse_list(row.get("diet_labels"))
    health_labels = parse_list(row.get("health_labels"))
    cautions = parse_list(row.get("cautions"))
    cuisine_type = parse_list(row.get("cuisine_type"))
    meal_type = parse_list(row.get("meal_type"))
    dish_type = parse_list(row.get("dish_type"))

    taste_words = [
        taste_word
        for column, taste_word in TASTE_COLUMNS.items()
        if parse_bool(row.get(column))
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

    return " ".join(parts).strip()


# -----------------------------
# Row cleaning
# -----------------------------

def clean_row(row: pd.Series) -> pd.Series:
    validation_text = row_text_for_validation(row)

    original_health_labels = parse_list(row.get("health_labels"))
    health_labels = original_health_labels.copy()

    removed_labels = []
    cleanup_reasons = []

    for rule in CONFLICT_RULES:
        matched_terms = find_matching_terms(validation_text, rule["terms"])

        if not matched_terms:
            continue

        labels_to_remove = [
            label for label in rule["remove_labels"]
            if label in health_labels
        ]

        if not labels_to_remove:
            continue

        health_labels = remove_labels(health_labels, labels_to_remove)

        removed_labels.extend(labels_to_remove)

        cleanup_reasons.append({
            "rule": rule["name"],
            "matched_terms": matched_terms,
            "removed_labels": labels_to_remove,
        })

    # Deduplicate removed labels while preserving order
    seen = set()
    removed_labels_unique = []

    for label in removed_labels:
        if label not in seen:
            seen.add(label)
            removed_labels_unique.append(label)

    # Update health_labels
    row["health_labels"] = json_list(health_labels)

    # Remove matching tags if health labels were removed
    if removed_labels_unique:
        row["tags"] = clean_tags(row.get("tags"), removed_labels_unique)

    # Rebuild embedding_text after health_labels and tags are corrected
    row["embedding_text"] = build_embedding_text(row)

    # Add audit/debug columns
    row["original_health_labels"] = json_list(original_health_labels)
    row["removed_bad_health_labels"] = json_list(removed_labels_unique)
    row["label_cleanup_reasons"] = json.dumps(cleanup_reasons, ensure_ascii=False)

    return row


# -----------------------------
# Main
# -----------------------------

def main():
    df = pd.read_csv(INPUT_CSV)

    required_columns = [
        ID_COLUMN,
        "recipe_name",
        "health_labels",
        "ingredients",
        "tags",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded rows: {len(df)}")
    print("Cleaning health labels, tags, and embedding_text...")

    df = df.apply(clean_row, axis=1)

    changed = df[df["removed_bad_health_labels"] != "[]"].copy()

    df.to_csv(OUTPUT_CSV, index=False)
    changed.to_csv(AUDIT_CSV, index=False)

    print()
    print(f"Saved cleaned CSV to: {OUTPUT_CSV}")
    print(f"Saved audit CSV to: {AUDIT_CSV}")
    print(f"Rows with labels removed: {len(changed)}")

    if len(changed) > 0:
        print()
        print("Sample changed rows:")
        print(
            changed[
                [
                    ID_COLUMN,
                    "recipe_name",
                    "health_labels",
                    "tags",
                    "removed_bad_health_labels",
                    "label_cleanup_reasons",
                ]
            ].head(25)
        )


if __name__ == "__main__":
    main()