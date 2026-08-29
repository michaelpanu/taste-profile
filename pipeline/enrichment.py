from openai import OpenAI
import pandas as pd
import json
from tqdm import tqdm
import ast
import time
from pathlib import Path

print("Enriching these files")
print("")

PAUSE_FILE = Path("pause.txt")
STOP_FILE = Path("stop.txt")

client = OpenAI()

df=pd.read_csv("data/recipes-with-nutrition.csv", usecols=["recipe_name", "diet_labels", "health_labels", "cautions", "cuisine_type", "meal_type", "dish_type", "ingredients"])
pd.set_option('display.max_columns', 8)


def check_controls():
    if STOP_FILE.exists():
        print("\nStop requested. Saving and exiting safely...")
        return "stop"

    while PAUSE_FILE.exists():
        print("\rPaused. Delete pause.txt to resume, or create stop.txt to stop safely.", end="", flush=True)
        time.sleep(1)

        if STOP_FILE.exists():
            print("\nStop requested while paused. Saving and exiting safely...")
            return "stop"

    return "continue"

def parse_jsonish(value, default):
    """
    Handles CSV cells that look like:
    ["salad"]
    [{"food":"salt","text":"1 tsp salt"}]
    """
    if pd.isna(value):
        return default

    if isinstance(value, (list, dict)):
        return value

    text = str(value).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return default


def list_to_text(value):
    parsed = parse_jsonish(value, [])

    if isinstance(parsed, list):
        return ", ".join(str(x) for x in parsed)

    return str(parsed)


def ingredients_to_text(value):
    parsed = parse_jsonish(value, [])

    if not isinstance(parsed, list):
        return str(value)

    ingredient_lines = []

    for item in parsed:
        if isinstance(item, dict):
            ingredient_lines.append(item.get("text") or item.get("food") or "")
        else:
            ingredient_lines.append(str(item))

    return "; ".join(x for x in ingredient_lines if x)


def enrich_recipe(row):
    prompt = f"""
You are enriching a recipe dataset for a food recommendation app.

Given the recipe data, return:
- A short description, max 25 words
- 5 to 8 lowercase tags
- Boolean taste labels

The description should focus on describing the taste, texture, and experience of eating this food. Avoid generic phrases like "perfect for lunch or dinner." Do not simply list the ingredients as well. 

Tags must be lowercase kebab-case with no spaces.
Use tags that help recommendation, such as cuisine, dish type, texture, diet, and major flavor.

Taste labels should be true only if that taste is a noticeable part of the eating experience.
Do not mark a taste true merely because a small amount of that ingredient appears.
For example, salt in a dessert does not make saltiness true unless the dessert tastes noticeably salty.

Taste labels:
- sweetness: true if the recipe is noticeably sweet
- saltiness: true if salty/briny ingredients are clearly present
- sourness: true if acidic/tangy ingredients are clearly present
- bitterness: true if bitter ingredients are clearly present
- savoriness: true if umami/savory ingredients are clearly present
- fattiness: true if creamy/oily/fat-rich ingredients are clearly present
- spiciness: true if hot/spicy/chilli heat ingredients are clearly present

Rules:
- Be conservative.
- Do not mark every true unless there is enough substance behind the claim.
- Use the ingredients, cuisine, dish type, and meal type as context.
- Tags should be simple and useful for recommendations.

Recipe:
Name: {row["recipe_name"]}
Cuisine: {list_to_text(row["cuisine_type"])}
Meal type: {list_to_text(row["meal_type"])}
Dish type: {list_to_text(row["dish_type"])}
Diet labels: {list_to_text(row["diet_labels"])}
Health labels: {list_to_text(row["health_labels"])}
Cautions: {list_to_text(row["cautions"])}
Ingredients: {ingredients_to_text(row["ingredients"])}
"""

    response = client.responses.create(
       model="gpt-4o-mini",
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "recipe_enrichment",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "short_description": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 5,
                            "maxItems": 8,
                        },
                        "sweetness": {"type": "boolean"},
                        "saltiness": {"type": "boolean"},
                        "sourness": {"type": "boolean"},
                        "bitterness": {"type": "boolean"},
                        "savoriness": {"type": "boolean"},
                        "fattiness": {"type": "boolean"},
                        "spiciness": {"type": "boolean"},
                    },
                    "required": [
                        "short_description",
                        "tags",
                        "sweetness",
                        "saltiness",
                        "sourness",
                        "bitterness",
                        "savoriness",
                        "fattiness",
                        "spiciness",
                    ],
                },
                "strict": True,
            }
        }
    )

    return json.loads(response.output_text)


KEEP_DISH_TYPES=["main course", "starter", "desserts"]
def contains_any_label(value, allowed_labels):
    text = list_to_text(value).lower()
    return any(label in text for label in allowed_labels)

before_count = len(df)

df = df[
    df["dish_type"].apply(lambda x: contains_any_label(x, KEEP_DISH_TYPES))
].copy()

after_count = len(df)

print(f"Kept {after_count} starter/main-course rows.")
print(f"Removed {before_count - after_count} non-core rows.")


results = []

pbar = tqdm(
    df.iterrows(),
    total=len(df),
    desc="Enriching",
    unit="recipe",
    dynamic_ncols=True,
    bar_format="{desc} | {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} | elapsed {elapsed} | left {remaining} | {rate_fmt} | {postfix}",
)

for index, row in pbar:
    control = check_controls()
    if control == "stop":
        break

    try:
        enriched = enrich_recipe(row)

        results.append({
            "short_description": enriched["short_description"],
            "tags": ", ".join(enriched["tags"]),
            "sweetness": enriched["sweetness"],
            "saltiness": enriched["saltiness"],
            "sourness": enriched["sourness"],
            "bitterness": enriched["bitterness"],
            "savoriness": enriched["savoriness"],
            "fattiness": enriched["fattiness"],
            "spiciness": enriched["spiciness"],
        })

    except Exception as e:
        print(f"\nFailed on row {index}: {e}")

        results.append({
            "short_description": "",
            "tags": "",
            "sweetness": False,
            "saltiness": False,
            "sourness": False,
            "bitterness": False,
            "savoriness": False,
            "fattiness": False,
            "spiciness": False,
        })

    if len(results) % 10 == 0:
        partial_df = pd.concat(
            [df.iloc[:len(results)].reset_index(drop=True), pd.DataFrame(results)],
            axis=1,
        )
        partial_df.to_csv("data/recipes_enriched_partial.csv", index=False)

        pbar.set_postfix_str(
            f"saved index={index}, name={row['recipe_name']}",
            refresh=True
        )
        time.sleep(0.5)

enriched_df = pd.DataFrame(results)
final_df = pd.concat(
    [df.iloc[:len(results)].reset_index(drop=True), enriched_df],
    axis=1,
)

OUTPUT_FILE = "data/test.csv"
final_df.to_csv(OUTPUT_FILE, index=False)

print()
print(f"Saved enriched CSV to {OUTPUT_FILE}")