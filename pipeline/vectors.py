import os
import ast
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import chromadb
from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHROMA_PATH, COLLECTION_NAME, DATAFOOD_CSV

INPUT_CSV = DATAFOOD_CSV

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

ID_COLUMN = "food_id"
EMBEDDING_TEXT_COLUMN = "embedding_text"

METADATA_COLUMNS = [
    "recipe_name",
    "diet_labels",
    "health_labels",
    "cautions",
    "cuisine_type",
    "meal_type",
    "dish_type",
    "ingredients",
    "short_description",
    "tags",
    "sweetness",
    "saltiness",
    "sourness",
    "bitterness",
    "savoriness",
    "fattiness",
    "spiciness",
]


LIST_LIKE_COLUMNS = {
    "diet_labels",
    "health_labels",
    "cautions",
    "cuisine_type",
    "meal_type",
    "dish_type",
}

BOOLEAN_COLUMNS = {
    "sweetness",
    "saltiness",
    "sourness",
    "bitterness",
    "savoriness",
    "fattiness",
    "spiciness",
}



client = OpenAI()


# -----------------------------
# Parsing helpers
# -----------------------------

def is_missing(value: Any) -> bool:
    return value is None or pd.isna(value)


def clean_text(value: Any) -> str:
    if is_missing(value):
        return ""
    return str(value).strip()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if is_missing(value):
        return False

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_list(value: Any) -> list[str]:
    """
    Handles:
    - Python lists
    - JSON strings like ["Vegan", "Vegetarian"]
    - Python-style strings like ['Vegan', 'Vegetarian']
    - comma-separated strings
    """
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


def parse_json_or_text(value: Any) -> str:
    """
    Chroma metadata should be scalar values.
    For ingredients, we store a compact string version.
    """
    if is_missing(value):
        return ""

    text = str(value).strip()

    if not text:
        return ""

    try:
        parsed = json.loads(text)
        return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        return text



# -----------------------------
# Metadata builder
# -----------------------------

def build_metadata(row: pd.Series) -> dict:
    health_labels = parse_list(row.get("health_labels"))
    diet_labels = parse_list(row.get("diet_labels"))
    cautions = parse_list(row.get("cautions"))
    cuisine_type = parse_list(row.get("cuisine_type"))
    meal_type = parse_list(row.get("meal_type"))
    dish_type = parse_list(row.get("dish_type"))

    metadata = {
        "food_id": str(row.get(ID_COLUMN)),
        "recipe_name": clean_text(row.get("recipe_name")),
        "short_description": clean_text(row.get("short_description")),
        "tags": clean_text(row.get("tags")),
        "ingredients": parse_json_or_text(row.get("ingredients")),

        # Taste metadata
        "sweetness": parse_bool(row.get("sweetness")),
        "saltiness": parse_bool(row.get("saltiness")),
        "sourness": parse_bool(row.get("sourness")),
        "bitterness": parse_bool(row.get("bitterness")),
        "savoriness": parse_bool(row.get("savoriness")),
        "fattiness": parse_bool(row.get("fattiness")),
        "spiciness": parse_bool(row.get("spiciness")),
    }

    # Add list metadata only if the list is non-empty.
    # Chroma rejects empty metadata lists.
    if diet_labels:
        metadata["diet_labels"] = diet_labels

    if health_labels:
        metadata["health_labels"] = health_labels

    if cautions:
        metadata["cautions"] = cautions

    if cuisine_type:
        metadata["cuisine_type"] = cuisine_type

    if meal_type:
        metadata["meal_type"] = meal_type

    if dish_type:
        metadata["dish_type"] = dish_type

    return metadata


# -----------------------------
# Embedding helper
# -----------------------------

def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]



df = pd.read_csv(INPUT_CSV)

required_columns = [ID_COLUMN, EMBEDDING_TEXT_COLUMN]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df[EMBEDDING_TEXT_COLUMN] = df[EMBEDDING_TEXT_COLUMN].fillna("").astype(str)
df = df[df[EMBEDDING_TEXT_COLUMN].str.strip() != ""].copy()

# Ensure IDs are strings and unique
df[ID_COLUMN] = df[ID_COLUMN].astype(str)

if df[ID_COLUMN].duplicated().any():
    duplicate_ids = df[df[ID_COLUMN].duplicated()][ID_COLUMN].tolist()[:10]
    raise ValueError(f"Duplicate food_id values found. Examples: {duplicate_ids}")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

total_rows = len(df)
total_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Starting embedding/loading process...")
print(f"Total rows: {total_rows}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Total batches: {total_batches}")
print()

for batch_number, start in enumerate(
    tqdm(range(0, total_rows, BATCH_SIZE), desc="Embedding and loading"),
    start=1
):
    end = min(start + BATCH_SIZE, total_rows)
    batch = df.iloc[start:end]

    print(f"\nBatch {batch_number}/{total_batches}")
    print(f"Processing rows {start + 1} to {end}")

    ids = batch[ID_COLUMN].astype(str).tolist()
    documents = batch[EMBEDDING_TEXT_COLUMN].tolist()
    metadatas = [build_metadata(row) for _, row in batch.iterrows()]

    print("Creating embeddings...")

    for attempt in range(3):
        try:
            embeddings = get_embeddings(documents)
            break
        except Exception as exc:
            if attempt == 2:
                raise
            wait_seconds = 2 ** attempt
            print(f"Embedding failed: {exc}. Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

    print("Saving batch to ChromaDB...")

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Finished batch {batch_number}/{total_batches}")
    print(f"Rows completed: {end}/{total_rows}")
    print(f"Records currently stored: {collection.count()}")

print()
print("Done.")
print(f"Stored {collection.count()} records in ChromaDB.")