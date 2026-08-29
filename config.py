import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_food_db_clean")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "foods_cleaned")

DATAFOOD_CSV = "data/datafood.csv"
EVAL_CASES_PATH = "eval/food_eval_cases_references_updated.json"
