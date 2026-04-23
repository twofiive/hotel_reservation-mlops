import os

# ── Ingestion ──
RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")
CONFIG_PATH = "config/config.yaml"

# ── OpenData ──
OPENDATA_DIR = "artifacts/opendata"
OPENDATA_FILE_PATH = os.path.join(OPENDATA_DIR, "tourisme_insee.csv")
OPENDATA_ENRICHED_PATH = os.path.join(OPENDATA_DIR, "hotel_enriched.csv")

# ── Processing ──
PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_test.csv")

# ── Modèle ──
MODEL_OUTPUT_PATH = "artifacts/models/lgbm_model.pkl"