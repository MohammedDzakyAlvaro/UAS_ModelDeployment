from pathlib import Path
import pandas as pd

# Path config
BASE_DIR     = Path(__file__).parent
INGESTED_DIR = BASE_DIR / "ingested"
INPUT_FILE   = BASE_DIR / "data_C.csv"
OUTPUT_FILE  = INGESTED_DIR / "data_C.csv"


def ingest_data():
    """Load raw data and save to ingested folder."""
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"[DATA INGESTION] Success — {len(df)} rows loaded.")
    return OUTPUT_FILE


if __name__ == "__main__":
    ingest_data()
