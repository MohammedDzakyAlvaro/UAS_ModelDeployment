"""
main_pipeline.py
Menjalankan seluruh pipeline secara berurutan:
    Step 1: Data Ingestion
    Step 2: Preprocessing
    Step 3: Training (dengan MLflow logging)
"""

from data_ingestion import ingest_data
from preprocess import CreditScorePreprocessor
from train import CreditScoreTrainer


def run_pipeline():
    print("\n" + "=" * 55)
    print("   CREDIT SCORE CLASSIFICATION — ML PIPELINE")
    print("=" * 55)

    print("\n[STEP 1] DATA INGESTION")
    print("-" * 40)
    ingest_data()

    print("\n[STEP 2] PREPROCESSING")
    print("-" * 40)
    preprocessor = CreditScorePreprocessor()
    preprocessor.run(data_path="ingested/data_C.csv")

    print("\n[STEP 3] TRAINING & MLFLOW LOGGING")
    print("-" * 40)
    trainer = CreditScoreTrainer()
    trainer.train()

    print("\n" + "=" * 55)
    print("   PIPELINE COMPLETE")
    print("=" * 55)
    print("Artifacts saved in: ./artifacts/")
    print("  - best_model.pkl")
    print("  - preprocess_config.pkl")
    print("  - train_processed.csv")
    print("  - test_processed.csv")
    print("\nMLflow UI: run  mlflow ui  then open http://localhost:5000")


if __name__ == "__main__":
    run_pipeline()
