"""
train.py
Training pipeline untuk Credit Score Classification (data_C).
OOP-based: class CreditScoreTrainer
Menggunakan MLflow untuk logging eksperimen.
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ─────────────────────────────────────────────
# EVALUATION CLASS
# ─────────────────────────────────────────────

class ModelEvaluator:
    """Evaluasi model klasifikasi."""

    @staticmethod
    def evaluate(y_true, y_pred) -> dict:
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average='weighted')
        report = classification_report(y_true, y_pred)
        return {'accuracy': acc, 'f1_weighted': f1, 'report': report}


# ─────────────────────────────────────────────
# TRAINING CLASS
# ─────────────────────────────────────────────

class CreditScoreTrainer:
    """
    Handles full training workflow:
        - Load preprocessed data & config
        - Build sklearn preprocessing pipeline
        - Train multiple classification models
        - Log each run to MLflow
        - Select & save best model
    """

    MODELS = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'xgboost':             XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss'),
    }

    def __init__(self):
        self.best_model       = None
        self.best_model_name  = None
        self.best_score       = -float('inf')
        self.evaluator        = ModelEvaluator()

    # ------------------------------------------------------------------
    def load_data(self):
        train_df = pd.read_csv("artifacts/train_processed.csv")
        test_df  = pd.read_csv("artifacts/test_processed.csv")
        return train_df, test_df

    def load_config(self) -> dict:
        return joblib.load("artifacts/preprocess_config.pkl")

    # ------------------------------------------------------------------
    def build_preprocessor(self, config: dict) -> ColumnTransformer:
        num_features = config['numerical_features']
        cat_features = config['categorical_features']

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
        ])
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])

        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features),
        ])
        return preprocessor

    # ------------------------------------------------------------------
    def train(self):
        os.makedirs("artifacts", exist_ok=True)

        train_df, test_df = self.load_data()
        config            = self.load_config()

        TARGET = config['target']
        X_train = train_df.drop(columns=[TARGET])
        y_train = train_df[TARGET]
        X_test  = test_df.drop(columns=[TARGET])
        y_test  = test_df[TARGET]

        preprocessor = self.build_preprocessor(config)

        experiment_name = "CreditScore_Classification"
        mlflow.set_experiment(experiment_name)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(y_train)

        for name, model in self.MODELS.items():
            with mlflow.start_run(run_name=name):

                pipe = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model',        model),
                ])

                if name == 'xgboost':
                    pipe.fit(X_train, le.transform(y_train))
                    y_pred = le.inverse_transform(pipe.predict(X_test))
                else:
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_test)

                metrics = self.evaluator.evaluate(y_test, y_pred)

                # ── Log Parameters ──────────────────────────────────────
                mlflow.log_param("model_type",       name)
                mlflow.log_param("train_size",       len(X_train))
                mlflow.log_param("test_size",        len(X_test))
                mlflow.log_param("n_features",       X_train.shape[1])
                mlflow.log_param("random_state",     42)
                # log hyperparameter tiap model
                model_params = model.get_params()
                for param_key, param_val in model_params.items():
                    mlflow.log_param(f"hp_{param_key}", param_val)

                # ── Log Metrics ──────────────────────────────────────────
                mlflow.log_metric("accuracy",        metrics['accuracy'])
                mlflow.log_metric("f1_weighted",     metrics['f1_weighted'])

                # precision, recall, f1 per kelas
                from sklearn.metrics import precision_score, recall_score
                for cls in le.classes_:
                    cls_mask_true = (y_test == cls)
                    cls_mask_pred = (y_pred == cls)
                    tp = (cls_mask_true & cls_mask_pred).sum()
                    fp = (~cls_mask_true & cls_mask_pred).sum()
                    fn = (cls_mask_true & ~cls_mask_pred).sum()
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1c  = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
                    mlflow.log_metric(f"precision_{cls}", round(prec, 4))
                    mlflow.log_metric(f"recall_{cls}",    round(rec,  4))
                    mlflow.log_metric(f"f1_{cls}",        round(f1c,  4))

                # ── Log Tags ─────────────────────────────────────────────
                mlflow.set_tag("dataset",            "data_C.csv")
                mlflow.set_tag("task",               "credit_score_classification")
                mlflow.set_tag("best_so_far",        str(metrics['f1_weighted'] > self.best_score))

                # ── Log Artifacts ────────────────────────────────────────
                import tempfile, json
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

                with tempfile.TemporaryDirectory() as tmpdir:
                    # 1. classification report sebagai .txt
                    report_path = os.path.join(tmpdir, "classification_report.txt")
                    with open(report_path, "w") as rf:
                        rf.write(f"Model: {name}\n")
                        rf.write("="*50 + "\n")
                        rf.write(metrics['report'])
                    mlflow.log_artifact(report_path)

                    # 2. confusion matrix sebagai gambar .png
                    cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
                    disp.plot(ax=ax, colorbar=False)
                    ax.set_title(f"Confusion Matrix — {name}")
                    cm_path = os.path.join(tmpdir, "confusion_matrix.png")
                    fig.savefig(cm_path, bbox_inches="tight")
                    plt.close(fig)
                    mlflow.log_artifact(cm_path)

                # ── Log Model ────────────────────────────────────────────
                mlflow.sklearn.log_model(
                    pipe,
                    "model",
                    registered_model_name=f"CreditScore_{name}",
                    
                )
                print(f"\n[{name}]")
                print(f"  Accuracy      : {metrics['accuracy']:.4f}")
                print(f"  F1 (weighted) : {metrics['f1_weighted']:.4f}")
                print(metrics['report'])

                # Track best model (by F1 weighted)
                if metrics['f1_weighted'] > self.best_score:
                    self.best_score      = metrics['f1_weighted']
                    self.best_model      = pipe
                    self.best_model_name = name

        # Save best model
        joblib.dump(self.best_model, "artifacts/best_model.pkl")

        # Update config with best model name
        config['best_model_name'] = self.best_model_name
        joblib.dump(config, "artifacts/preprocess_config.pkl")

        print("\n" + "=" * 50)
        print(f"[BEST MODEL] {self.best_model_name}")
        print(f"             F1 (weighted) = {self.best_score:.4f}")
        print("Saved: artifacts/best_model.pkl")
        print("=" * 50)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    trainer = CreditScoreTrainer()
    trainer.train()