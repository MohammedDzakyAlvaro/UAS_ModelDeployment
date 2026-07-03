"""
inference.py
Inference module untuk Credit Score Classification.
Digunakan oleh app.py (Streamlit) maupun dipanggil langsung.
"""

import os
import numpy as np
import pandas as pd
import joblib
from preprocess import parse_credit_history_age

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────
# INFERENCE CLASS
# ─────────────────────────────────────────────

class CreditScoreInference:
    """
    Load model artifacts dan lakukan prediksi credit score.

    Attributes:
        model   : sklearn Pipeline (preprocessor + model)
        config  : dict berisi feature lists dan metadata
    """

    LABEL_MAP = {
        'Good':     'Good',
        'Standard': 'Standard',
        'Poor':     'Poor',
    }

    NOISE_REPLACEMENTS = {
        'Payment_Behaviour': {'!@9#%8': np.nan},
        'Credit_Mix':        {'_':      np.nan},
        'Occupation':        {'_______': np.nan},
        'Payment_of_Min_Amount': {'NM': np.nan},
    }

    CLIP_BOUNDS = {
        'Interest_Rate':      (None, 100),
        'Num_Bank_Accounts':  (0, 20),
        'Num_Credit_Card':    (0, 20),
        'Num_of_Loan':        (0, 20),
        'Delay_from_due_date':(0, 100),
    }

    def __init__(self,
                 model_path: str  = None,
                 config_path: str = None):

        model_path  = model_path  or os.path.join(BASE_DIR, "artifacts/best_model.pkl")
        config_path = config_path or os.path.join(BASE_DIR, "artifacts/preprocess_config.pkl")

        self.model  = joblib.load(model_path)
        self.config = joblib.load(config_path)

    # ------------------------------------------------------------------
    def _clean_input(self, data: dict) -> dict:
        """Apply same cleaning as training preprocessing."""
        data = dict(data)

        # Parse Credit_History_Age if it's still a string
        if 'Credit_History_Age' in data:
            val = data['Credit_History_Age']
            if isinstance(val, str):
                data['Credit_History_Age'] = parse_credit_history_age(val)

        # Replace noise values
        for col, mapping in self.NOISE_REPLACEMENTS.items():
            if col in data and data[col] in mapping:
                data[col] = mapping[data[col]]

        # Clip
        for col, (lo, hi) in self.CLIP_BOUNDS.items():
            if col in data and data[col] is not None:
                v = float(data[col])
                if lo is not None:
                    v = max(v, lo)
                if hi is not None:
                    v = min(v, hi)
                data[col] = v

        return data

    # ------------------------------------------------------------------
    def predict(self, input_data: dict) -> dict:
        """
        Predict credit score for a single input.

        Args:
            input_data: dict with feature values

        Returns:
            dict with keys: label, label_display, probabilities (if available)
        """
        input_data = self._clean_input(input_data)

        all_features = (
            self.config['numerical_features'] +
            self.config['categorical_features']
        )

        # Fill missing features with NaN
        for col in all_features:
            if col not in input_data:
                input_data[col] = np.nan

        df = pd.DataFrame([input_data])[all_features]

        label = self.model.predict(df)[0]

        result = {
            'label':         label,
            'label_display': self.LABEL_MAP.get(label, label),
        }

        # Try to get probabilities
        if hasattr(self.model, 'predict_proba'):
            try:
                proba = self.model.predict_proba(df)[0]
                classes = self.model.classes_
                result['probabilities'] = {
                    cls: float(round(p, 4))
                    for cls, p in zip(classes, proba)
                }
            except Exception:
                pass

        return result


