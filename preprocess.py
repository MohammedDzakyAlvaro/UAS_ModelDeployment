"""
preprocess.py
Preprocessing pipeline untuk Credit Score Classification (data_C).
OOP-based: class CreditScorePreprocessor
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────

def parse_credit_history_age(val):
    """Convert '9 Years and 8 Months' → 116 (total months)."""
    if pd.isna(val):
        return np.nan
    try:
        parts = str(val).split()
        years  = int(parts[0]) if len(parts) > 2 and 'Year' in parts[2] else 0
        months = int(parts[3]) if len(parts) >= 5 else 0
        return years * 12 + months
    except Exception:
        return np.nan


# ─────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────

class CreditScorePreprocessor:
    """
    Handles all preprocessing steps for the Credit Score dataset.

    Steps:
        1. Drop irrelevant identity columns
        2. Parse Credit_History_Age to numeric (months)
        3. Clean noisy categorical values
        4. Clip extreme outliers
        5. Separate features (X) and target (y)
        6. Identify numerical / categorical feature lists
        7. Train-test split
        8. Save metadata (preprocess_config.pkl)
    """

    DROP_COLS  = ['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN']
    TARGET_COL = 'Credit_Score'

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

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size    = test_size
        self.random_state = random_state

        self.categorical_features = None
        self.numerical_features   = None

    # ------------------------------------------------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop, parse, clean noise, clip outliers."""
        df = df.copy()

        # 1. Drop identity columns
        df = df.drop(columns=[c for c in self.DROP_COLS if c in df.columns])

        # 2. Parse Credit_History_Age
        df['Credit_History_Age'] = df['Credit_History_Age'].apply(parse_credit_history_age)

        # 3. Replace noise strings with NaN
        for col, mapping in self.NOISE_REPLACEMENTS.items():
            if col in df.columns:
                df[col] = df[col].replace(mapping)

        # 3b. Coerce all INTENDED numeric columns to numeric
        #     Many cols have trailing '_' noise e.g. '28_', '43534.9_'
        INTENDED_NUMERIC = [
            'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
            'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
            'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
            'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
            'Amount_invested_monthly', 'Monthly_Balance',
        ]
        for col in INTENDED_NUMERIC:
            if col in df.columns:
                cleaned = df[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True)
                df[col] = pd.to_numeric(cleaned, errors='coerce')

        # 4. Clip outliers
        for col, (lo, hi) in self.CLIP_BOUNDS.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lo, upper=hi)

        return df

    # ------------------------------------------------------------------
    def split_features_target(self, df: pd.DataFrame):
        """Return X, y and identify feature types."""
        X = df.drop(columns=[self.TARGET_COL])
        y = df[self.TARGET_COL]

        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        self.numerical_features   = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        return X, y

    # ------------------------------------------------------------------
    def run(self, data_path: str = "ingested/data_C.csv"):
        """Full preprocessing pipeline: clean → split → save."""
        os.makedirs("artifacts", exist_ok=True)

        df = pd.read_csv(data_path)
        df = self.clean(df)

        X, y = self.split_features_target(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        # Combine and save
        train_df = pd.concat([X_train.reset_index(drop=True),
                               y_train.reset_index(drop=True)], axis=1)
        test_df  = pd.concat([X_test.reset_index(drop=True),
                               y_test.reset_index(drop=True)], axis=1)

        train_df.to_csv("artifacts/train_processed.csv", index=False)
        test_df.to_csv("artifacts/test_processed.csv",  index=False)

        # Save metadata
        config = {
            'categorical_features': self.categorical_features,
            'numerical_features':   self.numerical_features,
            'target':               self.TARGET_COL,
        }
        joblib.dump(config, "artifacts/preprocess_config.pkl")

        print(f"[PREPROCESS] Done.")
        print(f"  Train : {train_df.shape}")
        print(f"  Test  : {test_df.shape}")
        print(f"  Cat features  : {self.categorical_features}")
        print(f"  Num features  : {self.numerical_features}")

        return train_df, test_df


# ─────────────────────────────────────────────
if __name__ == "__main__":
    prep = CreditScorePreprocessor()
    prep.run()
