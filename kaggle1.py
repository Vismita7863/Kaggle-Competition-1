# ============================================================================
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ----------------------------
# 1. Load Data
# ----------------------------
train_df = pd.read_csv("/kaggle/input/iisc-umc-301-kaggle-competition-1/train.csv")
test_df = pd.read_csv("/kaggle/input/iisc-umc-301-kaggle-competition-1/test.csv")

TARGET = "song_popularity"
X = train_df.drop(columns=[TARGET, "id"], errors='ignore')
y = train_df[TARGET].astype(float)
X_test_final = test_df.drop(columns=['id'], errors='ignore')

# ----------------------------
# 2. Preprocessing Pipeline
# ----------------------------
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in ["key", "audio_mode", "time_signature"] if c in X.columns]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), 
                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols)
], remainder="drop")

# ----------------------------
# 3. Model Pipeline (Enhanced)
# ----------------------------
model = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(
        n_estimators=100,       
        max_depth=12,           
        max_features="sqrt",    
        min_samples_leaf=3,     
        random_state=42,
        n_jobs=-1
    ))
])

# ----------------------------
# 4. Train the Model
# ----------------------------
print("\n--- Training Enhanced Model ---")
model.fit(X, y)
print("✅ Model Trained")

# ----------------------------
# 5. Predict on Test Set
# ----------------------------
preds_final = np.clip(model.predict(X_test_final), 0, 1)

# ----------------------------
# 6. Create Submission
# ----------------------------
submission_df = pd.DataFrame({"id": test_df["id"], "song_popularity": preds_final})
submission_path = Path("/kaggle/working/submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"\n✅ Submission file created at {submission_path}")
display(submission_df.head())