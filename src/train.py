import pandas as pd
import os

from src.data import load_adult_data
from src.preprocessing import get_train_test_splits
from src.models import get_models
from src.metrics import evaluate_model

# Load dataset (UCI or local)
df = load_adult_data(source="uci")

# Get train/test splits with preprocessing
X_train, X_test, y_train, y_test = get_train_test_splits(df, target_column="income")

# Save train/test splits for reproducibility
os.makedirs("data", exist_ok=True)
train_df = pd.concat([pd.DataFrame(X_train, columns=X_train.columns), y_train.reset_index(drop=True)], axis=1)
test_df = pd.concat([pd.DataFrame(X_test, columns=X_test.columns), y_test.reset_index(drop=True)], axis=1)
train_df.to_csv("data/train_split.csv", index=False)
test_df.to_csv("data/test_split.csv", index=False)

# Train models
models = get_models()
results = []

os.makedirs("model", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    metrics["Model"] = name
    results.append(metrics)

# Save comparison metrics
df_metrics = pd.DataFrame(results)
df_metrics.to_csv("model/model_comparison_metrics.csv", index=False)
print("Training complete. Metrics saved to model/model_comparison_metrics.csv")
print(df_metrics)