# retrain_simple.py
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------
# CONFIG - edit these
# -----------------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
BEST_RUN_ID = "913820e4baae43c7902632227fbafff0"   # <--- set your best run id
DATA_PATH = "walmart_dataset_processed (1).csv"    # <--- set your full dataset path
TARGET_COL = "Weekly_Sales"                        # <--- target column name
DATE_COL = "Date"                                  # <--- drop this if present, else set to None
ARTIFACT_NAME = "model"                            # <--- common artifact name logged (e.g., "model" or "sklearn_pipeline")
RANDOM_STATE = 42
# -----------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- Load data ---
df = pd.read_csv(DATA_PATH)
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {df.columns.tolist()}")

drop_cols = [TARGET_COL]
if DATE_COL and DATE_COL in df.columns:
    drop_cols.append(DATE_COL)

X = df.drop(columns=drop_cols)
y = df[TARGET_COL]

# Train / val / test split: 70 / 20 / 10
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.3333333, random_state=RANDOM_STATE)

print(f"Sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

# --- Try to load a trainable sklearn model artifact from MLflow ---
sklearn_model = None
load_attempts = [
    f"runs:/{BEST_RUN_ID}/{ARTIFACT_NAME}",
    f"runs:/{BEST_RUN_ID}/sklearn_pipeline",
    f"runs:/{BEST_RUN_ID}/model"
]

for uri in load_attempts:
    try:
        print(f"Trying to load sklearn model from: {uri}")
        sklearn_model = mlflow.sklearn.load_model(uri)
        print(f"Loaded sklearn model from: {uri}")
        break
    except Exception as e:
        print(f"Could not load sklearn model from {uri}: {e}")

# If sklearn model not found, try pyfunc (predict-only)
if sklearn_model is None:
    try:
        pyfunc_uri = f"runs:/{BEST_RUN_ID}/{ARTIFACT_NAME}"
        print(f"Trying to load pyfunc model from: {pyfunc_uri}")
        pyfunc_model = mlflow.pyfunc.load_model(pyfunc_uri)
        print("Loaded pyfunc model (predict-only).")
        # Test predict on validation set to see if preprocessing is baked in
        try:
            preds = pyfunc_model.predict(X_val.head(5))
            print("pyfunc.predict() on a small sample succeeded -> preprocessing MAY be included in the saved model.")
            val_preds = pyfunc_model.predict(X_val)
            val_rmse = mean_squared_error(y_val, val_preds, squared=False)
            val_mae = mean_absolute_error(y_val, val_preds)
            val_r2 = r2_score(y_val, val_preds)
            print(f"Validation (pyfunc) -> RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}")
        except Exception as e:
            print("pyfunc model could NOT predict on raw validation rows. That indicates preprocessing is NOT baked in.")
            print("Prediction error:", e)

        raise RuntimeError(
            "The loaded model is a pyfunc (predict-only). You cannot call fit() on this object. "
            "To retrain: either (1) log & load the original sklearn pipeline artifact (mlflow.sklearn.log_model), "
            "or (2) rebuild the preprocessing + estimator in code with the same hyperparams and call fit() on that pipeline."
        )

    except RuntimeError as re:
        # Re-raise to show the message to the user (script stops here)
        raise
    except Exception as e:
        raise RuntimeError(f"Could not load any model artifacts for run {BEST_RUN_ID}. Errors: {e}")

# --- If we have a trainable sklearn model, fit it on full training set ---
if sklearn_model is not None:
    print("Training the sklearn model on the training set...")
    try:
        sklearn_model.fit(X_train, y_train)
    except Exception as e:
        raise RuntimeError(f"Error during fit(). Likely mismatch in expected input features or missing preprocessing. Original error: {e}")

    # Evaluate
    y_val_pred = sklearn_model.predict(X_val)
    y_test_pred = sklearn_model.predict(X_test)

    #val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    #test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("Validation metrics:")
    print(f"MAE: {val_mae:.4f}, R2: {val_r2:.4f}")

    print("Test metrics:")
    print(f"MAE: {test_mae:.4f}, R2: {test_r2:.4f}")

    # Log retrained model back to MLflow
    print("Logging retrained model to MLflow...")
    with mlflow.start_run(run_name="retrain_full_dataset"):
        #mlflow.log_metric("val_rmse", float(val_rmse))
        mlflow.log_metric("val_mae", float(val_mae))
        mlflow.log_metric("val_r2", float(val_r2))
        #mlflow.log_metric("test_rmse", float(test_rmse))
        mlflow.log_metric("test_mae", float(test_mae))
        mlflow.log_metric("test_r2", float(test_r2))
        mlflow.sklearn.log_model(sklearn_model, artifact_path="retrained_model")
    print("Retraining complete and model logged to MLflow.")
