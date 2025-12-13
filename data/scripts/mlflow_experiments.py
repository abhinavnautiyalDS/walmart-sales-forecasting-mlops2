import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json
import tempfile
import os
from datetime import datetime
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------
# CONFIG
# ---------------------------
DATA_PATH = Path("walmart_dataset_processed (1).csv")
EXPERIMENT_NAME = "walmart_experiments"
RANDOM_STATE = 42

# -------------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    y = df["Weekly_Sales"]
    X = df.drop(columns=["Weekly_Sales"])
    return X, y, df

# -------------------------------------------------------------------
# 2. TRAIN/VALID SPLIT (TIME-BASED + TRAIN SAMPLE)
# -------------------------------------------------------------------
def train_valid_split(df, X, y, valid_ratio=0.2, train_sample_size=100000):
    n = len(df)
    split_idx = int((1 - valid_ratio) * n)
    train_idx = df.index[:split_idx]
    valid_idx = df.index[split_idx:]

    train_df = df.loc[train_idx]
    valid_df = df.loc[valid_idx]

    train_sample = train_df.sample(min(train_sample_size, len(train_df)), random_state=RANDOM_STATE)

    X_train = train_sample.drop(columns=["Weekly_Sales"])
    y_train = train_sample["Weekly_Sales"]

    X_valid = valid_df.drop(columns=["Weekly_Sales"])
    y_valid = valid_df["Weekly_Sales"]

    # drop Date from features just before modeling
    if "Date" in X_train.columns:
        X_train = X_train.drop(columns=["Date"])
    if "Date" in X_valid.columns:
        X_valid = X_valid.drop(columns=["Date"])

    return X_train, X_valid, y_train, y_valid

# -------------------------------------------------------------------
# 3. DEFINE FEATURE SPACES (Safer: don't OHE high-card columns)
# -------------------------------------------------------------------
def get_feature_lists_and_transform(X):
    """
    Strategy:
      - small_categorical: one-hot encode (Type, Store_Size_Category, CPI_Category)
      - high_cardinality: Store and Dept -> convert to category codes (integer labels)
      - numeric: everything else treated numeric
    This function returns numeric_cols (including encoded Store/Dept codes) and categorical_cols (only small ones).
    It also modifies X in-place to convert Store/Dept to integer category codes.
    """
    small_categorical = ["Type", "Store_Size_Category", "CPI_Category"]
    high_cardinality = ["Store", "Dept"]

    # Ensure columns exist and convert high-card columns to category codes
    for col in high_cardinality:
        if col in X.columns:
            X[col] = X[col].astype("category").cat.codes
        else:
            # if column missing, create a placeholder zero column to avoid pipeline issues
            X[col] = 0

    # Ensure small categorical columns exist; if missing, create placeholder
    for col in small_categorical:
        if col not in X.columns:
            X[col] = "missing"

    # numeric columns = all columns except the small category names (we will OHE those)
    numeric_cols = [c for c in X.columns if c not in small_categorical]
    categorical_cols = small_categorical

    return numeric_cols, categorical_cols

# -------------------------------------------------------------------
# 4. BUILD PIPELINE GIVEN CONFIG
# -------------------------------------------------------------------
def build_pipeline(numeric_cols, categorical_cols,
                   scaling: str = "standard",
                   feature_selection: str = "none",
                   model_name: str = "random_forest"):
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = "passthrough"

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,  # helps with get_feature_names_out naming
    )

    if model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif model_name == "gbr":
        model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    elif model_name == "linear":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    steps = [("preprocessor", preprocessor)]

    if feature_selection == "kbest":
        steps.append(("select", SelectKBest(score_func=f_regression, k=30)))

    steps.append(("model", model))
    pipe = Pipeline(steps=steps)
    return pipe

# -------------------------------------------------------------------
# 5. METRICS
# -------------------------------------------------------------------
def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}

# -------------------------------------------------------------------
# Utility to extract final feature names after preprocessing & optional selection
# -------------------------------------------------------------------
def extract_feature_names(pipe, X_sample):
    """
    After fitting pipe, try to extract final feature names:
      - get names from preprocessor (ColumnTransformer + OHE)
      - if SelectKBest present, apply support mask to keep only selected features
    Returns list of feature names (strings).
    """
    feature_names = None
    try:
        preproc = pipe.named_steps["preprocessor"]
        # sklearn >=1.0: get_feature_names_out available
        feature_names = preproc.get_feature_names_out()
        # tidy names: ColumnTransformer prefixes (e.g., 'num__col') are fine
        feature_names = [str(f) for f in feature_names]
    except Exception as e:
        # fallback: create numeric column names plus categorical placeholders
        feature_names = list(X_sample.columns)

    # If feature selection exists, reduce the feature_names
    if "select" in pipe.named_steps:
        try:
            support = pipe.named_steps["select"].get_support()
            feature_names = [n for n, keep in zip(feature_names, support) if keep]
        except Exception:
            # can't map through selection -> leave as-is
            pass

    return feature_names

# -------------------------------------------------------------------
# 6. SINGLE EXPERIMENT RUN with deterministic run name and detailed logging
# -------------------------------------------------------------------
def run_single_experiment(X_train, X_valid, y_train, y_valid,
                          scaling, feature_selection, model_name, run_index=0):
    # prepare features (converts Store/Dept to codes in-place)
    numeric_cols, categorical_cols = get_feature_lists_and_transform(X_train)

    # build pipeline
    pipe = build_pipeline(numeric_cols, categorical_cols, scaling=scaling,
                          feature_selection=feature_selection, model_name=model_name)

    # deterministic run name -> readable and unique with timestamp & index
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_name = f"{model_name}__{scaling}__{feature_selection}__{ts}__run{run_index}"

    with mlflow.start_run(run_name=run_name):
        # tags & params
        mlflow.set_tag("model_type", model_name)
        mlflow.log_param("scaling", scaling)
        mlflow.log_param("feature_selection", feature_selection)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("valid_rows", len(X_valid))
        mlflow.log_param("numeric_cols_count", len(numeric_cols))
        mlflow.log_param("categorical_small_cols", ",".join(categorical_cols))

        # fit
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_valid)

        # evaluate
        metrics = evaluate_regression(y_valid, preds)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # try to log model params (best effort)
        try:
            model_params = pipe.named_steps["model"].get_params()
            # mlflow.log_params may fail for nested dicts/objects — write to artifact instead
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tf:
                json.dump(model_params, tf, default=str, indent=2)
                model_params_file = tf.name
            mlflow.log_artifact(model_params_file, artifact_path="model_params")
            os.remove(model_params_file)
        except Exception:
            pass

        # extract and log final feature names (artifact)
        try:
            # Use a small sample of X_valid for name extraction if needed
            X_sample = X_valid.head(5)
            feature_names = extract_feature_names(pipe, X_sample)
            if feature_names is None:
                feature_names = list(X_sample.columns)

            # write feature names to a file and log as artifact
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tf:
                tf.write("\n".join(feature_names))
                feat_file = tf.name
            mlflow.log_artifact(feat_file, artifact_path="feature_names")
            os.remove(feat_file)

            # also log feature names as a param if small enough
            try:
                mlflow.log_param("n_features", len(feature_names))
            except Exception:
                pass

        except Exception as e:
            print("Warning: could not extract feature names:", e)

        # log pipeline as an MLflow model artifact
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        print(f"Run finished: {run_name} -> RMSE: {metrics['rmse']:.2f}")

        return metrics

# -------------------------------------------------------------------
# 7. MAIN LOOP OVER CONFIGS
# -------------------------------------------------------------------
def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y, df = load_data()
    X_train, X_valid, y_train, y_valid = train_valid_split(
        df, X, y, valid_ratio=0.2, train_sample_size=85000
    )

    scaling_options = ["standard", "minmax"]
    feature_selection_options = ["none", "kbest"]
    model_options = ["random_forest", "gbr", "linear"]

    run_idx = 0
    for model_name in model_options:
        for scaling in scaling_options:
            for feat_sel in feature_selection_options:
                print(f"Running: model={model_name}, scaling={scaling}, feat_sel={feat_sel}")
                run_single_experiment(
                    X_train.copy(), X_valid.copy(), y_train, y_valid,
                    scaling=scaling,
                    feature_selection=feat_sel,
                    model_name=model_name,
                    run_index=run_idx
                )
                run_idx += 1

if __name__ == "__main__":
    main()  

