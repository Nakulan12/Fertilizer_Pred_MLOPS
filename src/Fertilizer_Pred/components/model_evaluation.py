import pandas as pd
from Fertilizer_Pred import logger
from Fertilizer_Pred.entity.config_entity import ModelEvaluationConfig
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
from urllib.parse import urlparse
from sklearn.preprocessing import OrdinalEncoder
from typing import List
import joblib
from pathlib import Path
from src.Fertilizer_Pred.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config):
        self.config = config

    def map_at_k(self, actual: np.ndarray, predicted: np.ndarray, k: int = 3) -> float:
        total_score = 0.0
        for true_idx, pred_top_k in zip(actual, predicted):
            if true_idx in pred_top_k[:k]:
                rank = np.where(pred_top_k[:k] == true_idx)[0][0] + 1
                total_score += 1.0 / rank
        return total_score / len(actual)

    def _load_models(self) -> List[xgb.Booster]:
        model_paths = list(self.config.model_dir.glob("xgb_fold*.bin"))
        models = []
        for path in model_paths:
            model = xgb.Booster()
            model.load_model(str(path))
            models.append(model)
        return models

    def evaluate_ensemble(self):
        test_data = pd.read_csv(self.config.test_data_path)

        # Load label encoder
        label_encoder_path = self.config.label_encoder_path
        if not label_encoder_path.exists():
            raise FileNotFoundError(f"Label encoder not found at: {label_encoder_path}")
        le = joblib.load(label_encoder_path)

        # Extract X_test and optionally y_test
        if self.config.target_column in test_data.columns:
            y_test = test_data[self.config.target_column]
            y_test_encoded = le.transform(y_test)
            X_test = test_data.drop(columns=[self.config.target_column])
        else:
            y_test_encoded = None
            X_test = test_data.copy()

        # Ordinal encode object columns
        obj_cols = X_test.select_dtypes(include='object').columns
        if len(obj_cols) > 0:
            X_test[obj_cols] = OrdinalEncoder().fit_transform(X_test[obj_cols])

        dtest = xgb.DMatrix(X_test)

        models = self._load_models()
        pred_probs = np.mean([model.predict(dtest) for model in models], axis=0)
        top3_preds = np.argsort(-pred_probs, axis=1)[:, :3]

        if y_test_encoded is not None:
            map3 = self.map_at_k(y_test_encoded, top3_preds)
            acc = np.mean(y_test_encoded == top3_preds[:, 0])
            return {"MAP@3": map3, "accuracy": acc}
        else:
            pred_labels = le.inverse_transform(np.argmax(pred_probs, axis=1))
            pd.DataFrame(pred_labels, columns=["Predicted Fertilizer"]).to_csv(
                self.config.root_dir / "predictions.csv", index=False
            )
            return {"status": "Prediction complete. Evaluation skipped (no true labels)."}

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Evaluation
            metrics = self.evaluate_ensemble()
            save_json(path=self.config.metric_file_name, data=metrics)

            # Log hyperparameters
            mlflow.log_params(self.config.all_params)

            # Log metrics
            for key, value in metrics.items():
                try:
                    mlflow.log_metric(key, float(value))
                except Exception as e:
                    print(f"[MLflow] Failed to log metric {key}: {e}")

            # Log label encoder
            if self.config.label_encoder_path.exists():
                mlflow.log_artifact(str(self.config.label_encoder_path), artifact_path="label_encoder")

            # Log each model
            if tracking_url_type_store != "file":
                for i, model_path in enumerate(self.config.model_dir.glob("xgb_fold*.bin")):
                    booster_model = xgb.Booster()
                    booster_model.load_model(str(model_path))

                    mlflow.xgboost.log_model(
                        xgb_model=booster_model,
                        artifact_path=f"model_fold_{i}",
                        registered_model_name=f"XGBoost_Fertilizer_Fold_{i}"
                    )
