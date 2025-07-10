# prediction_pipeline.py

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

class FertilizerPredictor:
    def __init__(self):
        self.model_dir = Path('artifacts/model_trainer/models')
        self.models = self._load_models()
        self.class_names = ["Urea", "DAP", "MOP", "NPK", "CAN", "SSP", "HYP"]  # Adjust if needed

    def _load_models(self):
        """Load all XGBoost models from model_dir"""
        model_paths = list(self.model_dir.glob("xgb_fold*.bin"))
        return [xgb.Booster(model_file=str(path)) for path in model_paths]

    def predict(self, input_data: dict) -> list:
        """Predict top 3 fertilizer recommendations"""
        data = pd.DataFrame({
            'Temparature': [float(input_data['temperature'])],
            'Humidity': [float(input_data['humidity'])],
            'Soil Type': [input_data['soil_type']],
            'Nitrogen': [float(input_data['nitrogen'])],
            'Potassium': [float(input_data['potassium'])],
            'Phosphorous': [float(input_data['phosphorous'])],
            'pH': [float(input_data['ph'])],
            'Rainfall': [float(input_data['rainfall'])],
            'const': [1]
        })

        # Encode categorical features
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].astype('category').cat.codes

        dmatrix = xgb.DMatrix(data)
        pred_probs = np.mean([model.predict(dmatrix) for model in self.models], axis=0)
        pred_probs = pred_probs.flatten()
        top_3_indices = np.argsort(pred_probs)[::-1][:3]
        top_3_fertilizers = [self.class_names[i] for i in top_3_indices]

        return top_3_fertilizers
