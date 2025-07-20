import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder

class FertilizerPredictor:
    def __init__(self):
        self.model_dir = Path('artifacts/model_trainer/models')
        self.label_encoder_path = Path('artifacts/model_trainer/models/label_encoder.pkl')
        self.models = self._load_models()
        self.label_encoder = self._load_label_encoder()

    def _load_models(self):
        """Load all XGBoost models from model_dir"""
        try:
            model_paths = list(self.model_dir.glob("xgb_fold*.bin"))
            if not model_paths:
                raise FileNotFoundError(f"No model files found in {self.model_dir}")
            
            models = []
            for path in model_paths:
                model = xgb.Booster()
                model.load_model(str(path))
                models.append(model)
            
            print(f"Loaded {len(models)} models successfully")
            return models
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def _load_label_encoder(self):
        """Load the label encoder"""
        try:
            if not self.label_encoder_path.exists():
                raise FileNotFoundError(f"Label encoder not found at {self.label_encoder_path}")
            
            label_encoder = joblib.load(self.label_encoder_path)
            print(f"Label encoder loaded with classes: {label_encoder.classes_}")
            return label_encoder
        except Exception as e:
            print(f"Error loading label encoder: {e}")
            raise

    def _preprocess_input(self, input_data):
        """Preprocess input data to match training format"""
        try:
            # Create DataFrame with the exact structure used during training
            data = pd.DataFrame({
                'Temparature': [float(input_data['temperature'])],  # Note: keeping original spelling
                'Humidity': [float(input_data['humidity'])],
                'Moisture': [50.0],  # Default value - you might want to add this to your form
                'Soil Type': [input_data['soil_type']],
                'Crop Type': ['Unknown'],  # Default value - you might want to add this to your form
                'Nitrogen': [float(input_data['nitrogen'])],
                'Potassium': [float(input_data['potassium'])],
                'Phosphorous': [float(input_data['phosphorous'])],
            })

            # Add engineered features (matching your training pipeline)
            for col in data.select_dtypes(include=['number']).columns:
                data[f"cat_{col}"] = data[col].astype(str)

            data["const"] = 1

            # Convert categorical columns to category type
            for col in data.select_dtypes(include=['object']).columns:
                data[col] = data[col].astype("category")

            # Convert categorical features to codes (matching training preprocessing)
            cat_cols = data.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                data[col] = data[col].astype('category').cat.codes

            print(f"Preprocessed data shape: {data.shape}")
            print(f"Preprocessed data columns: {data.columns.tolist()}")
            
            return data

        except Exception as e:
            print(f"Error in preprocessing: {e}")
            raise

    def predict(self, input_data: dict) -> list:
        """Predict top 3 fertilizer recommendations"""
        try:
            # Preprocess the input data
            processed_data = self._preprocess_input(input_data)
            
            # Create DMatrix for XGBoost
            dmatrix = xgb.DMatrix(processed_data)
            
            # Get predictions from all models and average them
            all_predictions = []
            for i, model in enumerate(self.models):
                pred = model.predict(dmatrix)
                all_predictions.append(pred)
                print(f"Model {i} prediction shape: {pred.shape}")
            
            # Average predictions across all models
            avg_predictions = np.mean(all_predictions, axis=0)
            
            # Handle both 1D and 2D prediction arrays
            if len(avg_predictions.shape) == 1:
                pred_probs = avg_predictions
            else:
                pred_probs = avg_predictions[0]  # Take first row if 2D
            
            print(f"Average prediction probabilities: {pred_probs}")
            
            # Get top 3 predictions
            top_3_indices = np.argsort(pred_probs)[::-1][:3]
            print(f"Top 3 indices: {top_3_indices}")
            
            # Convert indices back to fertilizer names
            top_3_fertilizers = self.label_encoder.inverse_transform(top_3_indices)
            
            print(f"Top 3 fertilizers: {top_3_fertilizers}")
            
            return top_3_fertilizers.tolist()

        except Exception as e:
            print(f"Error in prediction: {e}")
            # Return default recommendations in case of error
            return ["DAP", "Urea", "NPK"]

    def get_fertilizer_info(self, fertilizer_name):
        """Get additional information about a fertilizer"""
        fertilizer_info = {
            "Urea": {
                "description": "High nitrogen content fertilizer, excellent for leafy growth",
                "composition": "46% Nitrogen",
                "best_for": "Cereals, vegetables, and crops requiring high nitrogen"
            },
            "DAP": {
                "description": "Diammonium Phosphate - provides both nitrogen and phosphorus",
                "composition": "18% Nitrogen, 46% Phosphorus",
                "best_for": "Root development and early plant growth"
            },
            "NPK": {
                "description": "Balanced fertilizer containing nitrogen, phosphorus, and potassium",
                "composition": "Variable N-P-K ratios",
                "best_for": "General purpose fertilization for most crops"
            },
            "MOP": {
                "description": "Muriate of Potash - high potassium content",
                "composition": "60% Potassium",
                "best_for": "Fruit development and plant disease resistance"
            },
            "SSP": {
                "description": "Single Super Phosphate - phosphorus and sulfur",
                "composition": "16% Phosphorus, 12% Sulfur",
                "best_for": "Phosphorus deficient soils"
            },
            "CAN": {
                "description": "Calcium Ammonium Nitrate - nitrogen with calcium",
                "composition": "26% Nitrogen, 10% Calcium",
                "best_for": "Crops requiring both nitrogen and calcium"
            },
            "HYP": {
                "description": "High Yield Phosphate - specialized phosphorus fertilizer",
                "composition": "High Phosphorus content",
                "best_for": "Enhancing root development and flowering"
            }
        }
        
        return fertilizer_info.get(fertilizer_name, {
            "description": "Specialized fertilizer for your soil conditions",
            "composition": "Balanced nutrient composition",
            "best_for": "Optimized for your specific agricultural needs"
        })
