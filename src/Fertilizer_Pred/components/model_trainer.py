import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')
from src.Fertilizer_Pred.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def _load_data(self) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Load and validate training data"""
        try:
            # Verify files exist
            for path in [
                self.config.train_data_path,
                self.config.original_data_path,
                self.config.train_label_path,
                self.config.original_label_path
            ]:
                if not path.exists():
                    raise FileNotFoundError(f"Data file not found: {path}")

            # Load data
            df_train = pd.read_csv(self.config.train_data_path)
            df_original = pd.read_csv(self.config.original_data_path)
            
            train_labels = pd.read_csv(self.config.train_label_path)[self.config.target_column]
            original_labels = pd.read_csv(self.config.original_label_path)[self.config.target_column]
            
            # Encode labels
            le = LabelEncoder()
            y_train = le.fit_transform(train_labels)
            y_original = le.transform(original_labels)
            joblib.dump(le, os.path.join(self.config.model_dir, "label_encoder.pkl"))

            # Combine data
            X = pd.concat([df_train, df_original], axis=0)
            y = np.concatenate([y_train, y_original])
            
            return X, y, le.classes_
            
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}") from e

    def _train_model(self, X: pd.DataFrame, y: np.ndarray, classes: List[str]):
        """Train XGBoost model with configured parameters"""
        try:
            # Convert categorical columns to codes
            cat_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                X[col] = X[col].astype('category').cat.codes

            params = {
                'objective': 'multi:softprob',
                'num_class': len(classes),
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate,
                'reg_alpha': self.config.reg_alpha,
                'reg_lambda': self.config.reg_lambda,
                'gamma': self.config.gamma,
                'subsample': self.config.subsample,
                'colsample_bytree': self.config.colsample_bytree,
                'min_child_weight': self.config.min_child_weight,
                'random_state': self.config.random_seed,
                'n_jobs': self.config.n_jobs,
                'tree_method': 'hist',
                'eval_metric': 'mlogloss',
                'enable_categorical': False  # We've already converted categories
            }

            skf = StratifiedKFold(n_splits=self.config.n_folds,
                                shuffle=True,
                                random_state=self.config.random_seed)

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                print(f"\nTraining Fold {fold+1}")
                
                X_train, y_train = X.iloc[train_idx], y[train_idx]
                X_val, y_val = X.iloc[val_idx], y[val_idx]

                # Create DMatrix (no need for enable_categorical=True since we converted)
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)

                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.config.num_boost_round,
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    evals=[(dtrain, "train"), (dval, "val")],
                    verbose_eval=200
                )
                
                model_path = os.path.join(self.config.model_dir, f"xgb_fold{fold}.bin")
                model.save_model(model_path)
                print(f"Saved model for fold {fold+1} to {model_path}")

        except Exception as e:
            raise RuntimeError(f"Error during model training: {str(e)}")

    def train(self):
        """Execute full training pipeline"""
        try:
            os.makedirs(self.config.model_dir, exist_ok=True)
            X, y, classes = self._load_data()
            print(f"Starting training with {len(classes)} fertilizer classes")
            self._train_model(X, y, classes)
            print("Training completed successfully!")
        except Exception as e:
            raise RuntimeError(f"Training pipeline failed: {str(e)}")