import pandas as pd
import joblib
from Fertilizer_Pred.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform(self):
        # Load data
        df_train = pd.read_csv(self.config.train_data_path)
        df_original = pd.read_csv(self.config.original_data_path)

        if 'id' in df_train.columns:
            df_train.drop(columns=['id'], inplace=True)

        # Feature engineering
        for col in df_train.select_dtypes(include=['number']).columns:
            df_train[f"cat_{col}"] = df_train[col].astype(str)
            df_original[f"cat_{col}"] = df_original[col].astype(str)

        df_train["const"] = 1
        df_original["const"] = 1

        # Convert features to category (skip target column)
        for col in df_train.select_dtypes(include=['object']).columns:
            if col != "Fertilizer Name":  # Skip target column
                df_train[col] = df_train[col].astype("category")
                df_original[col] = df_original[col].astype("category")

        # Extract and save raw targets (NO ENCODING)
        target = df_train.pop("Fertilizer Name")
        target_org = df_original.pop("Fertilizer Name")

        # Save processed features and raw labels
        df_train.to_csv(self.config.transformed_train_path, index=False)
        df_original.to_csv(self.config.transformed_original_path, index=False)
        
        # Save raw labels (not encoded)
        target.to_csv(self.config.train_label_path, index=False)
        target_org.to_csv(self.config.original_label_path, index=False)