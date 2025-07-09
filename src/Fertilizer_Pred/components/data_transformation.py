import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from Fertilizer_Pred.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform(self):
        df_train = pd.read_csv(self.config.train_data_path)
        df_original = pd.read_csv(self.config.original_data_path)

        if 'id' in df_train.columns:
            df_train.drop(columns=['id'], inplace=True)

        # Make categorical features from numericals
        for col in df_train.select_dtypes(include=['number']).columns:
            df_train[f"cat_{col}"] = df_train[col].astype(str)
            df_original[f"cat_{col}"] = df_original[col].astype(str)

        # Add const
        df_train["const"] = 1
        df_original["const"] = 1

        # Convert object to category
        for col in df_train.select_dtypes(include=['object']).columns:
            df_train[col] = df_train[col].astype("category")
            df_original[col] = df_original[col].astype("category")

        # Encode target
        target = df_train.pop("Fertilizer Name")
        target_org = df_original.pop("Fertilizer Name")

        le = LabelEncoder()
        target_encoded = le.fit_transform(target)
        target_org_encoded = le.transform(target_org)

        # Save all outputs
        df_train.to_csv(self.config.transformed_train_path, index=False)
        df_original.to_csv(self.config.transformed_original_path, index=False)
        pd.DataFrame({"target": target_encoded}).to_csv(self.config.train_label_path, index=False)
        pd.DataFrame({"target": target_org_encoded}).to_csv(self.config.original_label_path, index=False)
        joblib.dump(le, self.config.label_encoder_path)
