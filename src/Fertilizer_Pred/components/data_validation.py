import pandas as pd
from Fertilizer_Pred import logger
from Fertilizer_Pred.entity.config_entity import DataValidationConfig
class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = True
            expected_columns = set(self.config.all_schema.keys())

            # Validate both train and original datasets
            for file_path in [self.config.train_data_path, self.config.original_data_path]:
                df = pd.read_csv(file_path)

                # Drop 'id' column if present
                if 'id' in df.columns:
                    df.drop(columns=['id'], inplace=True)

                actual_columns = set(df.columns)

                if not expected_columns.issubset(actual_columns):
                    validation_status = False
                    break

            # Write result to status file
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e
