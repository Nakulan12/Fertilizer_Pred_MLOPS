from src.Fertilizer_Pred.constant import *
from src.Fertilizer_Pred.utils.common import read_yaml,create_directories 
from src.Fertilizer_Pred.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            train_data_path=config.train_data_path,
            original_data_path=config.original_data_path,
            all_schema=schema,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            original_data_path=config.original_data_path,
            transformed_train_path=config.transformed_train_path,
            transformed_original_path=config.transformed_original_path,
            train_label_path=config.train_label_path,
            original_label_path=config.original_label_path,
            label_encoder_path=config.label_encoder_path
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            config = self.config.model_trainer
            params = self.params.XGBoost
            training = self.params.training
            schema = self.schema.TARGET_COLUMN

            # Create required directories
            create_directories([
                Path(config.root_dir),
                Path(config.model_dir)
            ])

             # Verify files exist before proceeding
            required_files = [
                config.train_data_path,
                config.original_data_path,
                config.train_label_path,
                config.original_label_path
            ]

            for file_path in required_files:
                if not Path(file_path).exists():
                    raise FileNotFoundError(f"Required file not found: {file_path}")

            return ModelTrainerConfig(
                root_dir=Path(config.root_dir),
                train_data_path=Path(config.train_data_path),
                original_data_path=Path(config.original_data_path),
                train_label_path=Path(config.train_label_path),
                original_label_path=Path(config.original_label_path),
                model_dir=Path(config.model_dir),
                model_name=config.model_name,
                target_column=schema.name,
                # XGBoost parameters
                max_depth=params.max_depth,
                learning_rate=params.learning_rate,
                reg_alpha=params.reg_alpha,
                reg_lambda=params.reg_lambda,
                gamma=params.gamma,
                subsample=params.subsample,
                colsample_bytree=params.colsample_bytree,
                min_child_weight=params.min_child_weight,
                #   Training parameters
                num_boost_round=training.num_boost_round,
                early_stopping_rounds=training.early_stopping_rounds,
                n_folds=training.n_folds,
                random_seed=training.random_seed,
                n_jobs=training.n_jobs
             )
        except Exception as e:
            raise ValueError(f"Configuration error: {str(e)}") from e
        

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.XGBoost  
        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_dir=Path(config.model_dir),
            metric_file_name=Path(config.metric_file_name),
            target_column=self.schema.TARGET_COLUMN.name,
            mlflow_uri=config.mlflow_uri,
            label_encoder_path=Path(config.label_encoder_path),
            all_params=params
        )

    
