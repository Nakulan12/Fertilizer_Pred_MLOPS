from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL:str
    local_data_file:Path
    unzip_dir:Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    train_data_path: Path
    original_data_path: Path
    all_schema: dict



@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    original_data_path: Path
    transformed_train_path: Path
    transformed_original_path: Path
    train_label_path: Path
    original_label_path: Path
    label_encoder_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    original_data_path: Path
    train_label_path: Path
    original_label_path: Path
    model_dir: Path
    model_name: str
    target_column: str
    max_depth: int
    learning_rate: float
    reg_alpha: float
    reg_lambda: float
    gamma: float
    subsample: float
    colsample_bytree: float
    min_child_weight: int
    num_boost_round: int
    early_stopping_rounds: int
    n_folds: int
    random_seed: int
    n_jobs: int
