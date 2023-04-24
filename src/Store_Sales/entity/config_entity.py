from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    kaggle_file_path: Path
    test_filename: str
    train_filename: str

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    schema_dir: Path
    report_file_name: str
    report_page_file_name: Path

@dataclass(frozen=True)
class DataCleaningConfig:
    root_dir: Path
    clean_csv_dir: Path
    clean_train_filename: str
    clean_validation_filename: str
    clean_test_filename: str

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path 
    transform_dir: Path 
    transform_train_filename: str
    transform_validation_filename: str
    column_transformer_object: str
    

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    experiment_runs_filename: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    base_accuracy: float
    model_filename: str


    