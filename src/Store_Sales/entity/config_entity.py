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