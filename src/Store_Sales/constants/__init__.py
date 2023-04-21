from pathlib import Path

CONFIG_FILE_PATH = Path("configs/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
SCHEMA_FILE_PATH= Path("configs/schema.yaml")
SECRETS_FILE_PATH=Path('configs/secrets.yaml')
TRAIN_FILE_PATH=Path("artifacts/data_ingestion/Train.csv")
TEST_FILE_PATH=Path("artifacts/data_ingestion//Test.csv")
CLEAN_TRAIN_FILE_PATH=Path('artifacts/data_cleaning/clean_data/train.csv')
CLEAN_VALID_FILE_PATH=Path('artifacts/data_cleaning/clean_data/validation.csv')
CLEAN_TEST_FILE_PATH=Path('artifacts/data_cleaning/clean_data/test.csv')
CLEAN_TRAIN_ARRAY_PATH=Path('artifacts/data_transformation/transformed_data/train.npy')
CLEAN_VALID_ARRAY_PATH=Path('artifacts/data_transformation/transformed_data/validation.npy')