from Store_Sales.constants import *
from Store_Sales.utils import *
from Store_Sales.entity import DataIngestionConfig

class ConfigurationManager:
    def __init__(
        self, 
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            kaggle_file_path=config.kaggle_file_path,
            train_filename=config.train_filename,
            test_filename=config.test_filename
        )

        return data_ingestion_config