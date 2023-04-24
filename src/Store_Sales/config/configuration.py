from Store_Sales.constants import *
from Store_Sales.utils import *
from Store_Sales.entity import *

class ConfigurationManager:
    def __init__(
        self, 
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        #schema_filepath = SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        #self.schema = read_yaml(schema_filepath)
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
    

    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            schema_dir=config.schema_dir,
            report_file_name=config.report_file_name,  
            report_page_file_name=config.report_page_file_name
           
        )
        
        return data_validation_config
    

    
    def get_data_cleaning_config(self) -> DataCleaningConfig:
        config = self.config.data_cleaning
        

        create_directories([config.root_dir,config.clean_csv_dir])

        data_cleaning_config = DataCleaningConfig(
            root_dir=config.root_dir,
            clean_csv_dir = config.clean_csv_dir,
            clean_train_filename = config.clean_train_filename,
            clean_validation_filename = config.clean_validation_filename,
            clean_test_filename = config.clean_test_filename
           
        )

        return data_cleaning_config
    


    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        

        create_directories([config.root_dir,config.transform_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            transform_dir= config.transform_dir,
            transform_train_filename=config.transform_train_filename,
            transform_validation_filename= config.transform_validation_filename,
            column_transformer_object= config.column_transformer_object
           
        )

        return data_transformation_config
    
    def get_model_training_config(self) -> ModelTrainingConfig:
            config = self.config.model_training
            

            create_directories([config.root_dir])

            model_training_config = ModelTrainingConfig(
                root_dir=config.root_dir,
                experiment_runs_filename=config.experiment_runs_filename
            )

            return model_training_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
            config = self.config.model_evaluation
            

            create_directories([config.root_dir])

            model_evaluation_config = ModelEvaluationConfig(
                root_dir=config.root_dir,
                base_accuracy=config.base_accuracy,
                model_filename=config.model_filename
            )

            return model_evaluation_config
    
    def get_model_pusher_config(self) -> ModelPusherConfig:
            config = self.config.model_pusher
            

            create_directories([config.dest_dir])

            model_pusher_config = ModelPusherConfig(
                dest_dir=config.dest_dir
            )

            return model_pusher_config

