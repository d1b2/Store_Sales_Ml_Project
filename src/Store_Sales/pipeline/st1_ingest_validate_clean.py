from Store_Sales.config import ConfigurationManager
from Store_Sales.components import DataIngestion,DataValidation,DataCleaning
from Store_Sales.entity import *
from Store_Sales import logger

STAGE_NAME = "Pipeline"

def main():
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config) 

    data_validation_config = config.get_data_validation_config()
    data_validation = DataValidation(config=data_validation_config)

    data_cleaning_config = config.get_data_cleaning_config()
    data_cleaning = DataCleaning(config=data_cleaning_config) 

    data_ingestion.initiate_data_ingestion()    
   
    data_validation.initiate_data_validation()

    data_cleaning.initiate_data_cleaning()



if __name__ == '__main__':
    try:
        logger.info(f"\n\n{'>'*20}  {STAGE_NAME} started {'<'*20}\n\n")
        main()
        logger.info(f"\n\n{'>'*20}  {STAGE_NAME} completed {'<'*20}\n\n")
    except Exception as e:
        logger.exception(e)
        raise e



