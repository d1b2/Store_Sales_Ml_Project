from Store_Sales.config import ConfigurationManager
from Store_Sales.components import DataIngestion
from Store_Sales.entity import *
from Store_Sales import logger

STAGE_NAME = "Data Ingestion stage"

def main():
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)      
    data_ingestion.initiate_data_ingestion()
    


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>  {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e



