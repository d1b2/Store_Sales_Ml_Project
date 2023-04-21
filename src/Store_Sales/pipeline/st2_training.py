from Store_Sales.config import ConfigurationManager
from Store_Sales.components import ModelTraining
from Store_Sales.entity import *
from Store_Sales import logger

STAGE_NAME = "Pipeline"

def main():
    config = ConfigurationManager()
    model_training_config = config.get_model_training_config()
    model_training = ModelTraining(config=model_training_config) 

    model_training.initiate_mlflow_experimentations()
    #model_training.del_model()
    #model_training.initiate_model_staging()
    model_training.save_mlflow_experiments()
    

    


if __name__ == '__main__':
    try:
        logger.info(f"\n\n{'>'*20}  {STAGE_NAME} started {'<'*20}\n\n")
        main()
        logger.info(f"\n\n{'>'*20}  {STAGE_NAME} completed {'<'*20}\n\n")
    except Exception as e:
        logger.exception(e)
        raise e



