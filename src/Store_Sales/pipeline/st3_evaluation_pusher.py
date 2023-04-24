from Store_Sales.config import ConfigurationManager
from Store_Sales.components import ModelEvaluation,ModelPusher
from Store_Sales.entity import *
from Store_Sales import logger

STAGE_NAME = "Pipeline"

def main():
    config = ConfigurationManager()
    model_evaluation_config = config.get_model_evaluation_config()
    model_evaluation = ModelEvaluation(config=model_evaluation_config) 

    model_pusher_config = config.get_model_pusher_config()
    model_pusher = ModelPusher(config= model_pusher_config) 

   
 
    model_evaluation.initiate_model_evaluation()

    model_pusher.initiate_model_pusher()       


if __name__ == '__main__':
    try:
        logger.info(f"\n\n{'>'*20}  {STAGE_NAME} started {'<'*20}\n\n")
        main()
        logger.info(f"\n\n{'>'*20}  {STAGE_NAME} completed {'<'*20}\n\n")
    except Exception as e:
        logger.exception(e)
        raise e