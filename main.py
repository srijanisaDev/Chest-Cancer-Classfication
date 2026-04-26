from Chest_Cancer_classifier import logger
from Chest_Cancer_classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Chest_Cancer_classifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
# from Chest_Cancer_classifier.pipeline.st import ModelTrainingPipeline
# from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline



STAGE_NAME = "Data Ingestion stage"


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e