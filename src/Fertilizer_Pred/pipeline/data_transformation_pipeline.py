from src.Fertilizer_Pred.config.configuration import ConfigurationManager
from src.Fertilizer_Pred.components.data_transformation import DataTransformation
from src.Fertilizer_Pred import logger
from pathlib import Path

STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Read validation status
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                # Proceed with transformation if schema is valid
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                transformer = DataTransformation(config=data_transformation_config)
                transformer.transform()
            else:
                raise Exception("❌ Data schema is not valid. Transformation aborted.")

        except Exception as e:
            print(f"[DataTransformation Error] {e}")
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
