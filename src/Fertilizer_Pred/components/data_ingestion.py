import os
import urllib.request as request
import zipfile
from Fertilizer_Pred import logger
from Fertilizer_Pred.utils.common import get_size
from pathlib import Path
from src.Fertilizer_Pred.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """Downloads the file from source_URL to local_data_file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
        
        if not os.path.exists(self.config.local_data_file):
            logger.info(f"Downloading data from {self.config.source_URL}...")
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,  # Use source_URL here
                filename=self.config.local_data_file  # Save to local_data_file
            )
            logger.info(f"Download completed to: {filename}")
            logger.debug(f"Download headers: {headers}")
        else:
            logger.info(f"File already exists at {self.config.local_data_file}, size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """Extracts the downloaded zip file to unzip_dir"""
        logger.info(f"Extracting zip file from {self.config.local_data_file} to {self.config.unzip_dir}")
        
        # Create extraction directory if it doesn't exist
        os.makedirs(self.config.unzip_dir, exist_ok=True)
        
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
        
        logger.info(f"Successfully extracted to {self.config.unzip_dir}")