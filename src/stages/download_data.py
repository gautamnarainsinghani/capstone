
import os
import wget
import yaml
import argparse

from src.utils.logs import get_logger

def download_data(config_path):
    
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('DOWNLOAD DATA', log_level=config['base']['log_level'])

    url = config['data_load']['source']
        
    logger.info(f'Downloading from {url}')
    local_folder = config['data_load']['local_folder']
    local_filename = os.path.join(local_folder, config['data_load']['local_name'])


    # Create the folder if it does not exist
    os.makedirs(local_folder, exist_ok=True)
    
    
    # Download the file
    wget.download(url, local_filename)
    logger.info(f"Downloaded {local_filename}")


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    download_data(config_path=args.config)
