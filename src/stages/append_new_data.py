
import wget
import pandas as pd
import argparse
import yaml
import shutil

from src.utils.logs import get_logger

def append_new_data(config_path):

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('APPEND NEW DATA', log_level=config['base']['log_level'])

    if not config['base']['continuous_learning']:
        logger.info("Continuous learning NOT enabled")
        # just rename the output
        
        shutil.copy(config['data_load']['dataset_csv'], config['data_load']['appended_dataset_csv'])
        return

    logger.info("continuous learning")
    new_data_url = config['data_load']['source_new']
    local_filename = config['data_load']['new_data']
    
    
    # Download the file
    logger.info(f'Downloading from {new_data_url}')
    wget.download(new_data_url, local_filename)
    logger.info(f"Downloaded {local_filename}")

    ## Append to the current data
    current_csv_name = config['data_load']['dataset_csv']
    df = pd.read_csv(current_csv_name)
    logger.info(f"Shape before adding new data {df.shape}")
    
    new_df = pd.read_csv(config['data_load']['new_data'])
    logger.info(f"Shape of new data {new_df.shape}")

    ## combine the two dataframe and save it

    result_df = pd.concat([df, new_df], ignore_index=True)
    
    logger.info(f"Shape after adding new data {result_df.shape}")

    current_csv_name = config['data_load']['appended_dataset_csv']
    result_df.to_csv(current_csv_name, index=False)
    
    
if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    append_new_data(config_path=args.config)


    
