
import argparse
import pandas as pd
import yaml
import os

from src.utils.utils import calc_prevalence
from src.utils.logs import get_logger

def load_data(config_path):

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('LOAD DATA', log_level=config['base']['log_level'])

    local_folder = config['data_load']['local_folder']
    local_filename = os.path.join(local_folder, config['data_load']['local_name'])

    # load the csv file
    df = pd.read_csv(local_filename)
    #Here we will label if a patient is likely to be re-admitted within 30 days of discharge.
    df['OUTPUT_LABEL'] = (df.readmitted == '<30').astype('int')
    logger.info('Prevalence:%.3f'%calc_prevalence(df['OUTPUT_LABEL'].values))

    # shuffle the samples
    df = df.sample(n = len(df), random_state = 42)
    df = df.reset_index(drop = True)
    
    # Save 30% of the data as future data 
    df_future=df.sample(frac=0.30,random_state=42)
    df_current = df.drop(df_future.index)
    
    logger.info('Split size current: %.3f'%(len(df_current)/len(df)))
    logger.info('Split size future: %.3f'%(len(df_future)/len(df)))

    logger.info('Prevalence current:%.3f'%calc_prevalence(df_current['OUTPUT_LABEL'].values))
    logger.info('Prevalence future:%.3f'%calc_prevalence(df_future['OUTPUT_LABEL'].values))
    
    # save raw data
    
    df_current.to_csv(config['data_load']['dataset_csv'], index=False)
    df_future.to_csv(config['data_load']['future_csv'], index=False)
    


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    load_data(config_path=args.config)
