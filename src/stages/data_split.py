
import argparse
import yaml
import pandas as pd
from src.utils.utils import calc_prevalence
from sklearn.preprocessing import StandardScaler
import pickle

from src.utils.logs import get_logger

def data_split(config_path):

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('DATA SPLIT', log_level=config['base']['log_level'])

    df_data = pd.read_csv(config['featurize']['features_path'])


    # Save 30% of the data as validation and test data 
    df_valid_test=df_data.sample(frac=0.30,random_state=42)
    logger.info('Split size: %.3f'%(len(df_valid_test)/len(df_data)))

    df_test = df_valid_test.sample(frac = 0.5, random_state = 42)
    df_valid = df_valid_test.drop(df_test.index)

    # use the rest of the data as training data
    df_train_all=df_data.drop(df_valid_test.index)


    # Fit the scaler using all training data
    scaler  = StandardScaler()
    #X_train_all = df_train_all[].values.astype('float32')
    target_column=config['featurize']['target_column']
    
    X_train_all = df_train_all.drop(target_column, axis=1).values.astype('float32')
    scaler.fit(X_train_all)
    scalerfile = config['data_split']['scaler_path']
    pickle.dump(scaler, open(scalerfile, 'wb'))

    logger.info('Test prevalence(n = %d):%.3f'%(len(df_test),calc_prevalence(df_test.OUTPUT_LABEL.values)))
    logger.info('Valid prevalence(n = %d):%.3f'%(len(df_valid),calc_prevalence(df_valid.OUTPUT_LABEL.values)))
    logger.info('Train all prevalence(n = %d):%.3f'%(len(df_train_all), calc_prevalence(df_train_all.OUTPUT_LABEL.values)))
    
    # Handling imbalance
    # use the sub-sample approach. Here, we will create a balanced training data set that has 50% positive and 50% negative.
    # You can also play with this ratio to see if you can get an improvement.
    
    # split the training data into positive and negative
    rows_pos = df_train_all.OUTPUT_LABEL == 1
    df_train_pos = df_train_all.loc[rows_pos]
    df_train_neg = df_train_all.loc[~rows_pos]

    # merge the balanced data
    df_train = pd.concat([df_train_pos, df_train_neg.sample(n = len(df_train_pos), random_state = 42)],axis = 0)

    # shuffle the order of training samples 
    df_train = df_train.sample(n = len(df_train), random_state = 42).reset_index(drop = True)

    logger.info('Train balanced prevalence(n = %d):%.3f'%(len(df_train), calc_prevalence(df_train.OUTPUT_LABEL.values)))


    
    train_csv_path = config['data_split']['trainset_path']
    validset_path = config['data_split']['validset_path']
    testset_path = config['data_split']['testset_path']
    train_unbalanced_path = config['data_split']['train_unbalanced_path']
    
    df_train_all.to_csv(train_unbalanced_path, index=False)
    df_train.to_csv(train_csv_path, index=False)
    df_valid.to_csv(validset_path, index=False)
    df_test.to_csv(testset_path, index=False)

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_split(config_path=args.config)
