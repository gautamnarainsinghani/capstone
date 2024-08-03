
import argparse
import pandas as pd
from typing import Text
import yaml
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

from src.utils.logs import get_logger

def featurize(config_path, train=False):

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('FEATURE ENG', log_level=config['base']['log_level'])

    # Create the folder if it does not exist
    os.makedirs(config['featurize']['folder_processed'], exist_ok=True)
    

    #df = pd.read_csv(path)
    if train:
        logger.info('Using training data, fittin encoder')
        df = pd.read_csv(config['data_load']['dataset_csv'])
    else:
        logger.info('new data, use previously fitted encoder')
        df = pd.read_csv(config['data_load']['inference_dataset_csv'])
    

    # replace ? with nan
    df = df.replace('?',np.nan)
    
    cols_num = ['time_in_hospital','num_lab_procedures', 'num_procedures', 'num_medications',
           'number_outpatient', 'number_emergency', 'number_inpatient','number_diagnoses']

    cols_cat = ['race', 'gender', 
           'max_glu_serum', 'A1Cresult',
           'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
           'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
           'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
           'tolazamide', 'insulin',
           'glyburide-metformin', 'glipizide-metformin',
           'glimepiride-pioglitazone', 'metformin-rosiglitazone',
           'metformin-pioglitazone', 'change', 'diabetesMed','payer_code']
    
    

    # handle missing values
    logger.info('Handling missing values')
    df['race'] = df['race'].fillna('UNK')
    df['payer_code'] = df['payer_code'].fillna('UNK')
    df['medical_specialty'] = df['medical_specialty'].fillna('UNK')

    df['max_glu_serum'] = df['max_glu_serum'].fillna('UNK')
    df['A1Cresult'] = df['A1Cresult'].fillna('UNK')

    # Bucket Medical Speciality
    top_10 = ['UNK','InternalMedicine','Emergency/Trauma',\
              'Family/GeneralPractice', 'Cardiology','Surgery-General' ,\
              'Nephrology','Orthopedics',\
              'Orthopedics-Reconstructive','Radiologist']

    # make a new column with duplicated data
    df['med_spec'] = df['medical_specialty'].copy()

    # replace all specialties not in top 10 with 'Other' category
    df.loc[~df.med_spec.isin(top_10),'med_spec'] = 'Other'


    # categorical numeric data into string type, to use with get_dummies
    cols_cat_num = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    df[cols_cat_num] = df[cols_cat_num].astype('str')


    cols_to_encode = cols_cat + cols_cat_num + ['med_spec']
    encoder_path = config['featurize']['encoder_path']

    if train:
        # Initialize OneHotEncoder with handle_unknown set to 'ignore'
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

        # Fit the encoder on your data
        logger.info('fitting encoder')
        encoder.fit(df[cols_to_encode])

        # Serialize the encoder to a file
        joblib.dump(encoder, encoder_path)
        logger.info('Encoder object serialized')
    
    else:
        # Deserialize the encoder from the file
        logger.info('Loadign encoder object')
        encoder = joblib.load(encoder_path)

    # Use the encoder to transform new data
    new_encoded_array = encoder.transform(df[cols_to_encode])
    new_encoded_df = pd.DataFrame(new_encoded_array, columns=encoder.get_feature_names_out(cols_to_encode))

    df = pd.concat([df,new_encoded_df], axis = 1)

    cols_all_cat = list(new_encoded_df.columns)

    age_id = {'[0-10)':0, 
              '[10-20)':10, 
              '[20-30)':20, 
              '[30-40)':30, 
              '[40-50)':40, 
              '[50-60)':50,
              '[60-70)':60, 
              '[70-80)':70, 
              '[80-90)':80, 
              '[90-100)':90}
    df['age_group'] = df.age.replace(age_id)

    df['has_weight'] = df.weight.notnull().astype('int')
    cols_extra = ['age_group','has_weight']


    logger.info(f'Total number of features: {len(cols_num + cols_all_cat + cols_extra)}')
    logger.info(f'Numerical Features:  {len(cols_num)}')
    logger.info(f'Categorical Features: {len(cols_all_cat)}')
    logger.info(f'Extra features: {len(cols_extra)}')
    logger.info(f'Data shape:{df.shape}')
    
    col2use = cols_num + cols_all_cat + cols_extra
    featured_dataset = df[col2use + ['OUTPUT_LABEL']]
    features_path = config['featurize']['features_path']
    #df_data.to_csv("../data/processed/featured.csv")
    featured_dataset.to_csv(features_path, index=False)

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args_parser.add_argument('--train', action='store_true', help='Fit and transform encoder')
    args = args_parser.parse_args()

    featurize(config_path=args.config, train=args.train)
