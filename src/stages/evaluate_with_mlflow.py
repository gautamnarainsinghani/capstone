
import argparse
import joblib
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score
from typing import Text, Dict
import yaml
import os
from urllib.parse import urlparse
import mlflow
from mlflow.tracking import MlflowClient

from src.report.visualize import plot_confusion_matrix
from src.utils.logs import get_logger

def convert_to_labels(indexes, labels):
    result = []
    for i in indexes:
        result.append(labels[i])
    return result

def write_confusion_matrix_data(y_true, predicted, labels, filename):
    assert len(predicted) == len(y_true)
    predicted_labels = convert_to_labels(predicted, labels)
    true_labels = convert_to_labels(y_true, labels)
    cf = pd.DataFrame(list(zip(true_labels, predicted_labels)), columns=["y_true", "predicted"])
    cf.to_csv(filename, index=False)

def evaluate_model(config_path: Text) -> None:
    """Evaluate model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('EVALUATE', log_level=config['base']['log_level'])

    logger.info('Load model')
    model_path = config['train']['model_path']
    model = joblib.load(model_path)

    logger.info('Load test dataset')
    print('Load test dataseet')
    test_df = pd.read_csv(config['data_split']['testset_path'])

    logger.info('Evaluate (build report)')
    target_column=config['featurize']['target_column']
    y_test = test_df.loc[:, target_column].values
    X_test = test_df.drop(target_column, axis=1).values

    prediction = model.predict(X_test)
    f1 = f1_score(y_true=y_test, y_pred=prediction, average='macro')

    labels = ['Yes', 'No']
    cm = confusion_matrix(prediction, y_test)
    report = {
        'f1': f1,
        'cm': cm,
        'actual': y_test,
        'predicted': prediction
    }

    logger.info('Save metrics and evaluate')
    mlflow.set_experiment("gautam")  # Set your experiment name
    client = MlflowClient()
    best_model_runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name("gautam").experiment_id],
        filter_string="",
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )

    if best_model_runs:
        best_f1_score = best_model_runs[0].data.metrics['f1_score']
        logger.info(f'Best F1 Score: {best_f1_score}')
        print(f'Best F1 Score: {best_f1_score}')
    else:
        best_f1_score = 0
    remote_server_uri = "http://ec2-54-196-182-184.compute-1.amazonaws.com:5000"
    mlflow.set_tracking_uri(remote_server_uri) 

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    with mlflow.start_run():
        mlflow.log_metric('f1_score', f1)
        print(f'f1_score : {f1}')
        if f1 > best_f1_score:
            logger.info('New model performs better, updating MLflow model')
            mlflow.sklearn.log_model(model, "model")
            # Optionally, log other artifacts, such as confusion matrix
            reports_folder = Path(config['evaluate']['reports_dir'])
            os.makedirs(reports_folder, exist_ok=True)
            confusion_matrix_png_path = reports_folder / config['evaluate']['confusion_matrix_image']
            plt = plot_confusion_matrix(cm=report['cm'], target_names=['Yes', 'No'], normalize=False)
            plt.savefig(confusion_matrix_png_path)
            mlflow.log_artifact(confusion_matrix_png_path)
        else:
            logger.info('New model does not perform better. Keeping the existing model.')

    logger.info('Evaluation complete')


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)
