from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
import sqlite3
#import evidently_config as evcfg
from datetime import datetime
from typing import Optional
import tqdm
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
import io
import pandas as pd
import os


app = FastAPI()

from os.path import dirname, join

current_dir = dirname(__file__)  # this will be the location of the current .py file
parent_directory = os.path.dirname(current_dir)
template_dir = join(parent_directory, 'templates')
file_to_upload = join(parent_directory, "file1.txt")


templates = Jinja2Templates(directory=template_dir)

@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# # Load the pre-trained model and tokenizer
# model_name = "RinInori/bert-base-uncased_finetuned_sentiments"
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)


# def write_prediction_to_db(text, prediction, ground_truth=None):

#     conn = sqlite3.connect(evcfg.SQLITE_DB)

#     cursor = conn.cursor()

#     data_to_insert = (text, prediction, ground_truth, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0)

#     insert_query = "INSERT INTO predictions (input_text, predicted_sentiment, ground_truth, timestamp, reported) VALUES (?, ?, ?, ?, ?)"
#     cursor.execute(insert_query, data_to_insert)

#     conn.commit()
#     conn.close()


@app.post("/predicts/")
async def predict_sentiment(file: UploadFile = File(...)):
    # Extract the text from the JSON payload
    contents = await file.read()
    # Convert the uploaded CSV file content to a DataFrame
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    # Now, save this DataFrame to another CSV file on the server
    output_file_path = "saved_file.csv"
    df.to_csv(output_file_path, index=False)
    
    # # Process the lines here
    # for line in lines:
    #     # Do something with each line
    #     print(line)

    return {"Status": "file read successfully" , "filename": file.filename}

def write_lines_to_file(file_path, lines):
    with open(file_path, 'w') as file:
        for line in lines:
            file.write(line + '\n')

        file.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
