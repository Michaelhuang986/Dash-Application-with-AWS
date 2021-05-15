import json
import boto3
import pickle
from io import StringIO
from DataFactoryForLambda2 import *
import sklearn
import pandas as pd
import joblib
import tempfile

s3_client = boto3.client("s3")

labeled_bucket = 'hw23-lab'
unlabeled_bucket = 'hw23-unlab-trigger'
MODEL_BUCKET = "loanmodel"

def lambda_handler(event, context):
    s3_file_name = event["Records"][0]["s3"]["object"]["key"]
    resp = s3_client.get_object(Bucket=unlabeled_bucket, Key=s3_file_name)
    df = pd.read_csv(resp['Body'])
    
    with tempfile.TemporaryFile() as fp:
        s3_client.download_fileobj(Fileobj=fp, Bucket=MODEL_BUCKET, Key='model.pkl')
        fp.seek(0)
        model = joblib.load(fp)
    
    dataFactory=DataFactory()
    df=dataFactory.Normalization(df)
    df['not_fully_paid'] = model.predict(df)
    
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3_client.put_object(Bucket=labeled_bucket, Key=s3_file_name, Body=csv_buffer.getvalue())
    
    s3_client.delete_object(Bucket=unlabeled_bucket, Key=s3_file_name) 