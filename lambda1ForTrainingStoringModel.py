import json
import pandas as pd
import boto3
from DataFactoryForLambda1 import *
import tempfile
import joblib
from sklearn.utils import resample
#from io import StringIO

#s3_resource = boto3.resource('s3')
key = "model.pkl"
s3_client = boto3.client("s3")
unlabeled_bucket = "hw23-unlab"
#labeled_bucket = "labeled-data"

def lambda_handler(event, context):
    
    # part 1 - get the filename of new unlabeled csv file
    s3_file_name = event["Records"][0]["s3"]["object"]["key"]
    resp = s3_client.get_object(Bucket=unlabeled_bucket, Key=s3_file_name)
    df = pd.read_csv(resp['Body'])
    
    loan_0 = df[df.not_fully_paid==0]
    loan_1 = df[df.not_fully_paid==1]
    n_majority_class = loan_0.shape[0]
    n_minority_class = loan_1.shape[0]
    loan_0_undersampled = resample(loan_0, replace=False, n_samples=n_minority_class, random_state=123)
    df_balanced = pd.concat([loan_0_undersampled, loan_1])
    
    dataFactory=DataFactory()
    model=dataFactory.TrainModel(df_balanced)
    with tempfile.TemporaryFile() as fp:
        joblib.dump(model, fp)
        fp.seek(0)
        s3_client.put_object(Body=fp.read(), Bucket='loanmodel', Key=key)
    
    #s3_resource.delete_object(Bucket='loanmodel', Key=key)
    s3_client.delete_object(Bucket=unlabeled_bucket, Key=s3_file_name)
