import dash_html_components as html
import dash_bootstrap_components as dbc
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

import boto3
import io
from DataFactory import *


s3client = boto3.client("s3")
response = s3client.get_object(Bucket = "loanmodel", Key = "loan_data_labeled.csv")
df = pd.read_csv(io.BytesIO(response['Body'].read()))

df_purpose = df.groupby(['purpose']).count().reset_index()

df_before_normalize = df.agg(['mean','std','skew'])
df_1 = df.groupby(['not_fully_paid']).count().reset_index()

loan_0 = df[df.not_fully_paid==0]
loan_1 = df[df.not_fully_paid==1]
n_majority_class = loan_0.shape[0]
n_minority_class = loan_1.shape[0]
loan_0_undersampled = resample(loan_0, replace=False, n_samples=n_minority_class, random_state=123)
#loan_0_undersampled.shape
df_balanced = pd.concat([loan_0_undersampled, loan_1])
df_2 = df_balanced.groupby(['not_fully_paid']).count().reset_index()
print(df_balanced.not_fully_paid.value_counts())

dataFactory=DataFactory()
X_train, X_test, y_train, y_test=dataFactory.DataSplit(df_balanced, 0.2, 1234)

#after normalizing the data,
df_after_normalize = df.agg(['mean','std','skew'])

imLog=dataFactory.TrainModelLogistic(X_train, X_test, y_train, y_test)
accuracyLogstic = dataFactory.Evaluate(X_train, X_test, y_train, y_test)
imLog=imLog.sort_values(ascending=False)
print(imLog)


imRF=dataFactory.TrainModelRF(X_train, X_test, y_train, y_test)
accuracyRF = dataFactory.Evaluate(X_train, X_test, y_train, y_test)
imRF=imRF.sort_values(ascending=False)
print(imRF)


print(accuracyLogstic)
print(accuracyRF)
# Paid vs Not_fully_paid before resample
fig_1 = px.bar(df_1, x="not_fully_paid", y= "credit_policy", title="Paid vs Not fully paid before resample", labels={"credit_policy": "total number"})

# Paid vs Not_fully_paid after resample
fig_2 = px.bar(df_2, x="not_fully_paid", y= "credit_policy", title="Paid vs Not fully paid after resample", labels={"credit_policy": "total number"})
fig_3 = px.pie(df_purpose, values='not_fully_paid', names='purpose', template = 'seaborn')#px.bar(df_purpose, x="purpose", y= "not_fully_paid")


layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Training Dataset")#,
            #className="mr-1"
            )
        ]),
        dbc.Row([
            dbc.Col(html.H5("Describe the training dataset."),
            className="mr-1")
        ]),
        dbc.Row([
            dbc.Col(html.P("LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California. \
                It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform."),
            #className="mx-1 my-1"
            )
        ]),
        dbc.Row([
            dbc.Col(html.P("Solving this case study will give us an idea about how real business problems are solved using EDA and Machine Learning. In this case study, we will also develop a basic understanding of risk analytics in banking and financial services and understand how data is used to minimise the risk of losing money while lending to customers."),
            #className="mx-1 my-1"
            )
        ]),        
        dbc.Row([
            dbc.Col(html.H2("Model Training"),
            className="mr-1")
        ]),
        dbc.Row([
            dbc.Col(html.P("Describe steps you have taken to train your final model. \
                Describe cross validation accuracy, model evaluation process \
                and draw plots to show confusion matrix."),
            className="mr-1")
        ]),
        dbc.Row([
            dbc.Col(html.H5("1. Data gathering"),
            className="mr-1")
        ]),
        dbc.Row([
            dbc.Col(html.H5("2. Exploratory data analysis (EDA)"),
            className="mr-1")
        ]),
        dcc.Graph(
        id='graph3',
        figure=fig_3
        ),
        dbc.Row([
            dbc.Col(html.H5("Before resample the date to make it balanced."),
            className="mr-1")
        ]),        
        dcc.Graph(
        id='graph1',
        figure=fig_1
        ),
        dbc.Row([
            dbc.Col(html.H5("After resample the date to make it balanced."),
            className="mr-1")
        ]), 
        dcc.Graph(
        id='graph2',
        figure=fig_2
        ),
        dbc.Row([
            dbc.Col(html.H5("3. Modeling"),
            className="mr-1")
        ]),
        dbc.Row([
            dbc.Col(html.P("Partition the data. This is done using the train_test_split() function in sklearn package."),
            className="mr-1")
        ]),
        dbc.Row([
            dbc.Col(html.H5("4. Choose the learning algorithm"),
            className="mr-1")
        ]),
        dbc.Row([
            dbc.Col(html.P("As the target not_fully_paid is discrete: 1 for not fully paid, and 0 for fully paid, the Lending Club problem is a classification problem."),
            className="mr-1")
        ]),
        dbc.Row([
            dbc.Col(html.H5("5. Fit your model"),
            className="mr-1")
        ]),
        dbc.Row([
            dbc.Col(html.H5("6. Evaluation"),
            className="mr-1")
        ]),
        dbc.Row([
            dbc.Col(html.P("The accuracy of Logistic is {}. The accuracy of RF is {}. Based on the accuracy score, we will use Random Forest for modeling and predicting.".format(accuracyLogstic,accuracyRF)),
            className="mx-1 my-1")
        ]),
        dbc.Row([
            dbc.Col(html.P("The importances of factors in RF model is {}. ".format(imRF.iloc[0:5])),
            className="mx-1 my-1")
        ]),
    ])
])





