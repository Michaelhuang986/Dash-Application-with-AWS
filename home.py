import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash_core_components as dcc
import boto3
import io
import pandas as pd
import plotly.express as px



s3client = boto3.client("s3")
response = s3client.get_object(Bucket = "hw23-lab", Key = "loan_data_new.csv")
df = pd.read_csv(io.BytesIO(response['Body'].read()))

df_1 = df.groupby(['not_fully_paid']).count().reset_index()
df_2 = df.groupby(['not_fully_paid']).mean().reset_index()
df_3 = df.groupby(['not_fully_paid']).median().reset_index()


# Paid vs Not_fully_paid
fig_1 = px.bar(df_1, x="not_fully_paid", y= "credit_policy", title="Paid vs Not fully paid", \
    color = "not_fully_paid", labels={"credit_policy": "total number"})

fig_4 = px.bar(df_3, x="not_fully_paid", y= "int_rate", color=['0', '1'], color_discrete_map = {'German Shephard': 'rgb(255,0,0)'})


fig_6 = px.bar(df_3, x="not_fully_paid", y= "installment1000", color=['0', '1'])

fig_7 = px.bar(df_3, x="not_fully_paid", y= "fico_ratio", color=['0', '1'])

fig_8 = px.bar(df_3, x="not_fully_paid", y= "revol_util", color=['0', '1'])

fig_9 = px.pie(df_2, values='inq_last_6mths', names='not_fully_paid', template = 'seaborn').update_traces(hoverinfo='label', textinfo='value')

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Loan Paid Prediction Dashboard", className="text-center")
            #className="mx-1 my-1"
            )
            ]),
        dbc.Row([
            dbc.Col(html.P("Background LendingClub.com is a peer-to-peer lending platform. For many years, the company makes its anonymized lending data available to the public. This dataset covers 9,578 loans funded by the platform between May 2007 and February 2010. "),
            className="mx-1 my-1")
            ]),
        dbc.Row([
            dbc.Col(html.P("Columns in this dataset 1. credit_policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise. 2. purpose: The purpose of the loan such as: credit_card, debt_consolidation, etc. 3. int_rate: The interest rate of the loan (proportion).  4. installment: The monthly installments ($) owed by the borrower if the loan is funded. 5. log_annual_inc: The natural log of the annual income of the borrower. 6. dti: The debt-to-income ratio of the borrower. 7. fico: The FICO credit score of the borrower. 8. days_with_cr_line: The number of days the borrower has had a credit line. 9. revol_bal: The borrower’s revolving balance. 10. revol_util: The borrower’s revolving line utilization rate. 11. inq_last_6mths: The borrower’s number of inquiries by creditors in the last 6 months. 12. delinq_2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years. 13. pub_rec: The borrower’s number of derogatory public records. 14. not_fully_paid: indicates whether the loan was not paid back in full (the borrower either defaulted or the borrower was deemed unlikely to pay it back)."),
            className="mx-1 my-1")
            ]),
        dbc.Row([
            dbc.Col(html.H1("----------------------------------------------------------------------------------------------------------------------------------------"),
            className="mx-1 my-1")
            ]),
        dcc.Graph(
            id='graph1',
            figure=fig_1
            ),
        dbc.Row([
            dbc.Col(html.P("As the figure shown, the model predicted that most customers would not pay back the loan in full."),
            className="mx-1 my-1")
            ]),
        dbc.Row([
            dbc.Col(html.H1("----------------------------------------------------------------------------------------------------------------------------------------"),
            className="mx-1 my-1")
            ]),

        dcc.Graph(
            id='graph3',
            figure=fig_4
            ),
        dbc.Row([
           dbc.Col(html.P("From the graph, we find that there is a difference in the median interest rate between paid customers and not paid customers. The lower interest rate is, the higher chance that customers would like to pay the loan."),
           className="mx-1 my-1")
           ]),
        dcc.Graph(
            id='graph6',
            figure=fig_6
            ),
        dbc.Row([
           dbc.Col(html.P("From the graph, we find that there is a difference in the median monthly installment between paid customers and not paid customers. The monthly installment of paid customers is lower than the monthly installment of unpaid customers."),
           className="mx-1 my-1")
           ]),
        dcc.Graph(
            id='graph7',
            figure=fig_7
            ),
        dbc.Row([
            dbc.Col(html.P("Our prediction shows that people who will default have a lower Fico Score (fico_ratio*850) than people who will pay back loan in full."),
            className="mx-1 my-1")
            ]),
        dcc.Graph(
            id='graph8',
            figure=fig_8
            ),
        dbc.Row([
            dbc.Col(html.P("Our prediction shows that people who will default have a higher revolving line utilization rate than people who will pay back loan in full."),
            className="mx-1 my-1")
            ]),
        dcc.Graph(
            id='graph9',
            figure=fig_9
            ),
        dbc.Row([
            dbc.Col(html.P("Among customers who will default, they have more than 3 inquiries in the last 6 months. By contrast, customers who will pay back loan have 1 or no inquiries."),
            className="mx-1 my-1")
            ]),
        ])

    ])
