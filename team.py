import dash_html_components as html
import dash_bootstrap_components as dbc


layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("About"), 
                html.P("We work for the 'LendingClub' company which specialises in lending various types of loans to urban customers. When the company receives a loan application, the company has to make a decision for loan approval based on the applicant’s profile. Two types of risks are associated with the bank’s decision: If the applicant is likely to repay the loan, then not approving the loan results in a loss of business to the companyIf the applicant is not likely to repay the loan, i.e. he/she is likely to default, then approving the loan may lead to a financial loss for the company The data given contains the information about past loan applicants and whether they ‘defaulted’ or not. The aim is to identify patterns which indicate if a person is likely to default, which may be used for taking actions such as denying the loan, reducing the amount of loan, lending (to risky applicants) at a higher interest rate, etc."),
                ],
                className='mx-2 my-2'
            )
        ]),
        dbc.Row([
            dbc.Col(html.H2("Group 6"), className='mx-2 my-2')
        ]),
        dbc.Row([
            dbc.Col(  
                dbc.Card(
                    dbc.Row([
                        dbc.Col(html.Img(src='static/img/HHY.jpg', className = "img-fluid"), className='col-3'),
                        dbc.Col([
                            html.H3('Haoyuan Huang'),
                            html.P('team member 1')
                            ],
                            className='col-9'
                        )
                    ])
                ),
                className="col-6 mx-2 my-2"
            )
        ]),
        dbc.Row([
            dbc.Col(  
                dbc.Card(
                    dbc.Row([
                        dbc.Col(html.Img(src='static/img/HH.jpg', className = "img-fluid"), className='col-3'),
                        dbc.Col([
                            html.H3('Hang He'),
                            html.P('team member 2')
                            ],
                            className='col-9'
                        )
                    ])
                ),
                className="col-6 mx-2 my-2"
            )
        ]),
        dbc.Row([
            dbc.Col(  
                dbc.Card(
                    dbc.Row([
                        dbc.Col(html.Img(src='static/img/QK.jpg', className = "img-fluid"), className='col-3'),
                        dbc.Col([
                            html.H3('Ken Qin'),
                            html.P('team member 3')
                            ],
                            className='col-9'
                        )
                    ])
                ),
                className="col-6 mx-2 my-2"
            )
        ])
    ])
])
