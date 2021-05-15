import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

class DataFactory:
    
    def DataReader(self,Filename):
        outPut = pd.read_csv(Filename)
        return outPut

    def GetDummie(self,Input):
        Input= pd.get_dummies(Input, columns=['purpose'])
        Input = Input.drop(columns = ['purpose_all_other'])
        return Input

    def Normalization(self,file):
        Input=file
        Input['installment1000'] = Input.installment / 1000
        Input.drop('installment', axis=1, inplace=True)
        Input['fico_ratio'] = Input.fico / 850
        Input.drop('fico', axis=1, inplace=True)
        Input['decades_with_cr_line'] = Input.days_with_cr_line / 3650
        Input.drop('days_with_cr_line', axis=1, inplace=True)
        Input['log_revol_bal'] = np.log(Input.revol_bal + 1)
        Input.drop('revol_bal', axis=1, inplace=True)
        Input.revol_util = Input.revol_util / 100
        if 'purpose' in Input.columns:
            Input=self.GetDummie(Input)
        return Input

    def _XySplite(self,Input):
        fileNormed=self.Normalization(Input)
        X = fileNormed.drop(columns=['not_fully_paid'])
        y = fileNormed['not_fully_paid']
        return X, y

    def DataSplit(self,Input,size, rand):
        X, y = self._XySplite(Input)
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=size, random_state=rand)
        return X_train, X_test, y_train, y_test

    def TrainModelLogistic(self,X_train, X_test, y_train, y_test):
        model = LogisticRegression(penalty='l2',random_state=1,solver='lbfgs', max_iter=1000)
        model.fit(X_train,y_train)
        importances=pd.Series(np.transpose(model.coef_.tolist()[0]), index=X_train.columns)
        joblib.dump(model, 'LOAN_MODEL')
        return importances

    def TrainModelRF(self,X_train, X_test, y_train, y_test):
        model = RandomForestClassifier(random_state=1,max_depth=5)
        model.fit(X_train,y_train)
        joblib.dump(model, 'LOAN_MODEL')
        importances = model.feature_importances_
        importances=pd.Series(importances, index=X_train.columns)
        return importances

    def Evaluate(self,X_train, X_test, y_train, y_test):
        model = joblib.load('LOAN_MODEL')
        X_train, X_test, y_train, y_test=X_train, X_test, y_train, y_test
        y_predict = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_predict).round(4)
        return accuracy

class Predictor:

    def ApplyModel(self,test_csv):  

        dataFactory=DataFactory()
        df_X=dataFactory.Normalization(test_csv)
        model = joblib.load('LOAN_MODEL')
        y_predicted = model.predict(df_X)

        df_X['not_fully_paid'] = y_predicted
        return df_X