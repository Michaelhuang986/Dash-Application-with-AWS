import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class DataFactory:
    
    def Normalization(self,filename):
        Input = filename
        Input['installment1000'] = Input.installment / 1000
        Input.drop('installment', axis=1, inplace=True)
        Input['fico_ratio'] = Input.fico / 850
        Input.drop('fico', axis=1, inplace=True)
        Input['decades_with_cr_line'] = Input.days_with_cr_line / 3650
        Input.drop('days_with_cr_line', axis=1, inplace=True)
        Input['log_revol_bal'] = np.log(Input.revol_bal + 1)
        Input.drop('revol_bal', axis=1, inplace=True)
        Input.revol_util = Input.revol_util / 100
        Input = pd.get_dummies(Input, columns=['purpose'])
        Input = Input.drop(columns = ['purpose_all_other'])
        #Input = Input.drop(columns = ['purpose_debt_consolidation'])
        return Input

    def _XySplite(self,filename):
        fileNormed=self.Normalization(filename)
        X = fileNormed.drop(columns=['not_fully_paid'])
        y = fileNormed['not_fully_paid']
        return X, y

    def _DataSplit(self,filename,size, rand):
        X, y = self._XySplite(filename)
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=size, random_state=rand)
        return X_train, X_test, y_train, y_test

    def TrainModel(self,filename):
        X_train, X_test, y_train, y_test = self._DataSplit(filename, 0.2, 1234)
        model = RandomForestClassifier(random_state=1,max_depth=4).fit(X_train,y_train)
        return model
