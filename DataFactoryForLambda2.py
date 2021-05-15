import numpy as np
import pandas as pd
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
        return Input