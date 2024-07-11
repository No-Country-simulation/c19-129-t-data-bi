import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def get_outliers(df = pd.DataFrame, implace=False , q1= 0.25 , q3= 0.75):
    '''
    Function to detect outliers in a DataFrame
    Parameters:
    df: DataFrame
    implace: bool, default False (if True, remove the outliers from the DataFrame)
    q1: float, default 0.25
    q3: float, default 0.75
    Returns:
    DataFrame
    '''
    outliers_column = df.select_dtypes(include=[np.number]).columns
    for column in outliers_column:
        Q1 = df[column].quantile(q1)
        Q3 = df[column].quantile(q3)
        IQR = Q3 - Q1
        outliers = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
        print(f'Outliers in {column}: {outliers.sum()}')
        if implace:
            df = df[~outliers]
    return df    