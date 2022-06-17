import numpy as np
import pandas as pd

def checkNaN(df):
    print("Missing Value List")
    for col in df.columns:
            if df[col].isnull().sum():
                print(f"{col} : {df[col].isnull().sum()} ")

def dropNaN(df):
    print("\nPre-Processing...")
    for i in df.columns.to_list() :
        dpIdx = df[df[i].isnull()==True].index
        df.drop(index=dpIdx, inplace=True)
    print(f"{len(dpIdx)} Columns Dropped.")    
    return df