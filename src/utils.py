# Util Imports
from fileinput import filename
import os
import numpy as np
import pandas as pd

def read_data(file_name = None, index_col=False):
    """
    Read data from excel or csv file.
    """
    ext = file_name.split(".")[-1]
    assert ext in ["csv", "xlsx"]
    if ext == "csv":
        data_df = pd.read_csv(file_name, index_col=index_col)
    else:
        data_df = pd.read_excel(file_name, index_col=index_col)

    return data_df


def evaluate_performance(TP, FN , FP , TN):
    """
    Evaluate performance.

    INPUT:
    TP: # True Positive 
    FN: # False Negatives
    FP: # False Positives
    TN: # True Negatives

    RETURN:
    PERFORMANCE: Dictionary with following performance metrics as keys.
    WA: Weighted Accuracy
    ACCU: Accuracy
    SEN: Sensitivity
    SPE: Specificity
    GM: Geometric Mean
    PRECISION: Precision
    RECALL: Recall
    F1: F1-Score
    """
    try:
        PERFORMANCE = {}
        POS = (FN+TP)
        NEG = (FP+TN)
        TOTAL = POS+NEG

        wt = round(NEG/POS, 3)
        neu = TN+(wt*TP)
        den = NEG+(POS*wt)


        PERFORMANCE["WA"] = round(neu/den*100, 2)
        PERFORMANCE["ACCU"] = round((TP+TN)/TOTAL*100, 2)
        PERFORMANCE["SEN"] = round(TP/POS*100, 2)
        PERFORMANCE["SPE"] = round(TN/NEG*100, 2)
        PERFORMANCE["GM"] = round(np.sqrt(PERFORMANCE["SPE"]*PERFORMANCE["SEN"]),2)
        PERFORMANCE["PRECISION"] = round(TP/(TP+FP)*100, 2)
        PERFORMANCE["RECALL"] = round(TP/POS*100, 2)
        PERFORMANCE["F1"] = round((2*PERFORMANCE["PRECISION"]*PERFORMANCE["RECALL"])/(PERFORMANCE["PRECISION"]+PERFORMANCE["RECALL"]), 2)
    except Exception as ex:
        print("ERROR: while calculating performance metrics.", str(ex))
        PERFORMANCE["WA"] = 0
        PERFORMANCE["ACCU"] = 0
        PERFORMANCE["SEN"] = 0
        PERFORMANCE["SPE"] = 0
        PERFORMANCE["GM"] = 0
        PERFORMANCE["PRECISION"] = 0
        PERFORMANCE["RECALL"] = 0
        PERFORMANCE["F1"] = 0
        
    return PERFORMANCE