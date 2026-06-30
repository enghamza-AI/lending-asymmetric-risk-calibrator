# data_loader.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import os


def load_config(config_path="config/business_rules.yaml"):
   
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config



POSITIVE_LABEL = "Charged Off"   # default  → 1
NEGATIVE_LABEL = "Fully Paid"    # repaid   → 0


COLUMNS_TO_KEEP = [
    "loan_amnt",        # amount requested — known at application
    "int_rate",         # interest rate assigned — known at approval
    "installment",      # monthly payment amount
    "annual_inc",       # self-reported income
    "dti",              # debt-to-income ratio
    "delinq_2yrs",      # delinquencies in past 2 years
    "inq_last_6mths",   # hard credit inquiries last 6 months
    "open_acc",         # number of open credit lines
    "pub_rec",          # public derogatory records (bankruptcies etc)
    "revol_bal",        # total revolving balance across all cards
    "revol_util",       # revolving utilization % — how maxed out cards are
    "total_acc",        # total credit lines ever opened
    "emp_length",       # years employed — text like "10+ years"
    "home_ownership",   # RENT / OWN / MORTGAGE — categorical
    "grade",            # Lending Club internal grade A through G
    "purpose",          # reason for loan — debt_consolidation, etc
    "loan_status",      # TARGET — what we predict (drop after encoding)
]



def load_raw_chunks(config):
 

    raw_path = config["data"]["raw_path"]
    chunksize = 50000  

   

    print(f"Loading data from: {raw_path}")
    print(f"Reading in chunks of {chunksize:,} rows...")

    chunks = []  



    reader = pd.read_csv(
        raw_path,
        usecols=COLUMNS_TO_KEEP,    
        chunksize=chunksize,
        low_memory=False            
    )

    for i, chunk in enumerate(reader):

        
        completed = chunk[
            chunk["loan_status"].isin([POSITIVE_LABEL, NEGATIVE_LABEL])
        ]

        
        if len(completed) > 0:
            chunks.append(completed)

       
        if (i + 1) % 10 == 0:
            rows_processed = (i + 1) * chunksize
            print(f"  Processed ~{rows_processed:,} rows...")

    print("Chunked loading complete. Combining filtered chunks...")

    
    df = pd.concat(chunks, ignore_index=True)

    print(f"Total completed loans found: {len(df):,}")
    return df




def encode_target(df):
 

    df["target"] = (df["loan_status"] == POSITIVE_LABEL).astype(int)

   
    df = df.drop(columns=["loan_status"])

    
    print("\nTarget distribution after encoding:")
    counts = df["target"].value_counts()
    pct = df["target"].value_counts(normalize=True) * 100
    print(f"  Class 0 (Fully Paid):   {counts[0]:,} ({pct[0]:.1f}%)")
    print(f"  Class 1 (Charged Off):  {counts[1]:,} ({pct[1]:.1f}%)")
    print(f"  Imbalance ratio: {counts[0]/counts[1]:.1f}:1")

    

    return df


def stratified_sample(df, config):


    sample_size = config["data"]["sample_size"]
    random_state = config["data"]["random_state"]

    
    if len(df) <= sample_size:
        print(f"\nDataset ({len(df):,} rows) smaller than sample_size. Using all rows.")
        return df

    print(f"\nStratified sampling {sample_size:,} rows from {len(df):,}...")

   

    fraction = sample_size / len(df)

    sampled = (
        df.groupby("target", group_keys=False)
        .apply(lambda x: x.sample(frac=fraction, random_state=random_state))
    )

    sampled = sampled.reset_index(drop=True)
    print(f"Sample size: {len(sampled):,} rows")

    return sampled




def split_data(df, config):
 
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y             
    )

    print(f"\nTrain set: {len(X_train):,} rows")
    print(f"Test set:  {len(X_test):,} rows")
    print(f"Train default rate: {y_train.mean():.3f}")
    print(f"Test default rate:  {y_test.mean():.3f}")

  

    return X_train, X_test, y_train, y_test




def load_data(config_path="config/business_rules.yaml"):

    print("=" * 50)
    print("STARTING DATA LOADER")
    print("=" * 50)

    
    config = load_config(config_path)

    
    df = load_raw_chunks(config)

    
    df = encode_target(df)

    
    df = stratified_sample(df, config)

    
    X_train, X_test, y_train, y_test = split_data(df, config)

    print("\n" + "=" * 50)
    print("DATA LOADER COMPLETE")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print("=" * 50)

    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print("\nSample of X_train:")
    print(X_train.head())
    print("\nSample of y_train:")
    print(y_train.value_counts())