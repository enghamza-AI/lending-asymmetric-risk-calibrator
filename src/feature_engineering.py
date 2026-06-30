# feature_engineering.py


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
import joblib
import os




NUMERICAL_COLS = [
    "loan_amnt",
    "int_rate",       
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",    
    "total_acc",
]



ORDINAL_COLS = ["grade", "emp_length"]


GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]


EMP_LENGTH_ORDER = [
    "< 1 year", "1 year", "2 years", "3 years", "4 years",
    "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"
]



NOMINAL_COLS = ["home_ownership", "purpose"]




def clean_raw_features(X):
   

    X = X.copy()

  

    if "emp_length" in X.columns:
        X["emp_length"] = X["emp_length"].fillna("< 1 year")
        X["emp_length"] = X["emp_length"].replace("n/a", "< 1 year")
        # strip extra whitespace — common in real datasets
        X["emp_length"] = X["emp_length"].str.strip()

    
    if "grade" in X.columns:
        X["grade"] = X["grade"].str.strip()

 

    if "home_ownership" in X.columns:
        rare_values = ["ANY", "NONE"]
        X["home_ownership"] = X["home_ownership"].replace(rare_values, "OTHER")
        X["home_ownership"] = X["home_ownership"].str.strip()

    return X




def build_pipeline():
    """
    Builds and returns the full feature engineering pipeline.
    Does NOT fit it — fitting happens in engineer_features().

    RETURNS:
        preprocessor (ColumnTransformer): unfitted pipeline ready to fit
    """

    

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

   

    ordinal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[GRADE_ORDER, EMP_LENGTH_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

 

    nominal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])



    preprocessor = ColumnTransformer(transformers=[
        ("numerical", numerical_pipeline, NUMERICAL_COLS),
        ("ordinal",   ordinal_pipeline,   ORDINAL_COLS),
        ("nominal",   nominal_pipeline,   NOMINAL_COLS),
    ], remainder="drop")

    return preprocessor




def engineer_features(X_train, X_test):


    print("=" * 50)
    print("STARTING FEATURE ENGINEERING")
    print("=" * 50)


    print("Cleaning raw features...")
    X_train_clean = clean_raw_features(X_train)
    X_test_clean  = clean_raw_features(X_test)

   
    print("Building preprocessing pipeline...")
    preprocessor = build_pipeline()

    print("Fitting pipeline on training data...")
    X_train_processed = preprocessor.fit_transform(X_train_clean)

    
    print("Transforming test data...")
    X_test_processed = preprocessor.transform(X_test_clean)

 
    feature_names = get_feature_names(preprocessor)

    print(f"\nOriginal features:    {X_train.shape[1]}")
    print(f"Engineered features:  {X_train_processed.shape[1]}")
    print(f"Train matrix shape:   {X_train_processed.shape}")
    print(f"Test matrix shape:    {X_test_processed.shape}")
    print("=" * 50)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 50)

    return X_train_processed, X_test_processed, feature_names, preprocessor




def get_feature_names(preprocessor):

    feature_names = []

    for name, pipeline, columns in preprocessor.transformers_:
        if name == "numerical":
            feature_names.extend(columns)
        elif name == "ordinal":
            feature_names.extend(columns)
        elif name == "nominal":
          
            encoder = pipeline.named_steps["encoder"]
            ohe_names = encoder.get_feature_names_out(columns)
            feature_names.extend(ohe_names)

    return feature_names


def save_preprocessor(preprocessor, path="outputs/preprocessor.joblib"):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved to: {path}")


def load_preprocessor(path="outputs/preprocessor.joblib"):
  
    return joblib.load(path)




if __name__ == "__main__":

 

    print("Running feature engineering smoke test with fake data...")

    fake_train = pd.DataFrame({
        "loan_amnt":      [5000, 10000, 15000, 8000, 12000],
        "int_rate":       [10.5, 15.2, 13.1, 8.9, 20.0],
        "installment":    [150, 300, 450, 200, 350],
        "annual_inc":     [50000, 75000, None, 90000, 60000],
        "dti":            [15.0, 22.5, 10.0, None, 30.0],
        "delinq_2yrs":    [0, 1, 0, 0, 2],
        "inq_last_6mths": [1, 2, 0, 1, 3],
        "open_acc":       [5, 8, 12, 6, 9],
        "pub_rec":        [0, 0, 1, 0, 0],
        "revol_bal":      [8000, 15000, 5000, 12000, 20000],
        "revol_util":     [45.0, 70.0, None, 30.0, 85.0],
        "total_acc":      [12, 20, 15, 18, 25],
        "emp_length":     ["5 years", "10+ years", "< 1 year", None, "3 years"],
        "home_ownership": ["RENT", "MORTGAGE", "OWN", "RENT", "MORTGAGE"],
        "grade":          ["A", "C", "B", "D", "E"],
        "purpose":        ["debt_consolidation", "car", "medical", "home_improvement", "other"],
    })

    fake_test = fake_train.copy().iloc[:2]

    X_train_p, X_test_p, names, pipeline = engineer_features(fake_train, fake_test)

    print(f"\nFeature names ({len(names)} total):")
    for i, name in enumerate(names):
        print(f"  {i:2d}: {name}")

    print("\nFirst row of processed training data:")
    print(X_train_p[0])

    print("\nSmoke test passed. Pipeline is working correctly.")