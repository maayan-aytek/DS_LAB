from xgboost import XGBClassifier
import argparse
import os
import pickle
import pandas as pd
import numpy as np

def concat_all_patients(file_path):
    directory = os.listdir(file_path)
    all_df = pd.DataFrame()
    for filename in directory:
        with open(os.path.join(file_path, filename), 'r') as f:
            df = pd.read_csv(f, sep='|')
            # index = df.index[df['SepsisLabel'] == 1].min()
            # if not math.isnan(index):
            #     df = df.loc[:index]
            df['patient id'] = filename[:-4]
            all_df = all_df.append(df, ignore_index=True)

    return all_df
def outliers_perc_of_col(target, df):
    df_not_null = df[target].dropna()
    Q1 = np.percentile(df_not_null, 25)
    Q3 = np.percentile(df_not_null, 75)
    IQR = Q3 - Q1
    ul = Q3 + 1.5 * IQR
    ll = Q1 - 1.5 * IQR
    outliers = df_not_null[(df_not_null > ul) | (df_not_null < ll)]
    outliers_per = round((len(outliers)/len(df_not_null))*100,2)
    return outliers_per


def pre_process(df, type_df):
    patients_size = df.groupby("patient id").size()
    patients_size = patients_size[patients_size >= 36]
    full_hours_df = df[df['patient id'].isin(patients_size.index)].reset_index(drop=True)

    # backward and forward fill
    df_processed = df.groupby('patient id').apply(lambda x: x.bfill().ffill()).reset_index(drop=True)
    outlier_perc_dict = {}
    # Filling the rest of null values based on median/mean/mode according to the feature's outliers
    for col in df_processed.columns:
        if col != "patient id":
            outlier_perc_dict[col] = [outliers_perc_of_col(col, df_processed)]
    outliers_perc_df = pd.DataFrame(outlier_perc_dict).T
    outliers_perc_df.columns = ["Outliers %"]
    outliers_perc_df = outliers_perc_df.sort_values(by=['Outliers %'], ascending=False)
    columns = df_processed.columns
    for col in columns:
        if col != "patient id":
            median = full_hours_df[col].median()
            df_processed[col] = df_processed[col].fillna(median)
            avg = full_hours_df[col].mean()
            mode = full_hours_df[col].mode()[0]
            outlier_perc = outliers_perc_df.loc[col, 'Outliers %']
            if df_processed[col].nunique() == 2:
                df_processed[col] = df_processed[col].fillna(mode)
            else:
                if outlier_perc <= 5:
                    df_processed[col] = df_processed[col].fillna(avg)
                elif outlier_perc > 5:
                    df_processed[col] = df_processed[col].fillna(median)

    # data transformations
    condition_temp = (df_processed['Temp'] > 38) | (df_processed['Temp'] < 36)
    condition_HR = (df_processed['HR'] > 90)
    condition_Resp = (df_processed['Resp'] > 20) | (df_processed['PaCO2'] < 32)
    condition_wbc = (df_processed['WBC'] * 10 ** 3 > 12000) | (df_processed['WBC'] * 10 ** 3 < 4000)
    df_processed['SIRS'] = np.where(condition_temp, 1, 0) + \
                           np.where(condition_HR, 1, 0) + np.where(condition_Resp, 1, 0) + \
                           np.where(condition_wbc, 1, 0)
    df_processed['SIRS'] = np.where(df_processed['SIRS'] >= 2, 1, 0)

    df_processed["BUN_Creatinine_ratio"] = df_processed["BUN"] / df_processed["Creatinine"]
    df_processed["qSOFA"] = np.where((df_processed['SBP'] >= 100) & (df_processed['Resp'] >= 22), 1, 0)

    # Feature selection
    cols = ["ICULOS", "Lactate", "Temp", "SIRS", "PaCO2", "Unit1", "FiO2", "AST", "pH", "Hgb", "WBC", "BUN", "Calcium",
            "BaseExcess",
            "HospAdmTime", "patient id", "Hct", "HCO3", "HR", "MAP", "SaO2", "Alkalinephos", "Magnesium", "Potassium",
            "Bilirubin_total", "Phosphate", "O2Sat", "BUN_Creatinine_ratio", "PTT", "SBP", "SepsisLabel"]
    df_processed = df_processed[cols]
    # Aggregations - remain only one record for each patient
    groups = df_processed.groupby('patient id')
    # Initialize an empty DataFrame to store the results
    results = pd.DataFrame(columns=df_processed.columns)
    # Loop through the groups
    for name, group in groups:
        # Check if there is at least one row with label 1
        if (group['SepsisLabel'] == 1).any():
            # If there is, select the first row with label 1
            row = group[group['SepsisLabel'] == 1].iloc[0]
        else:
            # If there is no row with label 1, select the last row
            row = group.iloc[-1]

        # Add the selected row to the results DataFrame
        results.loc[name] = row

    # undersampling
    if type_df == "train":
        majority_class = results[results['SepsisLabel'] == 0]
        minority_class = results[results['SepsisLabel'] == 1]
        majority_subset = majority_class.sample(n=5 * len(minority_class), random_state=1000)
        results = results[(results['SepsisLabel'] == 1) | (
                    (results['SepsisLabel'] == 0) & (results.index.isin(majority_subset.index)))].reset_index(drop=True)

    to_model_df = results.drop('patient id', axis=1)
    to_model_df = to_model_df.apply(pd.to_numeric)
    X = to_model_df.drop('SepsisLabel', axis=1)
    y = to_model_df["SepsisLabel"]

    return X, y, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='File path prediction')
    parser.add_argument('filepath', type=str, help='Path to the file for prediction')
    args = parser.parse_args()
    train_df = concat_all_patients(args.filepath)
    X_train, y_train, train_df = pre_process(train_df, type_df="train")
    model = XGBClassifier(n_estimators=200, max_depth=50, eta=0.05, gamma=0.1, reg_lambda=0.8, min_child_weight=2)
    model.fit(X_train, y_train)
    # Saving model to pickle
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)