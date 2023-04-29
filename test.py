import pandas as pd

if __name__ == '__main__':
    all_df = pd.read_pickle("all_df.pkl")
    print(all_df.dtypes)