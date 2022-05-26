import pandas as pd
import numpy as np
def remove_nan(csv):
    df = pd.read_csv(csv, delimiter='\t', encoding='utf-8')
    df = df[['dr_id', 'dialog_id', 'q', 'parsed_dialog']]
    print("orginal df: ", df.shape)
    df = df[~pd.isna(df["parsed_dialog"])]
    print("after clean df: ", df.shape)
    df.to_csv(csv.replace(".csv", "") + '_cleaned.csv', index=False)
    return df

def main():
    remove_nan("embed.csv")
    remove_nan("train.csv")
    remove_nan("test.csv")
    remove_nan("valid.csv")


if __name__ == '__main__':
    main()
