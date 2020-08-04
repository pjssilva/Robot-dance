""""
But covid data from init to end.
"""

import pandas as pd
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="name of imput file")
    parser.add_argument("output", help="name of imput file")
    parser.add_argument('--init', dest='init', action='store', default="2020-01-01",
        help="Initial date (YYYY-mm-dd)")
    parser.add_argument('--end', dest='end', action='store', default="2030-01-01",
        help="Final date (YYYY-mm-dd)")
    args = parser.parse_args()
    args.init = pd.to_datetime(args.init)
    args.end = pd.to_datetime(args.end)
    return args


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    df = pd.read_csv(args.input, index_col=0)
    df["date"] = df["date"].astype("datetime64")
    df = df.loc[df["date"] >= args.init, :]
    df = df.loc[df["date"] <= args.end, :]
    df.reset_index(drop=True, inplace=True)
    print(df)
    df.to_csv(args.output)
