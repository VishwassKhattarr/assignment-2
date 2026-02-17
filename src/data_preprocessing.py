import pandas as pd
import numpy as np

def load_no2_data(path):
    df = pd.read_csv(path, encoding="latin1")


   
    print("Columns:", df.columns)

    
    x = df["no2"]


    
    x = x.dropna().values.astype(float)

    return x


if __name__ == "__main__":
    x = load_no2_data("../data/india_air_quality.csv")
    print("Sample NO2 values:", x[:10])
    print("Total samples:", len(x))
