import pandas as pd


matches_data = pd.read_csv("data/manuited_dataset.csv")

print("Shape of data :",matches_data.shape)
print("Head of Data:\n")
print(matches_data.head())
print("")