import numpy as np
import pandas as pd

data = pd.read_csv("emissions_CO2_2011-2021.csv", header=1)
data.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
data = data.set_index('Year')
data = data.apply(pd.to_numeric, errors='coerce')

selected_data = data.loc[["2011", "2021"]]
emission_reductions = ((selected_data.loc["2021"] - selected_data.loc["2011"]) / selected_data.loc["2011"]) * 100

print(emission_reductions)