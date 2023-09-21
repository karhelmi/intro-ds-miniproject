import numpy as np
import pandas as pd

data = pd.read_csv("emissions_CO2_2011-2021.csv", header=1)
print(data.head())
change_data = data[2021] / data[2011] -1
