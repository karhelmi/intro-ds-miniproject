import numpy as np
import pandas as pd

#In MtCO2
data_CO2 = pd.read_csv("emissions_CO2_2011-2021.csv", header=1) #source: https://globalcarbonatlas.org/
data_CO2.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
data_CO2 = data_CO2.set_index('Year')
data_CO2 = data_CO2.apply(pd.to_numeric, errors='coerce')
emission_reductions_CO2 = ((data_CO2.loc["2021"] - data_CO2.loc["2011"]) / data_CO2.loc["2011"]) * 100
print("Sorted based on MtCO2")
print(emission_reductions_CO2.sort_values().to_string())

#In kgCO2/GDP
data_CO2_GDP = pd.read_csv("emissions_CO2_GDP_2011-2021.csv", header=1) #source: https://globalcarbonatlas.org/
data_CO2_GDP.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
data_CO2_GDP = data_CO2_GDP.set_index('Year')
data_CO2_GDP = data_CO2_GDP.apply(pd.to_numeric, errors='coerce')
emission_reductions_CO2_GDP = ((data_CO2_GDP.loc["2021"] - data_CO2_GDP.loc["2011"]) / data_CO2_GDP.loc["2011"]) * 100
print("\nSorted based on kgCO2/GDP")
print(emission_reductions_CO2_GDP.sort_values().to_string())

#In tCO2/person
data_CO2_person = pd.read_csv("emissions_CO2_person_2011-2021.csv", header=1) #source: https://globalcarbonatlas.org/
data_CO2_person.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
data_CO2_person = data_CO2_person.set_index('Year')
data_CO2_person = data_CO2_person.apply(pd.to_numeric, errors='coerce')
emission_reductions_CO2_person = ((data_CO2_person.loc["2021"] - data_CO2_person.loc["2011"]) / data_CO2_person.loc["2011"]) * 100
print("\nSorted based on tCO2/person")
print(emission_reductions_CO2_person.sort_values().to_string())