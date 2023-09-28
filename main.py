import numpy as np
import pandas as pd

#In MtCO2
data_CO2 = pd.read_csv("emissions_CO2_2011-2021.csv", header=1) #source: https://globalcarbonatlas.org/
data_CO2.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
data_CO2 = data_CO2.set_index('Year')
data_CO2 = data_CO2.drop([np.NaN,'SOURCES','Territorial'])
data_CO2 = data_CO2.apply(pd.to_numeric, errors='coerce')
emission_reductions_CO2 = ((data_CO2.loc["2021"] - data_CO2.loc["2011"]) / data_CO2.loc["2011"]) * 100
#print("Sorted based on MtCO2")
#print(emission_reductions_CO2.sort_values().to_string())
print(data_CO2)

#In kgCO2/GDP
data_CO2_GDP = pd.read_csv("emissions_CO2_GDP_2011-2021.csv", header=1) #source: https://globalcarbonatlas.org/
data_CO2_GDP.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
data_CO2_GDP = data_CO2_GDP.set_index('Year')
data_CO2_GDP = data_CO2_GDP.apply(pd.to_numeric, errors='coerce')
emission_reductions_CO2_GDP = ((data_CO2_GDP.loc["2021"] - data_CO2_GDP.loc["2011"]) / data_CO2_GDP.loc["2011"]) * 100
#print("\nSorted based on kgCO2/GDP")
#print(emission_reductions_CO2_GDP.sort_values().to_string())

#In tCO2/person
data_CO2_person = pd.read_csv("emissions_CO2_person_2011-2021.csv", header=1) #source: https://globalcarbonatlas.org/
data_CO2_person.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
data_CO2_person = data_CO2_person.set_index('Year')
data_CO2_person = data_CO2_person.apply(pd.to_numeric, errors='coerce')
emission_reductions_CO2_person = ((data_CO2_person.loc["2021"] - data_CO2_person.loc["2011"]) / data_CO2_person.loc["2011"]) * 100
#print("\nSorted based on tCO2/person")
#print(emission_reductions_CO2_person.sort_values().to_string())

data_electricity_renewables = pd.read_csv("share-electricity-renewables.csv")
data_electricity_renewables.drop('Code', axis=1, inplace=True)
data_electricity_renewables = data_electricity_renewables[(data_electricity_renewables['Year'] >= 2011) & (data_electricity_renewables['Year'] <= 2021)]
#print(data_electricity_renewables)

def filter_by_country(df, country):
    filtered_df = df[df['Entity'] == country]
    return filtered_df

filtered_finland_df = filter_by_country(data_electricity_renewables, 'Finland')
filtered_finland_df = filtered_finland_df.set_index('Year')
#print(filtered_finland_df)

filtered_finland_df['CO2_emissions'] = data_CO2.loc['2011':'2021']['Finland'].index.astype('int64')
#merged_df = filtered_finland_df.merge(data_CO2['Finland'], left_index=True, right_index=True, how='inner')
#print(data_CO2['Finland'].index.dtype)
#print(merged_df)
print(filtered_finland_df)