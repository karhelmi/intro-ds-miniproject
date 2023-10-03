import numpy as np
import pandas as pd

# In MtCO2
# Source: https://globalcarbonatlas.org/
data_CO2 = pd.read_csv("emissions_CO2_2011-2021.csv", header=1)
data_CO2.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
data_CO2 = data_CO2.set_index('Year')
data_CO2 = data_CO2.drop([np.NaN, 'SOURCES', 'Territorial'])
data_CO2 = data_CO2.apply(pd.to_numeric, errors='coerce')
emission_reductions_CO2 = ((data_CO2.loc["2021"] - data_CO2.loc["2011"]) / data_CO2.loc["2011"]) * 100
print("Sorted based on MtCO2")
print(emission_reductions_CO2.sort_values().to_string())

# In kgCO2/GDP
# Source: https://globalcarbonatlas.org/
data_CO2_GDP = pd.read_csv("emissions_CO2_GDP_2011-2021.csv", header=1)
data_CO2_GDP.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
data_CO2_GDP = data_CO2_GDP.set_index('Year')
data_CO2_GDP = data_CO2_GDP.apply(pd.to_numeric, errors='coerce')
emission_reductions_CO2_GDP = ((data_CO2_GDP.loc["2021"] - data_CO2_GDP.loc["2011"]) / data_CO2_GDP.loc["2011"]) * 100
print("\nSorted based on kgCO2/GDP")
print(emission_reductions_CO2_GDP.sort_values().to_string())

# In tCO2/person
# Source: https://globalcarbonatlas.org/
data_CO2_person = pd.read_csv("emissions_CO2_person_2011-2021.csv", header=1)
data_CO2_person.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
data_CO2_person = data_CO2_person.set_index('Year')
data_CO2_person = data_CO2_person.apply(pd.to_numeric, errors='coerce')
emission_reductions_CO2_person = ((data_CO2_person.loc["2021"] - data_CO2_person.loc["2011"]) / data_CO2_person.loc["2011"]) * 100
print("\nSorted based on tCO2/person")
print(emission_reductions_CO2_person.sort_values().to_string())

# Function to get data of specific country
def filter_by_country(df, country):
    filtered_df = df[df['Entity'] == country]
    return filtered_df

# Original dataframe for Finland; new columns can be added to this
finland_df = data_CO2['Finland'].to_frame()
finland_df.rename(columns={'Finland': 'CO2_emissions'}, inplace=True)
finland_df.index = finland_df.index.astype(int)

# Source: https://ourworldindata.org/
electricity_renewables_df = pd.read_csv("share-electricity-renewables.csv")
electricity_renewables_df = electricity_renewables_df[(electricity_renewables_df['Year'] >= 2011) & (electricity_renewables_df['Year'] <= 2021)]

# Filter data from finland and merge it into original dataframe
finland_electricity_renewables_df = filter_by_country(electricity_renewables_df, 'Finland')
finland_electricity_renewables_df = finland_electricity_renewables_df.set_index('Year')
finland_electricity_renewables_df = finland_electricity_renewables_df['Renewables (% electricity)'].to_frame()
finland_electricity_renewables_df.index = finland_electricity_renewables_df.index.astype(int)
finland_df = finland_df.join(finland_electricity_renewables_df)

print("\nFinland dataframe")
print(finland_df)