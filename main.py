import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# In MtCO2
# Source: https://globalcarbonatlas.org/
data_CO2 = pd.read_csv("csv_data/0_emissions_CO2_2011-2021.csv", header=1)
data_CO2.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
data_CO2 = data_CO2.set_index('Year')
data_CO2 = data_CO2.drop([np.NaN, 'SOURCES', 'Territorial'])
data_CO2 = data_CO2.apply(pd.to_numeric, errors='coerce')
emission_reductions_CO2 = ((data_CO2.loc["2021"] - data_CO2.loc["2011"]) / data_CO2.loc["2011"]) * 100
print("Sorted based on MtCO2")
print(emission_reductions_CO2.sort_values().to_string())

# In kgCO2/GDP
# Source: https://globalcarbonatlas.org/
data_CO2_GDP = pd.read_csv("csv_data/emissions_CO2_GDP_2011-2021.csv", header=1)
data_CO2_GDP.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
data_CO2_GDP = data_CO2_GDP.set_index('Year')
data_CO2_GDP = data_CO2_GDP.apply(pd.to_numeric, errors='coerce')
emission_reductions_CO2_GDP = ((data_CO2_GDP.loc["2021"] - data_CO2_GDP.loc["2011"]) / data_CO2_GDP.loc["2011"]) * 100
print("\nSorted based on kgCO2/GDP")
print(emission_reductions_CO2_GDP.sort_values().to_string())

# In tCO2/person
# Source: https://globalcarbonatlas.org/
data_CO2_person = pd.read_csv("csv_data/emissions_CO2_person_2011-2021.csv", header=1)
data_CO2_person.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
data_CO2_person = data_CO2_person.set_index('Year')
data_CO2_person = data_CO2_person.apply(pd.to_numeric, errors='coerce')
emission_reductions_CO2_person = ((data_CO2_person.loc["2021"] - data_CO2_person.loc["2011"]) / data_CO2_person.loc["2011"]) * 100
print("\nSorted based on tCO2/person")
print(emission_reductions_CO2_person.sort_values().to_string())

# Dictionary of the csv files; filename as key and wanted column as value
csv_column_dict = {
    'csv_data/1_global-meat-production.csv': 'Meat, total | 00001765 || Production | 005510 || tonnes',
    'csv_data/2_life-expectancy.csv': 'Life expectancy at birth (historical)', #NEW
    'csv_data/3_gdp-per-capita-worldbank.csv': 'GDP per capita, PPP (constant 2017 international $)',
    'csv_data/4_share-of-individuals-using-the-internet.csv': 'Individuals using the Internet (% of population)',
    'csv_data/5_human-development-index.csv': 'Human Development Index',
    'csv_data/6_human-rights-index-vdem.csv': 'civ_libs_vdem_owid', # Central estimate chosen 10.10.23
    'csv_data/7_population-and-demography.csv': 'Population', #NEW
    'csv_data/8_nuclear-energy-generation.csv': 'Electricity from nuclear (TWh)', 
    'csv_data/9_per-capita-energy-use.csv': 'Primary energy consumption per capita (kWh/person)',
    'csv_data/10_share-electricity-renewables.csv': 'Renewables (% electricity)'
}

# Variables we want to include to a dataframe of a country; Uncomment to include in the dataframe
variable_csv_list = [
    'csv_data/1_global-meat-production.csv',
    'csv_data/2_life-expectancy.csv',
    'csv_data/3_gdp-per-capita-worldbank.csv',
    'csv_data/4_share-of-individuals-using-the-internet.csv',
    'csv_data/5_human-development-index.csv',
    'csv_data/6_human-rights-index-vdem.csv',
    'csv_data/7_population-and-demography.csv',
    'csv_data/8_nuclear-energy-generation.csv',
    'csv_data/9_per-capita-energy-use.csv',
    'csv_data/10_share-electricity-renewables.csv'
]

# Create original dataframe for a country with CO2
def create_country_df(country):
    country_df = data_CO2[country].to_frame()
    country_df.rename(columns={country: 'CO2_emissions'}, inplace=True)
    country_df.index = country_df.index.astype(int)
    return country_df

# Function to get data of specific country
def filter_by_country(df, country):
    filtered_df = df[df['Entity'] == country]
    return filtered_df

def variable_csv_to_dataframe(variable_csv, startyear, endyear):
    variable_df = pd.read_csv(variable_csv)
    variable_df = variable_df[(variable_df['Year'] >= startyear) & (variable_df['Year'] <= endyear)]
    return variable_df

def filter_data(variable_df, country_df, country, column):
    country_variable_df = filter_by_country(variable_df, country)
    country_variable_df = country_variable_df.set_index('Year')
    country_variable_df = country_variable_df[column].to_frame()
    country_variable_df.index = country_variable_df.index.astype(int)
    country_df = country_df.join(country_variable_df)
    country_df.name = country #This should add a name to the dataframe. Needed in plotting.
    return country_df

# Create an empty dataframe with an index for r-squared values of the different countries
r_df_country = pd.DataFrame(index=range(1,11)) ## NEW

# Draw a scatter plot with linear regression line & calculate R-squared for the model
def plot_country_variables_vs_CO2(country_df):  
    num_columns = 2
    num_rows = country_df.shape[1] // 2
    plt.figure(figsize=(15, 10))

    r_list_country = [] # Create a list to add the different R-squared figures of a country.

    for column_index in range(1,country_df.shape[1]):
        selected_column = country_df.iloc[:,column_index]

        #Create subplots
        plt.subplot(num_rows, num_columns, column_index)

        plt.scatter(x=selected_column, y=country_df["CO2_emissions"], c="orange")
        plt.title(f"CO2 vs {selected_column.name}, {country_df.name}")
        plt.xlabel(f"{selected_column.name}")
        plt.ylabel("MtCO2 emissions")
        
        model = LinearRegression()
        model.fit(country_df.iloc[:,column_index].values.reshape(-1,1), country_df.iloc[:,0].values.reshape(-1,1))
        y_fitted = model.predict(country_df.iloc[:,column_index].values.reshape(-1,1))
        r_squared = model.score(country_df.iloc[:,column_index].values.reshape(-1,1), country_df.iloc[:,0].values.reshape(-1,1))
        r_list_country.append("{:.2f}".format(r_squared))  # Add the r-squared value of the variable to the list

        plt.text(0.8,1.1,f"R squared: {r_squared:.2f}", fontsize=12, color="green", transform=plt.gca().transAxes)
        plt.plot(selected_column, y_fitted, color="green")

    r_df_country[f"{country_df.name}"] = r_list_country # Add the r-squared values of the country to the R-squared dataframe
    plt.tight_layout()
    plt.show()

# Creates dataframe of a country that includes all the uncommented variables
def country_df_with_data(country, variable_csv_list):
    country_df = create_country_df(country)

    for variable_csv in variable_csv_list:
        variable_df = variable_csv_to_dataframe(variable_csv, 2011, 2021)
        column = csv_column_dict.get(variable_csv)
        country_df = filter_data(variable_df, country_df, country, column)

    plot_country_variables_vs_CO2(country_df) # Here we create the plots as well.

    return country_df

finland_df = country_df_with_data('Finland', variable_csv_list)
estonia_df = country_df_with_data('Estonia', variable_csv_list)
sweden_df = country_df_with_data('Sweden', variable_csv_list)

print("\nFinland dataframe")
print(finland_df)
print("\nEstonia dataframe")
print(estonia_df)
print("\nSweden dataframe")
print(sweden_df)
print("\nR-squared table of the different countries for the variables 1-10")
print(r_df_country)