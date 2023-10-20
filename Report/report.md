# Mini-Project Report - Analyzing and Recognizing Leading Countries in CO2 Emission Reduction

#### Project Group: Helmi Karesti, Janne Penttala, Salla-Mari Uitto
#### GitHub Repository: [https://github.com/karhelmi/intro-ds-miniproject](https://github.com/karhelmi/intro-ds-miniproject)

## Motivation and Added Value

Our approach in analyzing countries that have reduced their CO2 emissions effectively highlights positive achievements and encourages the sharing of best practices in the fight against climate change. The thought behind all of this is that the world needs more positive reinforcement and hope.

The target groups of our project are political decision makers, environmental organizations and researchers. These are the groups that aim to find ways to combat climate change: our project aids them in their pursuit by giving examples of countries that have succeeded in reducing CO2 emissions. Our project also introduces measures that have led to the reduction of CO2 emissions. This will aid our target group in decision making.

## Data Collection

Data about CO2 emissions is widely available on the Internet. We compared the data on different sites to each other to make sure that the numbers didn't differ from each other. In the end we decided to use [Global Carbon Atlas](https://globalcarbonatlas.org/) for fetching CO2 emission data for every country. The fetched data is stored in a .csv file. This data is used to find the countries that have reduced their CO2 emissions the most. 

Another aspect in our project is to find what has affected the CO2 emission reduction in the selected countries. For this, we chose to use [Our World in Data](https://ourworldindata.org/). This choice was fairly easy, because we need a lot of different factors to analyze in our project and this single source can provide this for us. The choice on what to analyze was made by thorough evaluation as a group. For example, renewable energy was chosen as one factor because it is generally known to reduce CO2 emissions. This data is also fetched as a .csv file.

When our project progressed, we encountered some problems with the data we initially thought of using. At the beginning, we thought that we would analyze the change between the years 2011 and 2021. However, once we had done our code, we realized fast that not all countries have available data up to the year 2021 on our chosen factors. As a team, we discussed different possibilities for handling this: one simple option was to change the time frame to something else, like 2008-2018, and another simple option was to choose only factors and countries to analyze so that we have the needed data. Some thought was also put on filling the missing values by using linear regression or other means of predicting the values, but due to our tight schedule this was dismissed rather quickly: we wanted to keep our focus on analyzing the factors and not get too sidetracked. In the end, we decided to use variables and countries that have all the data between the years 2011 and 2021. So, now that we were aware of all of our limitations, we decided to include the following 10 variables: meat production, life expectancy, GDP per capita, percentage of internet users (in the population of the country), human development index (HDI), human rights index, population, nuclear energy, energy usage per capita and renewable energy.


## Preprocessing

Our project has two sub-tasks: first one is to find the countries that have reduced CO2 emissions and second is to analyze factors that have helped the country to reduce them. Without even actually looking at the data, we knew that we would have to merge data from several data sources or files together to be able to compare and analyze CO2 emission reduction and factors for each country.

Before starting to write the code, we looked at the csv data in Excel: this way we got a rough idea about what we were up against. For example, we noticed that in addition to each country, our CO2 emission data file contained CO2 emissions also for each continent.

### Code

From the beginning it was clear that we would work with Python. Another option would have been R, but overall Python was the more attractive choice. Our team did not have much experience with it, but we all wanted to learn more. With Python, it was obvious that we would be using Pandas and Numpy to help us process the data and Matplotlib for the visualization.

At first, we started with implementing code that gives us a list of countries and their CO2 emission reduction percentage. We read the csv file to a pandas dataframe and continued to process it in that format. We had to rename and re-arrange columns and rows, change data types and finally calculate the reduction in CO2 emissions. We calculated the reduction in three different units: in MtCO2, in kgCO2/GDP and in tCO2/person.

Next, we had to decide on how we want to analyze the factors that have led to a country reducing their CO2 emissions. This started to form out to be the trickiest part of our whole project and we spent a lot of time just discussing different possibilities. Finally we settled on creating a pandas dataframe for each country, and to that country specific data frame, we would add the CO2 emissions and all the factors as separate columns.

At first, we implemented code that did all of this for one country, Finland. Once that case worked, we started to refactor the code and make functions for separate tasks to make it easier to reuse the code. First, the data frame for a country is created with its CO2 emissions. Then, the frame is filled with all of our chosen variables. Again, we had to wrangle the data in different ways: we set indexes, changed data types and took data from another frame to our country-specific frame. After doing these steps, we have a dataframe for a country with all of the variables we will be analyzing and the CO2 emissions for each year. Based on these dataframes, we create linear regression models, calculate selected statistical measures and create various plots visualizing the data and results. Code is available in the GitHub repository (link available at the beginning of this report).

## Learning Task and Approach

To analyze factors that have possibly aided countries in reducing their CO2 emissions, we decided to use linear regression. Linear regression is a fairly simple model that can be used to analyze relationships between variables. In our project, we use (simple) linear regression to analyze the relationship between CO2 emissions and our chosen factors. 

We are also calculating R-Squared value for each of our simple linear regression. This statistical measure tells us how well the linear regression model fits the data: if the value is 1, the regression model fits the data (nearly) perfectly. In our project, this value is used to help analyze the impact of different factors to the CO2 emission reduction. In addition to R-Squared values we calculated the slope for each linear regression to see if the relationship is negative or positive.

Towards the end of the project, we also calculated the p-value for our linear regressions, which helps us to understand which factors are statistically significant. On top of these statistical values, we need domain knowledge to make correct conclusions. In this case this would be knowledge about climate change.

## Results and Visualizations

We organized all countries in order based on how much they have reduced their CO2 emissions. Figure 1 shows the countries with the biggest reductions and the countries which we analyzed in more detail are marked with dark green.

(figure to be added later)

The countries for further analysis are Estonia, Bosnia and Herzegovina, Greece, Serbia, Finland, Denmark, Malta, Sweden, Montenegro, Hong Kong, Luxembourg, Slovenia and Portugal. All of these countries have reduced their CO2 emissions more than 20% between the years 2011 and 2021. The countries with the biggest reduction, Aruba and Curaçao, were left out of our study due to them being small islands.

For each selected country, we created plots that visualize the linear regression for each of our factors. Figure 2 displays one example of these. Since we have ten factors that we analyzed for each country, each country had ten plots.

(figure to be added later)

In the final analysis we collected the R-squared values, slope values and p-values to excel sheet (available in GitHub) and used them to conclude our final results. The plots (Figure 2) had issues with the axis not starting from zero and thus giving an impression of a bigger change than there truly is.

Based on our analysis, we divided our factors into two groups: to those that had an impact on CO2 emission reductions and to those that need further analysis. In total, four factors have an effect, and these are share of renewables, energy usage per capita, HDI and GDP per capita.

Share of renewables has a negative relationship with CO2 emissions is all countries, so increasing the share of renewable energy reduces CO2 emissions. For most countries, the R-Squared values suggest that our model fits the data well and p-values indicate that the finding is statistically significant. Another clear relationship was detected between energy usage per capita and CO2 emission reduction. In this case, the relationship is positive and thus the more energy is consumed per capita, the more CO2 emissions increase. High R-Squared values and statistically significant p-values support this.

For GDP per capita, the relationship is negative in most countries and in several countries, based on p-value, this is statistically significant. It seems that economic growth is possible even while reducing CO2 emissions. For HDI, the relationship is negative for all countries, which means that as HDI increases, CO2 emissions reduce. However, we recognize that many aspects affect both GDP and HDI and thus these topics would be interesting to study in more detail.

Further analysis would be needed for the other six factors that we analyzed, which are meat production, life expectancy, percentage of internet users, human rights index, population and nuclear energy. For meat production, our results are statistically significant only for four countries, and even in those countries meat production for each year varies. However, based on other  research, meat production is a significant contributor to greenhouse gas emissions, including CO2 emissions [2]. Same goes for internet usage - other research suggests that it is a notable source of CO2 emissions due to its energy consumption [1], but our analysis did not find a significant relationship there.

Our analysis concludes for life expectancy that it could be a possible motivator for reducing CO2 emissions. However, a more interesting approach would be to compare countries with low life expectancy to countries with high life expectancy. For the human rights index, changes are quite small in the countries that we observed and this leads us to not drawing any conclusions about it, especially since our analysis suggests that as the human rights index goes lower, CO2 emissions are reducing.For population growth, it seems that while population grows, it is possible to also reduce CO2 emissions. 

The last factor that we categorized as a “needs further analysis”, is nuclear energy. Only three countries of the 13 we chose produce nuclear energy and our results are not statistically significant for those countries. To analyze this properly, we would need to focus on more countries that produce nuclear energy or find data on how the share of nuclear energy that the country uses and gets from other countries has changed over time.

### Conclusions

Based on our project work, we would suggest our target group to look into the countries that have reduced CO2 emissions. Our work indicates that using renewable energy and reducing energy usage reduces CO2 emissions efficiently. These both have been actions that countries could implement to reduce their CO2 emissions. Based on our study, reducing CO2 emissions is possible while the economy and HDI grows, so tackling climate change in this perspective does not mean giving up these.

## Final Thoughts and Future Steps

Overall, our project went well, and we did not have to change our initial idea. From the start, we worked on the project on a weekly basis and we did not have to hurry to finish this. As this was overall a great experience for us, we are quite sad that this is only a mini project. Within this topic, there is a lot to analyze and discover.

We did not use Jupyter Notebook in our project and towards the end we realized that it might have been a useful tool. With more time, our code could use some refactoring to make it easily reusable. Missing value handling could also be a palace for more development, as well as making it possible to handle different time periods.

In addition to the factors we have analyzed in this project, it would be interesting to analyze even more factors, like transport and agriculture. Some of our current factors also provide many possibilities for further analysis as we discussed previously. In addition, other greenhouse gasses could also be taken into account.

## References

[1]	Erol Gelenbe and Yves Caseau. 2015. The impact of information technology on energy consumption and carbon emissions. Ubiquity 2015, June, Article 1 (June 2015), 15 pages. doi: 10.1145/2755977

[2] Lynch J and Pierrehumbert R. 2019. Climate Impacts of Cultured Meat and Beef Cattle. Front. Sustain. Food Syst. 3:5. doi: 10.3389/fsufs.2019.00005

## Appendix 1

We calculated R-Squared values for each country and each factor. Table 1 (continued on the next page) displays these values.

|                         |     Estonia |   Bosnia and Herzegovina |     Greece |     Serbia |   Finland |   Denmark |    Malta |    Sweden |   Montenegro |   Hong Kong |   Luxembourg |   Slovenia |   Portugal |
|:------------------------|------------:|-------------------------:|-----------:|-----------:|----------:|----------:|---------:|----------:|-------------:|------------:|-------------:|-----------:|-----------:|
| Meat prod               | 0.013603    |                0.0185706 | 0.00045002 | 0.234585   | 0.57964   |  0.155629 | 0.740025 | 0.622154  |  0.0623682   |   0.0948399 |     0.656421 |  0.056336  | 0.00416474 |
| Life expectancy         | 0.198289    |                0.666987  | 0.0235934  | 0.205442   | 0.883166  |  0.922373 | 0.784595 | 0.622434  |  0.029899    |   0.364679  |     0.47634  |  0.141936  | 0.00898104 |
| GDP per capita          | 0.609242    |                0.305699  | 0.0390091  | 0.249408   | 0.280324  |  0.754281 | 0.733757 | 0.594905  |  0.000547547 |   0.0357641 |     0.120301 |  0.338849  | 0.0185354  |
| % of internet users     | 0.394728    |                0.182779  | 0.950651   | 0.200895   | 0.116718  |  0.676735 | 0.664362 | 0.0561736 |  0.0841578   |   0.331929  |     0.880982 |  0.608512  | 0.237508   |
| Human dev index         | 0.443168    |                0.181699  | 0.81814    | 0.0310892  | 0.79321   |  0.822894 | 0.688802 | 0.671089  |  9.52317e-05 |   0.312209  |     0.387029 |  0.454514  | 0.182147   |
| Human rights idx        | 5.18679e-08 |                0.118763  | 0.666948   | 0.128292   | 0.673891  |  0.681922 | 0.468933 | 0.699257  |  0.45185     |   0.787809  |     0.534427 |  0.551165  | 0.831457   |
| Population              | 0.450583    |                0.307154  | 0.91923    | 0.273189   | 0.853001  |  0.908947 | 0.567978 | 0.899717  |  5.90336e-05 |   0.2577    |     0.765279 |  0.522919  | 0.0769646  |
| Nuclear energy          | 0           |                0         | 0          | 0          | 0.0262912 |  0        | 0        | 0.306248  |  0           |   0         |     0        |  0.0346827 | 0          |
| Energy usage per capita | 0.906636    |                0.387338  | 0.626798   | 0.00261263 | 0.968722  |  0.912185 | 0.445411 | 0.663608  |  0.0258262   |   0.778499  |     0.976505 |  0.61269   | 0.376072   |
| Share of renewables     | 0.937885    |                0.318715  | 0.952203   | 0.774314   | 0.917012  |  0.973718 | 0.888893 | 0.569583  |  0.13081     |   0.433622  |     0.685072 |  0.530691  | 0.667267   |

Table 1. R-Squared values for each country and factor.

## Appendix 2

We calculated p-values for each linear regression. Table 2 (continues on the next page) presents these values for each country and factor.

|                         |     Estonia |   Bosnia and Herzegovina |      Greece |      Serbia |     Finland |     Denmark |       Malta |      Sweden |   Montenegro |   Hong Kong |   Luxembourg |   Slovenia |    Portugal |
|:------------------------|------------:|-------------------------:|------------:|------------:|------------:|------------:|------------:|------------:|-------------:|------------:|-------------:|-----------:|------------:|
| Meat prod               | 0.732723    |               0.689505   | 0.950636    | 0.131118    | 0.00648671  | 0.22989     | 0.000679663 | 0.00390883  |    0.458934  | 0.356882    |  0.00249674  | 0.482211   | 0.850477    |
| Life expectancy         | 0.169902    |               0.00215609 | 0.652055    | 0.161483    | 1.7327e-05  | 2.70463e-06 | 0.000284956 | 0.00389515  |    0.611153  | 0.0491296   |  0.0187424   | 0.253419   | 0.781656    |
| GDP per capita          | 0.00458323  |               0.0777156  | 0.560485    | 0.117811    | 0.0939486   | 0.000523393 | 0.000759092 | 0.00543865  |    0.945555  | 0.577588    |  0.29602     | 0.0602615  | 0.689786    |
| % of internet users     | 0.0384417   |               0.189652   | 3.47924e-07 | 0.166787    | 0.303812    | 0.00187577  | 0.00223703  | 0.482862    |    0.386841  | 0.0635907   |  1.88516e-05 | 0.00462394 | 0.128388    |
| Human dev index         | 0.0253571   |               0.191109   | 0.000130841 | 0.604033    | 0.00023612  | 0.000115873 | 0.0015699   | 0.00203432  |    0.977282  | 0.0739752   |  0.0409707   | 0.0229061  | 0.190503    |
| Human rights idx        | 0.99947     |               0.299337   | 0.00215727  | 0.279433    | 0.00195435  | 0.00173897  | 0.0200782   | 0.0013383   |    0.0234629 | 0.000265885 |  0.0105875   | 0.00887727 | 9.23288e-05 |
| Population              | 0.0237319   |               0.0768652  | 3.2381e-06  | 0.099029    | 4.93878e-05 | 5.57757e-06 | 0.00739275  | 8.64837e-06 |    0.982113  | 0.110923    |  0.000423519 | 0.0119124  | 0.408837    |
| Nuclear energy          | 0           |               0          | 0           | 0           | 0.633844    | 0           | 0           | 0.0773939   |    0         | 0           |  0           | 0.583504   | 0           |
| Energy usage per capita | 6.25004e-06 |               0.0408664  | 0.00368684  | 0.881358    | 4.43607e-08 | 4.73208e-06 | 0.024856    | 0.00226069  |    0.636902  | 0.000324077 |  1.22004e-08 | 0.00439459 | 0.0448127   |
| Share of renewables     | 9.85156e-07 |               0.0703962  | 3.0113e-07  | 0.000353312 | 3.66139e-06 | 2.02278e-08 | 1.37836e-05 | 0.0072623   |    0.274423  | 0.0275854   |  0.00165985  | 0.0110037  | 0.0021476   |

Table 2. P-values for each country and factor.

## Appendix 3

We calculated the slope of each linear regression. Table 3 (continues on the next page) presents these values for each country and factor.

|                         |        Estonia |   Bosnia and Herzegovina |          Greece |        Serbia |        Finland |        Denmark |         Malta |         Sweden |   Montenegro |      Hong Kong |     Luxembourg |      Slovenia |       Portugal |
|:------------------------|---------------:|-------------------------:|----------------:|--------------:|---------------:|---------------:|--------------:|---------------:|-------------:|---------------:|---------------:|--------------:|---------------:|
| Meat prod               |   -0.000113327 |              4.72141e-05 |    -1.47527e-05 |  -0.000115331 |   -0.000500131 |    2.41381e-05 |   0.000394431 |   -0.000127489 | -3.93073e-05 |   -0.000349698 |   -0.000457697 |  -3.47221e-05 |   -1.12875e-05 |
| Life expectancy         |   -2.25722     |              3.49444     |    -4.75075     |   2.90191     |   -9.86541     |   -8.56156     |  -0.481802    |   -7.10843     |  0.110093    |   -3.74065     |   -1.2586      |  -0.823283    |    1.07765     |
| GDP per capita          |   -0.000834537 |             -0.000954747 |     0.00261298  |  -0.00147484  |   -0.0022203   |   -0.00178712  |  -9.32286e-05 |   -0.00167758  | -3.22697e-06 |   -0.000347581 |   -0.000225743 |  -0.000246376 |   -0.000355872 |
| % of internet users     |   -0.452538    |             -0.0988321   |    -1.33977     |  -0.169993    |   -0.901873    |   -1.54805     |  -0.0532989   |    0.415391    | -0.00502619  |   -0.336464    |   -0.303578    |  -0.125348    |   -0.255395    |
| Human dev index         | -312.533       |            -70.3084      | -1343.44        | -74.4727      | -587.231       | -599.726       | -19.0558      | -259.177       | -0.244372    | -194.014       | -109.25        | -75.4525      | -165.143       |
| Human rights idx        |   -0.570588    |             43.7507      |   572.092       |  33.5991      |  906.371       |  446.378       |  13.848       |  713.819       | -6.97767     |   45.2245      |   66.0471      |  28.6942      |  271.412       |
| Population              |   -0.000426155 |              1.01119e-05 |     6.35036e-05 |   2.50802e-05 |   -0.000108327 |   -5.10366e-05 |  -1.03872e-05 |   -1.13031e-05 |  7.9654e-07  |   -2.12608e-05 |   -1.98098e-05 |  -4.13063e-05 |    1.20052e-05 |
| Nuclear energy          |    0           |              0           |     0           |   0           |   -2.33213     |    0           |   0           |    0.357113    |  0           |    0           |    0           |  -0.591957    |    0           |
| Energy usage per capita |    0.000656423 |             -0.00165779  |     0.00505767  |   0.000184129 |    0.0024004   |    0.00199924  |  -4.45592e-05 |    0.00136576  |  2.48236e-05 |    0.000815958 |    0.00010296  |   0.000575193 |    0.00160848  |
| Share of renewables     |   -0.267209    |             -0.28727     |    -1.53812     |  -1.32873     |   -0.891191    |   -0.348437    |  -0.0868438   |   -0.623488    | -0.0127849   |  -12.5359      |   -0.0266713   |  -0.220004    |   -0.453147    |

Table 3. Slope of each linear regression from country and factor.