# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import plotly.offline as pyo
import plotly.graph_objects as go
pyo.init_notebook_mode()
import matplotlib.pyplot as plt
import folium
import pandas as pd
import plotly.express as px

# %% [markdown]
# # Exploratory Data Anaylsis

# %%
cases = pd.read_csv("covid_19_india.csv")
cases.head(10)

# It contains latitude and longitude coordinates of indian states.
lat_long = pd.read_csv("Latitude_and_Longitude_State.csv")
lat_long.head()

# It can provide us the ratio or rate or percentage of positive cases per population statewise.
popul = pd.read_csv("population_india_census2011.csv")
popul
population = popul.copy()
# Sno	State / Union Territory	Population	Rural population	Urban population	


# %%
lat_long


# %%
cols_po = popul['State / Union Territory'].unique()


# %%
cols_lat = lat_long['State'].unique()


# %%
def compare_replace_col(cols_po,cols_lat):
    i = 0 
    for col_p, col_l in zip(cols_po, cols_lat):
        if col_p != col_l:
            cols_lat[i] = col_p
    return cols_lat
cols_lat = compare_replace_col(cols_po,cols_lat)


# %%
lat_long['State'] = cols_lat


# %%
lat_long


# %%
cases.info()


# %%
cases['State/UnionTerritory'].unique()


# %%
cases['State/UnionTerritory'].replace({"Telengana" : "Telangana", "Telengana***" : "Telangana",
                                        "Telangana***" : "Telangana"}, inplace = True)

cases['State/UnionTerritory'].replace({"Daman & Diu" : "Dadra and Nagar Haveli and Daman and Diu",
                                          "Dadar Nagar Haveli" : "Dadra and Nagar Haveli and Daman and Diu"},
                                         inplace = True)
cases = cases[(cases['State/UnionTerritory'] != 'Unassigned') &
                    (cases['State/UnionTerritory'] != 'Cases being reassigned to states')]
cases['State/UnionTerritory'].unique()


# %%
popul.rename(columns={'State / Union Territory':'State/UnionTerritory'}, inplace=True)
popul['State/UnionTerritory'].replace({"Telengana" : "Telangana"})


# %%
cases.Date = pd.to_datetime(cases.Date, dayfirst=True)

cases.drop(['Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational'], axis = 1, inplace=True)
cases.head()


# %%
print("Starting date : ", min(cases.Date.values))
print("Ending date : ", max(cases.Date.values))


# %%
daily_cases = cases.groupby('Date').sum().reset_index()
daily_cases['Active'] = 1

for val in daily_cases.index:
    if val != 0:
        daily_cases['Active'].loc[val] = daily_cases['Confirmed'].loc[val] - daily_cases['Cured'].loc[val-1] - daily_cases['Deaths'].loc[val-1]
    
daily_cases


# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x = daily_cases.Date, y = daily_cases.Confirmed, name = 'Confirmed'))
fig.add_trace(go.Scatter(x = daily_cases.Date, y = daily_cases.Cured, name = 'Cured'))
fig.add_trace(go.Scatter(x = daily_cases.Date, y = daily_cases.Deaths, name = 'Deaths'))
fig.add_trace(go.Scatter(x = daily_cases.Date, y = daily_cases.Active, name = 'Active Cases'))

fig.update_layout(title = 'CORONA VIRUS CASES IN INDIA', yaxis_title = 'Cases Count (in lakhs)')

fig.show()


# %%
country_cases = cases[cases.Date == max(cases.Date)]
print(country_cases.shape)
country_cases.head()


# %%
lat_long.rename(columns={"State":"State/UnionTerritory"}, inplace=True)
lat_long = lat_long[['State/UnionTerritory', 'Latitude', 'Longitude']]
country_cases = pd.merge(country_cases, lat_long, on='State/UnionTerritory')
country_cases.head()


# %%
m = folium.Map(location=[28,77], zoom_start=4)
country_cases['Confirmed'] = country_cases['Confirmed'].astype(float)

# I can add marker one by one on the map
for i in range(0,len(country_cases)):
    folium.Circle(location = [country_cases.iloc[i]['Latitude'], country_cases.iloc[i]['Longitude']],
                popup = [country_cases.iloc[i]['State/UnionTerritory'],country_cases.iloc[i]['Confirmed']],
                radius = country_cases.iloc[i]['Confirmed']/2,
                color = 'crimson', fill = True, fill_color='crimson').add_to(m)
    
m


# %%
total_pop = popul['Population'].sum()
print("The total population of India is : ", total_pop)

popul = cases.merge(popul[['State/UnionTerritory', 'Population']])
popul['ConfirmPerc'] = 0
popul['ConfirmPerc'] = (popul['Confirmed']/popul['Population'])*100


# %%
fig = go.Figure()
for st in popul['State/UnionTerritory'].unique():
    df = popul[popul['State/UnionTerritory'] == st]
    fig.add_trace(go.Scatter(x = df.Date, y = df.ConfirmPerc, name = st))
    
fig.update_layout(title = 'Positive Cases Percentage Per Population', yaxis_title = 'Percentage (%)')
fig.show()


# %%
# Here, we're grouping the data by date bcoz we want to visualize the data on per basis
popul_dates = popul.drop('ConfirmPerc', axis=1).groupby('Date').sum()

# In this, population should be same, as we're talking about the positive percentage per population
popul_dates['Population'] = total_pop

# Calculating total percentage of positive cases
popul_dates['TotConfirmPerc'] = (popul_dates['Confirmed']/popul_dates['Population'])*100
popul_dates


# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x = popul_dates.index, y = popul_dates.TotConfirmPerc))
fig.update_layout(title = 'Percentage of positive cases across India', yaxis_title = 'Percentage (%)')
fig.show()


# %%
daily_cases = cases.groupby('Date').sum().reset_index()
daily_cases['Active'] = 1

for val in daily_cases.index:
    if val != 0:
        daily_cases['Active'].loc[val] = daily_cases['Confirmed'].loc[val] - daily_cases['Cured'].loc[val-1] - daily_cases['Deaths'].loc[val-1]
    
daily_cases


# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x = daily_cases.Date, y = daily_cases.Active, name = 'Active Cases'))

fig.update_layout(title = 'Daily Active Cases', xaxis_title = 'Time', yaxis_title = 'Count (in lakhs)')
fig.show()


# %%
state_daily_cases = cases.sort_values(by=['State/UnionTerritory', 'Date']).reset_index(drop=True)
state_daily_cases['ActiveCases'] = 0

for st in sorted(cases['State/UnionTerritory'].unique()):
    df = state_daily_cases[state_daily_cases['State/UnionTerritory'] == st]
    for i in df.index:
        conf = state_daily_cases['Confirmed'].iloc[i]
        rec = state_daily_cases['Cured'].iloc[i-1]
        death = state_daily_cases['Deaths'].iloc[i-1]
            
        state_daily_cases['ActiveCases'].iloc[i] = conf - rec - death
    state_daily_cases['ActiveCases'].iloc[df.index[0]] = state_daily_cases['Confirmed'].iloc[df.index[0]]


# %%
fig = go.Figure()
for st in state_daily_cases['State/UnionTerritory'].unique():
    df = state_daily_cases[state_daily_cases['State/UnionTerritory'] == st]
    fig.add_trace(go.Scatter(x = df['Date'], y = df['ActiveCases'], name = st))

fig.update_layout(title = 'Daily Active Cases', xaxis_title = 'Time', yaxis_title = 'Count (in lakhs)')
fig.show()


# %%
beds = pd.read_csv("hospital_beds.csv")


# %%
total = beds.iloc[35]
print("Total beds available all over India", total)
beds.drop([35], inplace=True)
beds.head()


# %%
beds_per_state = state_daily_cases.set_index('State/UnionTerritory').join(beds.set_index('States'))
beds_per_state['AvailableBeds'] = beds_per_state['Total hospital beds'] - beds_per_state['ActiveCases']
beds_per_state


# %%
beds_df = beds_per_state[['Date', 'AvailableBeds']]
beds_df = pd.pivot_table(beds_per_state, values = 'AvailableBeds', index = 'Date',
                               columns = beds_per_state.index)
for st in beds_df.columns:
    val = beds[beds['States'] == st]['Total hospital beds']
    beds_df[st].fillna(int(val), inplace=True)
    
beds_df.head()


# %%
fig = go.Figure()
for col in beds_df.columns:
    fig.add_trace(go.Scatter(x = beds_df.index, y = beds_df[col], name = col))

fig.update_layout(title = 'Number of available beds statewise', yaxis_title = 'Number of available beds')
fig.show()


# %%
age_group = pd.read_csv("AgeGroupDetails.csv")


# %%
age_group


# %%

fig = px.bar(age_group, x="AgeGroup", y="TotalCases", orientation='v')
fig.show()


# %%
popul_density = population


# %%
popul_density["Density"] = popul_density["Density"].str.replace('/km2','')


# %%
popul_density["Density"] = popul_density["Density"].str.replace(',','') 


# %%
popul_density["Density"]=popul_density["Density"].str.replace(r"\(.*\)","")


# %%
popul_density["Density"] = popul_density["Density"].astype('float')


# %%
popul_density = popul_density[['State / Union Territory','Density']]


# %%
country_cases.sort_values('State/UnionTerritory',inplace=True)


# %%
popul_density.sort_values('State / Union Territory' , inplace= True)


# %%
popul_density['State/UnionTerritory'] = popul_density['State / Union Territory']


# %%
drop_col = ['Andaman and Nicobar Islands', 'Dadra and Nagar Haveli and Daman and Diu' ,'Lakshadweep']
popul_density.drop('State / Union Territory' , axis =1,inplace=True)


# %%
popul_density.set_index('State/UnionTerritory',inplace=True)


# %%
popul_density.drop(drop_col,inplace=True)


# %%
popul_density.reset_index(inplace=True)


# %%
country_cases.sort_values('State/UnionTerritory',inplace=True)


# %%
country_cases.reset_index(inplace=True)


# %%
country_cases['Density'] = popul_density['Density']


# %%
country_cases['Confirmed'].corr(country_cases['Density'])

# %% [markdown]
# Conclusion : I thought there will be some positive correlation between confirmed cases and the density, but it is not the case !!!! 

# %%
px.bar(beds,'States','Total hospital beds')


# %%
testing = pd.read_csv("StatewiseTestingDetails.csv")


# %%
testing


# %%
testing_states = testing.groupby('State')

# %% [markdown]
# # Prediction using ARIMA

# %%
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.simplefilter('ignore')


# %%
sns.set(palette = 'Set1',style='darkgrid')
#Function for making a time serie on a designated country and plotting the rolled mean and standard 
def roll(country,case='ConfirmedCases'):
    ts=df.loc[(df['Country_Region']==country)]  
    ts=ts[['Date',case]]
    ts=ts.set_index('Date')
    ts.astype('int64')
    a=len(ts.loc[(ts['ConfirmedCases']>=10)])
    ts=ts[-a:]
    return (ts.rolling(window=4,center=False).mean().dropna())


def rollPlot(country, case='ConfirmedCases'):
    ts=df.loc[(df['Country_Region']==country)]  
    ts=ts[['Date',case]]
    ts=ts.set_index('Date')
    ts.astype('int64')
    a=len(ts.loc[(ts['ConfirmedCases']>=10)])
    ts=ts[-a:]
    plt.figure(figsize=(16,6))
    plt.plot(ts.rolling(window=7,center=False).mean().dropna(),label='Rolling Mean')
    plt.plot(ts[case])
    plt.plot(ts.rolling(window=7,center=False).std(),label='Rolling std')
    plt.legend()
    plt.title('Cases distribution in %s with rolling mean and standard' %country)
    plt.xticks([])


