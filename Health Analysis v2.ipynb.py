#!/usr/bin/env python
# coding: utf-8

# Import needed packages/libraries 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from babel.numbers import format_currency
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick
import plotly.express as px
import plotly.graph_objects as go


# Import csv file as a pandas dataframe

# In[ ]:


df = pd.read_csv('us_county_sociohealth_data.csv')

# find out the percentage of each column that's null or missing 

# In[ ]:


for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

# Drop columns that are unrelated to objective

# In[ ]:


df1 = df.drop(columns=['num_deaths','years_of_potential_life_lost_rate','num_mental_health_providers','mental_health_provider_rate','annual_average_violent_crimes','violent_crime_rate','num_deaths_3','child_mortality_rate','num_deaths_4','infant_mortality_rate','num_hiv_cases','hiv_prevalence_rate','num_drug_overdose_deaths','drug_overdose_mortality_rate','num_motor_vehicle_deaths','motor_vehicle_mortality_rate','percent_disconnected_youth','average_grade_performance','average_grade_performance_2','segregation_index','segregation_index_2','homicide_rate','num_deaths_5','suicide_rate_age_adjusted','num_firearm_fatalities','firearm_fatalities_rate','juvenile_arrest_rate'])

# Create new dataframe with related columns that have less than 5% missing values

# In[ ]:


avg= df1[['state','county','area_sqmi','total_population','population_density_per_sqmi','average_number_of_physically_unhealthy_days','average_number_of_mentally_unhealthy_days','food_environment_index','teen_birth_rate','high_school_graduation_rate','num_unemployed_CHR','income_ratio','life_expectancy','num_food_insecure','median_household_income','num_households_with_severe_cost_burden','num_below_poverty','per_capita_income','num_no_highschool_diploma','percent_fair_or_poor_health','percent_adults_with_obesity','percent_physically_inactive','percent_food_insecure','percent_limited_access_to_healthy_foods','percent_below_poverty','age_adjusted_death_rate']]

# Fill missing values with the average for each column

# In[ ]:


avg.fillna((avg.mean()), inplace=True)

# Drop off all records of counties whose food environment index is 6 or greater

# In[ ]:


avg = avg[avg['food_environment_index'] < 6]


# Find the correlation between the food environment index and other related columns 

# In[ ]:


avg.corr('pearson')['food_environment_index'].sort_values()

# Combine the two columns counting unhealthy days into one and drop the original two 

# In[ ]:


avg['total_num_of_unhealthy_days'] = avg['average_number_of_mentally_unhealthy_days'] + avg['average_number_of_physically_unhealthy_days']

avg = avg.drop('average_number_of_physically_unhealthy_days', axis =1)
avg = avg.drop('average_number_of_mentally_unhealthy_days', axis =1)

# Group each state by averaging each attribute together from the individual counties

# In[ ]:


pop = avg.groupby('state')['total_population'].sum()
count = avg.groupby('state')['county'].count()
life = avg.groupby('state')['life_expectancy'].mean()
fei = avg.groupby('state')['food_environment_index'].mean()
days = avg.groupby('state')['total_num_of_unhealthy_days'].mean()
death = avg.groupby('state')['age_adjusted_death_rate'].mean()
food = avg.groupby('state')['percent_limited_access_to_healthy_foods'].mean()
insecure = avg.groupby('state')['percent_food_insecure'].mean()
pov = avg.groupby('state')['percent_below_poverty'].mean()
health = avg.groupby('state')['percent_fair_or_poor_health'].mean()

# Concatenate each series together to make a dataframe 

# In[ ]:


total = pd.concat([pop,count,life,fei,days,death,food,insecure,pov,health], axis=1)

# Reset the index of new dataframe

# In[ ]:


total = total.reset_index(0)

# Round the age adjusted death rate column to the nearest whole number for cleaning purposes

# In[ ]:


total['age_adjusted_death_rate'] = total['age_adjusted_death_rate'].round(0)

total = total.round(1)

# Plot a bar chart showing the food environment index by each state

# In[ ]:


total = total.sort_values(by='food_environment_index')

cmap = plt.get_cmap('YlGnBu')
norm = plt.Normalize(total['food_environment_index'].min(), total['food_environment_index'].max())
colors = cmap(norm(total['food_environment_index']))

plt.figure(figsize=(10,5))
plt.bar(total['state'], total['food_environment_index'], color = colors)
plt.xlabel("State")
plt.xticks(rotation=85)
plt.ylabel("Food Environment Index")
plt.title("Food Environment Index by State")
plt.show()

# Plot a bar chart of each state's percentage of people with limited access to healthy food

# In[ ]:


total = total.sort_values(by='percent_limited_access_to_healthy_foods', ascending=False)

cmap = plt.get_cmap('plasma')
norm = plt.Normalize(total['percent_limited_access_to_healthy_foods'].min(), total['percent_limited_access_to_healthy_foods'].max())
colors1 = cmap(norm(total['percent_limited_access_to_healthy_foods']))

plt.figure(figsize=(10,5))
plt.bar(total['state'], total['percent_limited_access_to_healthy_foods'], color=colors1)
plt.xlabel("State")
plt.xticks(rotation=85)
plt.ylabel("Limited Access Percentage")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
plt.title("Percent with Limited Access to Healthy Food by State")
plt.show()

# Plot a bar chart of each state's percentage of citizens below the poverty line

# In[ ]:


total = total.sort_values(by='percent_below_poverty', ascending=False)

cmap = plt.get_cmap('viridis')
norm = plt.Normalize(total['percent_below_poverty'].min(), total['percent_below_poverty'].max())
colors1 = cmap(norm(total['percent_below_poverty']))

plt.figure(figsize=(10,5))
plt.bar(total['state'], total['percent_below_poverty'], color=colors1)
plt.xlabel("State")
plt.xticks(rotation=85)
plt.ylabel("Percent Below Poverty")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
plt.title("Percent Below Poverty by State")
plt.show()


# Plot a bar chart of each state's percentage of citizens having continually fair or poor health 

# In[ ]:


total = total.sort_values(by='percent_fair_or_poor_health', ascending=False)

cmap = plt.get_cmap('cividis')
norm = plt.Normalize(total['percent_fair_or_poor_health'].min(), total['percent_fair_or_poor_health'].max())
colors1 = cmap(norm(total['percent_fair_or_poor_health']))

plt.figure(figsize=(10,5))
plt.bar(total['state'], total['percent_fair_or_poor_health'], color=colors1)
plt.xlabel("State")
plt.xticks(rotation=85)
plt.ylabel("Percent Fair/Poor Health")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
plt.title("Percent Fair/Poor Health by State")
plt.show()

# Plot a bar chart of each state's percentage of citizens who are considered food insecure

# In[ ]:


total = total.sort_values(by='percent_food_insecure', ascending=False)

cmap = plt.get_cmap('inferno')
norm = plt.Normalize(total['percent_food_insecure'].min(), total['percent_food_insecure'].max())
colors1 = cmap(norm(total['percent_food_insecure']))

plt.figure(figsize=(10,5))
plt.bar(total['state'], total['percent_food_insecure'], color=colors1)
plt.xlabel('State')
plt.xticks(rotation=85)
plt.ylabel("Percent of Food Insecure")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
plt.title("Percent Food Insecure by State")
plt.show()

# Sort dataframe by state alphabetically 

# In[ ]:


total = total.sort_values(by='state')

# Select rows containing the states of each region represented 

# In[ ]:


south = total.iloc[np.r_[0:1, 3:4, 6:8, 11:14, 19:20, 22:23, 24:25, 26:28, 29:30]]
midwest = total.iloc[np.r_[9:11, 14:15, 16:17, 20:22, 25:26,]]
west = total.iloc[np.r_[1:3, 4:6, 8:9, 15:16, 17:19, 23:24, 28:29, 30:31]]


# Create a pivot table heatmap with each region's dataframe based on their food environment index

# In[ ]:


result = west.pivot(index='state', columns='food_environment_index', values='food_environment_index')
sns.heatmap(result, annot=True, fmt="g", cmap='RdYlGn')
plt.show()

result1 = south.pivot(index='state', columns='food_environment_index', values='food_environment_index')
sns.heatmap(result1, annot=True, fmt="g", cmap='RdYlGn')
plt.show()

result2 = midwest.pivot(index='state', columns='food_environment_index', values='food_environment_index')
sns.heatmap(result2, annot=True, fmt="g", cmap='RdYlGn')
plt.show()

# Seperate the original dataframe for each state below a 4.5 food environment index and concatenate them together

# In[ ]:


sd = avg[avg['state'] == 'South Dakota']
idaho = avg[avg['state'] == 'Idaho']
alaska = avg[avg['state'] == 'Alaska']
miss = avg[avg['state'] == 'Mississippi']
lst = [sd, idaho, alaska, miss] 
food5 = pd.concat(lst)

# Find the most correlated attributes to the food environment index in Mississippi

# In[ ]:


miss.corr('pearson')['food_environment_index'].sort_values()

# Find the most correlated attributes to the food environment index in South Dakota

# In[ ]:


sd.corr('pearson')['food_environment_index'].sort_values()

# Find the most correlated attributes to the food environment index in Alaska

# In[ ]:


alaska.corr('pearson')['food_environment_index'].sort_values()

# Find the most correlated attributes to the food environment index in Idaho

# In[ ]:


idaho.corr('pearson')['food_environment_index'].sort_values()

# Create subset of original dataframe wit those attributes that are closely correlated to the food environment index in those four states. 

# In[ ]:


avg1 = avg[['state','life_expectancy','food_environment_index','total_num_of_unhealthy_days','age_adjusted_death_rate','percent_limited_access_to_healthy_foods','percent_food_insecure','percent_below_poverty','percent_fair_or_poor_health']]

# Rename columns to more readable formats for plotting purposes and drop the old columns 

# In[ ]:


avg1['% limited access to healthy foods'] = avg1['percent_limited_access_to_healthy_foods']
avg1['total num of unhealthy days'] = avg1['total_num_of_unhealthy_days']
avg1['life expectancy'] = avg1['life_expectancy']
avg1['food environment index'] = avg1['food_environment_index']
avg1['age adjusted death rate'] = avg1['age_adjusted_death_rate']
avg1['% food insecure'] = avg1['percent_food_insecure']
avg1['% below poverty'] = avg1['percent_below_poverty']
avg1['% fair/poor health'] = avg1['percent_fair_or_poor_health']

avg1 = avg1.drop(columns= ['life_expectancy','food_environment_index','total_num_of_unhealthy_days','age_adjusted_death_rate','percent_limited_access_to_healthy_foods','percent_food_insecure','percent_below_poverty','percent_fair_or_poor_health'])


# Create new subset dataframes for each specific state

# In[ ]:


sd1 = avg1[avg1['state'] == 'South Dakota']
idaho1 = avg1[avg1['state'] == 'Idaho']
alaska1 = avg1[avg1['state'] == 'Alaska']
miss1 = avg1[avg1['state'] == 'Mississippi']

# Create food environment index correlation heatmap for South Dakota

# In[ ]:


f = plt.figure(figsize=(10, 10))
plt.matshow(sd1.corr(), fignum=f.number, cmap = 'RdYlGn')
plt.xticks(range(sd1.select_dtypes(['float','int']).shape[1]), sd1.select_dtypes(['float','int']).columns, fontsize=14, rotation=75, ha="left")
plt.yticks(range(sd1.select_dtypes(['float','int']).shape[1]), sd1.select_dtypes(['float','int']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=16)
plt.title('South Dakota Correlation', fontsize=16);

# Create food environment index correlation heatmap for Mississippi

# In[ ]:


t = plt.figure(figsize=(10, 10))
plt.matshow(miss1.corr(), fignum=t.number, cmap = 'RdYlGn')
plt.xticks(range(miss1.select_dtypes(['float','int']).shape[1]), miss1.select_dtypes(['float','int']).columns, fontsize=14, rotation=75, ha="left")
plt.yticks(range(miss1.select_dtypes(['float','int']).shape[1]), miss1.select_dtypes(['float','int']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.title('Mississippi Correlation', fontsize=16);

# Create food environment index correlation heatmap for Alaska

# In[ ]:


j = plt.figure(figsize=(10, 10))
plt.matshow(alaska1.corr(), fignum=j.number, cmap='RdYlGn')
plt.xticks(range(alaska1.select_dtypes(['float','int']).shape[1]), alaska1.select_dtypes(['float','int']).columns, fontsize=14, rotation=75, ha="left")
plt.yticks(range(alaska1.select_dtypes(['float','int']).shape[1]), alaska1.select_dtypes(['float','int']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.title('Alaska Correlation', fontsize=16);

# Create food environment index correlation heatmap for Idaho

# In[ ]:


g = plt.figure(figsize=(10, 10))
plt.matshow(idaho1.corr(), fignum=g.number, cmap = 'RdYlGn')
plt.xticks(range(idaho1.select_dtypes(['float','int']).shape[1]), idaho1.select_dtypes(['float','int']).columns, fontsize=14, rotation=75, ha="left")
plt.yticks(range(idaho1.select_dtypes(['float','int']).shape[1]), idaho1.select_dtypes(['float','int']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.title('Idaho Correlation', fontsize=16);

# There's a multitude of socioeconomic factors determining a state's food environment index. The above highlighted some of the most important components.
