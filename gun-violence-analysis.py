# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from numpy import nan as NaN
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import datetime
import calendar
import gmaps
import gmaps.datasets
import plotly.express as pltex
from wordcloud import WordCloud
import openpyxl
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm


# %%
# Importing a file
data = pd.read_csv("./DATA/data.csv");

# Droping unwanted columns
data.drop(columns=['incident_url','source_url','incident_url_fields_missing','gun_stolen','gun_type','incident_characteristics', 'notes', 'participant_age_group', 'participant_gender', 'participant_name', 'participant_relationship', 'participant_status', 'participant_type', 'sources', 'state_house_district', 'state_senate_district'], inplace=True)


# %%
# Time related trends (years, months)
    # total incidents per year number
sub_data = data[["date", "n_killed", "n_injured"]]
sub_data["date"] = sub_data["date"].astype('datetime64[ns]'); 
sub_data["date"] = pd.DatetimeIndex(sub_data['date']).year
sub_data = sub_data.groupby(by=["date"]).sum()
total_incidents_per_year = sub_data

    # average incidents per month
sub_data = data[["date", "n_killed", "n_injured"]]
sub_data["date"] = sub_data["date"].astype('datetime64[ns]');
sub_data["date"] = pd.DatetimeIndex(sub_data['date']).month
sub_data = sub_data.groupby(by=["date"]).sum()
sub_data = sub_data.div(len(total_incidents_per_year.index))
average_incidents_per_month = sub_data

months = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec' 
]

months = pd.DataFrame(data=months, columns=["month"]);
months.index = np.arange(1, len(months) + 1)
average_incidents_per_month = average_incidents_per_month.join(months)
average_incidents_per_month.set_index("month", inplace=True);

    # average incidents per day of week
def weekday_count(start, end):
  start_date  = datetime.datetime.strptime(start, '%d/%m/%Y')
  end_date    = datetime.datetime.strptime(end, '%d/%m/%Y')
  week        = {}
  for i in range((end_date - start_date).days):
    day       = calendar.day_name[(start_date + datetime.timedelta(days=i+1)).weekday()]
    week[day] = week[day] + 1 if day in week else 1
  return week

def changeDateFormat(date):
    new_date = date[8::] + '/' + date[5:7] + '/' + date[0:4]
    return new_date

t_week = weekday_count(changeDateFormat(data["date"].loc[0]), changeDateFormat(data["date"].loc[len(data.index)-1]))

sub_data = data[["date", "n_killed", "n_injured"]]
sub_data["date"] = sub_data["date"].astype('datetime64[ns]');
sub_data["date"] = pd.DatetimeIndex(sub_data['date']).weekday
sub_data = sub_data.groupby(by=["date"]).sum()
sub_data.loc[0] = sub_data.loc[0].div(t_week['Monday'])
sub_data.loc[1] = sub_data.loc[1].div(t_week['Tuesday'])
sub_data.loc[2] = sub_data.loc[2].div(t_week['Wednesday'])
sub_data.loc[3] = sub_data.loc[3].div(t_week['Thursday'])
sub_data.loc[4] = sub_data.loc[4].div(t_week['Friday'])
sub_data.loc[5] = sub_data.loc[5].div(t_week['Saturday'])
sub_data.loc[6] = sub_data.loc[6].div(t_week['Sunday'])

days = [
  'Mon',
  'Tue',
  'Wed',
  'Thu',
  'Fri',
  'Sat',
  'Sun'
]

days = pd.DataFrame(data=days, columns=["day"])
average_incidents_per_day = sub_data
average_incidents_per_day = average_incidents_per_day.join(days)
average_incidents_per_day.set_index("day", inplace=True)


# %%
# Time related trends plots
    # total incidents per year
total_incidents_per_year.plot(kind="bar")
    # average incidents per month
average_incidents_per_month.plot(kind="bar")
    # average incidents per day
average_incidents_per_day.plot(kind="bar")


# %%
# Location Related Trends
# Visualizing amount of incidents on map
gmaps.configure(api_key='')
inj_map_df = data[['latitude','longitude','n_injured','n_killed']]
inj_map_df = inj_map_df.dropna()
locations = inj_map_df[['latitude', 'longitude']]
weights = inj_map_df['n_injured'] + inj_map_df['n_killed']
fig = gmaps.figure(map_type = 'HYBRID')
fig.add_layer(gmaps.heatmap_layer(locations, weights=weights))
fig


# %%
# Addition of data about registered guns amount according to state
guns_data = pd.read_excel('./DATA/guns.xlsx', sheet_name="Data", engine="openpyxl")
guns_data.drop([0,1,2,3], axis=0, inplace=True)
guns_data.drop(columns=["Unnamed: 0", "Unnamed: 3"], inplace=True)
guns_data = guns_data.rename(columns={"Unnamed: 1": "state", "Unnamed: 2": "registered_guns"})
guns_data.reset_index(inplace=True)
guns_data.drop(columns=["index"], inplace=True)
guns_data.drop(len(guns_data)-1, inplace=True)


# %%
# Visualizing amount of deaths according to given state
states = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

data['state_code'] = data['state'].apply(lambda x : states[x])
fig = pltex.choropleth(data.groupby('state_code')['n_killed'].sum().reset_index(), locations='state_code', locationmode="USA-states", color='n_killed', scope="usa", color_continuous_scale="Plasma")
fig.show()


# %%
guns_data['state_code'] = guns_data["state"].apply(lambda x : states[x])
fig = pltex.choropleth(guns_data["registered_guns"], locations=guns_data["state_code"], locationmode="USA-states", color='registered_guns', scope="usa", color_continuous_scale="Plasma")
fig.show()


# %%
# Location Word Cloud
text = ''

location_text = data['location_description'].dropna()

for index, value in location_text.items():
    text = text + ' ' + value

wc = WordCloud(background_color="white", max_words=500, width=1000, height=700)

# generate word cloud
wc.generate(text)
plt.figure( figsize=(20,10) )
plt.axis("off")
plt.imshow(wc, interpolation="bilinear")
plt.show()


# %%
# # Importing a file
# data = pd.read_csv("./DATA/data.csv");

# # Droping unwanted columns
# data.drop(columns=['incident_url','source_url','incident_url_fields_missing','gun_stolen','gun_type','incident_characteristics', 'notes', 'participant_age_group', 'participant_gender', 'participant_name', 'participant_relationship', 'participant_status', 'participant_type', 'sources', 'state_house_district', 'state_senate_district'], inplace=True)
# sub_data = data.drop(columns=["incident_id", "city_or_county", "address", "congressional_district", "latitude", "longitude", "location_description", "participant_age", "date", "state"])
# sub_data.dropna(inplace=True)

# sub_data = data[["state", "n_killed", "n_injured"]]
# sub_data = sub_data.groupby(by=["state"]).sum()
# guns_data.set_index("state", inplace=True)
# guns_data = guns_data.drop(columns=["state_code"])
# sub_data = sub_data.join(guns_data)


# %%
# PCA

# components_number = 2;
# pca = PCA(components_number)
# pca = pca.fit_transform(sub_data)
# pca=pd.DataFrame(data = pca, columns=["component_1", "component_2"])
# x = pca["component_1"]
# y = pca["component_2"]
# fig = plt.figure(figsize=(8,8));
# ax = fig.add_subplot(1,1,1);
# ax.scatter(x,y)
# ax.grid()
# ax.set_xlabel("Component 1", fontsize=15)
# ax.set_ylabel("Component 2", fontsize=15)
# ax.set_title("2 component PCA - all", fontsize=20)

# states = sub_data.index
# for i in range(len(states)):
#     ax.annotate(str(states[i]), (x[i],y[i]))


# %%
# KNN
# knn_df = data[['n_killed', 'n_injured','latitude', 'longitude', 'n_guns_involved']]
# knn_columns = knn_df.columns
# knn_index = knn_df.index

# imputer = KNNImputer(n_neighbors = 2)
# knn_df = imputer.fit_transform(knn_df)
# knn_df = pd.DataFrame(knn_df, columns=knn_columns, index= knn_index)
# df.update(knn_df)


# %%
data = pd.read_excel('./DATA/knn_data.xlsx', sheet_name="Sheet1", engine="openpyxl")


# %%
data['number_of_participants'] = 0
data['total_age_of_participants'] = 0

def age_group(row):
    age_col = row['participant_age']
    if str(age_col) == 'nan':
        return row
    ages = age_col.split("||")
    if len(ages) == 1:
        ages = ages[0].split("|")
    for age in ages:
        try:
            age_value = int(age.split('::')[1])
        except:
            age_value = int(age.split(':')[1])

        row['number_of_participants'] += 1
        row['total_age_of_participants'] += age_value
    return row

data = data.apply(age_group, axis = 1)
data['mean_age_of_participants'] = data['total_age_of_participants'].div(data['number_of_participants'])
data = data.drop(columns=["total_age_of_participants", "Unnamed: 0"])


# %%
# Change Data According To Given State
data_states = data[["state", "n_killed", "n_injured", "n_guns_involved", "number_of_participants"]]
data_states = data_states.set_index("state")
data_states = data_states.groupby(by=["state"]).sum()

data_mean_age = data[["state","mean_age_of_participants"]]
data_mean_age = data_mean_age.set_index("state")
data_mean_age = data_mean_age.groupby(by=["state"]).mean()

data_states = data_states.join(data_mean_age)


# %%
# Outliers detection
sub_data = data_states
missing_data = sub_data.isnull().sum()
missing_data.name = 'missing_data'

max_data = sub_data.max()
max_data.name = 'max'

min_data = sub_data.min()
min_data.name = 'min'

mean_data = sub_data.mean(skipna=True,numeric_only=True)
mean_data.name = 'mean'

median_data = sub_data.median()
median_data.name ='median'

std_deviation_data = sub_data.std()
std_deviation_data.name = 'std'


df_description = pd.DataFrame()
df_description = df_description.append([max_data, min_data, mean_data, median_data, std_deviation_data])
std_deviation = df_description.loc['std']
mean_values = df_description.loc['mean']

sig = mean_values + (std_deviation * 2)

mask = (sub_data > sig)
outliers = sub_data[mask]
outliers = outliers.dropna(how='all')


# %%
# Add victims column
data_states["victims"] = data_states["n_killed"] + data_states["n_injured"]

# Create 2D data sets
sub_data_victims_guns = data_states[["victims", "n_guns_involved"]]
sub_data_victims_age = data_states[["victims", "mean_age_of_participants"]]
sub_data_victims_part = data_states[["victims", "number_of_participants"]]
sub_data_age_part = data_states[["mean_age_of_participants","number_of_participants"]]


# %%
# Clustering
clustering_victims_guns = AgglomerativeClustering(distance_threshold=None, n_clusters=3).fit(sub_data_victims_guns)
clusters1 = pd.DataFrame(data=clustering_victims_guns.labels_, columns=["cluster"])
clusters1.set_index(sub_data_victims_guns.index, inplace=True)
sub_data_victims_guns = sub_data_victims_guns.join(pd.DataFrame(clusters1))

clustering_victims_age = AgglomerativeClustering(distance_threshold=None, n_clusters=3).fit(sub_data_victims_age)
clusters2 = pd.DataFrame(data=clustering_victims_age.labels_, columns=["cluster"])
clusters2.set_index(sub_data_victims_age.index, inplace=True)
sub_data_victims_age = sub_data_victims_age.join(pd.DataFrame(clusters2))

clustering_victims_part = AgglomerativeClustering(distance_threshold=None, n_clusters=5).fit(sub_data_victims_part)
clusters3 = pd.DataFrame(data=clustering_victims_part.labels_, columns=["cluster"])
clusters3.set_index(sub_data_victims_part.index, inplace=True)
sub_data_victims_part = sub_data_victims_part.join(pd.DataFrame(clusters3))

clustering_age_part = AgglomerativeClustering(distance_threshold=None, n_clusters=5).fit(sub_data_age_part)
clusters4 = pd.DataFrame(data=clustering_age_part.labels_, columns=["cluster"])
clusters4.set_index(sub_data_age_part.index, inplace=True)
sub_data_age_part = sub_data_age_part.join(pd.DataFrame(clusters4))


# %%
# Clustering 1
states = data_states.index
x = sub_data_victims_guns["victims"]
y = sub_data_victims_guns["n_guns_involved"]
z = clusters1.reset_index(0)
z = z["cluster"]
fig = plt.figure(figsize=(16,16));
ax = fig.add_subplot(1,1,1);
colormap = np.array(['r', 'g', 'b', 'c', 'y', 'k'])

ax.scatter(x = x, y = y, c=colormap[z])

ax.grid()
ax.set_xlabel("Victims ( killed + injured )", fontsize=15)
ax.set_ylabel("Inolved Guns", fontsize=15)
ax.set_title("Clustering 1", fontsize=20)

for i in range(len(states)):
    ax.annotate(str(states[i]), (x[i],y[i]))


# %%
# Clustering 2
x = sub_data_victims_age["victims"]
y = sub_data_victims_age["mean_age_of_participants"]
z = clusters2.reset_index(0)
z = z["cluster"]
fig = plt.figure(figsize=(16,16));
ax = fig.add_subplot(1,1,1);
colormap = np.array(['r', 'g', 'b', 'c', 'y', 'k'])

ax.scatter(x = x, y = y, c=colormap[z])

ax.grid()
ax.set_xlabel("Victims ( killed + injured )", fontsize=15)
ax.set_ylabel("Mean Age Of Participants", fontsize=15)
ax.set_title("Clustering 2", fontsize=20)

for i in range(len(states)):
    ax.annotate(str(states[i]), (x[i],y[i]))


# %%
# Clustering 3
x = sub_data_victims_part["victims"]
y = sub_data_victims_part["number_of_participants"]
z = clusters3.reset_index(0)
z = z["cluster"]
fig = plt.figure(figsize=(16,16));
ax = fig.add_subplot(1,1,1);
colormap = np.array(['r', 'g', 'b', 'c', 'y', 'k'])

ax.scatter(x = x, y = y, c=colormap[z])

ax.grid()
ax.set_xlabel("Victims ( killed + injured )", fontsize=15)
ax.set_ylabel("Number Of Participants", fontsize=15)
ax.set_title("Clustering 3", fontsize=20)

for i in range(len(states)):
    ax.annotate(str(states[i]), (x[i],y[i]))


# %%
# Clustering 4
x = sub_data_age_part["mean_age_of_participants"]
y = sub_data_age_part["number_of_participants"]
z = clusters4.reset_index(0)
z = z["cluster"]
fig = plt.figure(figsize=(16,16));
ax = fig.add_subplot(1,1,1);
colormap = np.array(['r', 'g', 'b', 'c', 'y', 'k'])

ax.scatter(x = x, y = y, c=colormap[z])

ax.grid()
ax.set_xlabel("Mean Age Of Participants", fontsize=15)
ax.set_ylabel("Number Of Participants", fontsize=15)
ax.set_title("Clustering 4", fontsize=20)

for i in range(len(states)):
    ax.annotate(str(states[i]), (x[i],y[i]))


# %%
# Classification 1
X_test_sing = sub_data_victims_guns.loc[['Texas']]
X_test_sing = X_test_sing.drop(["cluster"], axis=1)

sub_data_class_1 = sub_data_victims_guns.drop(["cluster"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(sub_data_class_1, clustering_victims_guns.labels_, test_size=0.3)

classifier = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test_sing)

equality = np.equal(y_pred, y_test)

classifier.score(X_test, y_test)
print('predicted: ' + str(y_pred) + '\n')
print(sub_data_victims_guns.loc[['Texas']])
print('\nscore: ' + str(classifier.score(X_test, y_test)))


# %%
# Classification 2
X_test_sing = sub_data_victims_age.loc[['Texas']]
X_test_sing = X_test_sing.drop(["cluster"], axis=1)

sub_data_class_2 = sub_data_victims_age.drop(["cluster"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(sub_data_class_2, clustering_victims_age.labels_, test_size=0.3)

classifier = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test_sing)

equality = np.equal(y_pred, y_test)

classifier.score(X_test, y_test)
print('predicted: ' + str(y_pred) + '\n')
print(sub_data_victims_age.loc[['Texas']])
print('\nscore: ' + str(classifier.score(X_test, y_test)))


# %%
# Classification 3
X_test_sing = sub_data_victims_part.loc[['Texas']]
X_test_sing = X_test_sing.drop(["cluster"], axis=1)

sub_data_class_3 = sub_data_victims_part.drop(["cluster"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(sub_data_class_3, clustering_victims_part.labels_, test_size=0.3)

classifier = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test_sing)

equality = np.equal(y_pred, y_test)

classifier.score(X_test, y_test)
print('predicted: ' + str(y_pred) + '\n')
print(sub_data_victims_part.loc[['Texas']])
print('\nscore: ' + str(classifier.score(X_test, y_test)))


# %%
# Classification 4
X_test_sing = sub_data_age_part.loc[['Texas']]
X_test_sing = X_test_sing.drop(["cluster"], axis=1)

sub_data_class_4 = sub_data_age_part.drop(["cluster"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(sub_data_class_4, clustering_age_part.labels_, test_size=0.3)

classifier = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test_sing)

equality = np.equal(y_pred, y_test)

classifier.score(X_test, y_test)
print('predicted: ' + str(y_pred) + '\n')
print(sub_data_age_part.loc[['Texas']])
print('\nscore: ' + str(classifier.score(X_test, y_test)))


# %%
# Classification Question:
# How does the number of participants, the mean age and number of guns has an impact on victims ?

sub_data = data_states[["number_of_participants", "mean_age_of_participants", "n_guns_involved"]]

amount = []
for state in data_states.index:
    data = data_states.loc[state]["victims"] 

    if(data >= 10000):
        amount.append(4)

    if(data < 10000 and data >= 6000):
        amount.append(3)

    if(data < 6000 and data >= 3000):
        amount.append(2)

    if(data < 3000 and data >= 1000):
        amount.append(1)

    if(data < 1000):
        amount.append(0)

# amount = pd.DataFrame(amount)
# amount = amount.set_index(states)
# print(type(amount))
# print("\n")
# print(type(clustering_age_part.labels_))
# print("\n")




# %%
# Classification 5
# X_test_sing = sub_data_age_part.loc[['Texas']]

sub_data_class_5 = sub_data
X_train, X_test, y_train, y_test = train_test_split(sub_data_class_5, amount, test_size=0.3)

classifier = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

equality = np.equal(y_pred, y_test)

classifier.score(X_test, y_test)
# print('predicted: ' + str(y_pred) + '\n')
print('\nscore: ' + str(classifier.score(X_test, y_test)))


# %%



