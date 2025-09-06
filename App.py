import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.title("Olympics Data Analysis")

st.header("1️⃣ Loading Data")
st.markdown("We load the main Olympics dataset and take a quick overview of the data.")

olympdata = pd.read_excel("olympics-data.xlsx")

buffer = io.StringIO()
olympdata.info(buf=buffer)
s = buffer.getvalue()

st.subheader("Dataset Info")
st.text(s) 

st.subheader("Central Stats")
st.dataframe(olympdata.describe())

st.subheader("First 5 rows")
st.dataframe(olympdata.head())


st.header("2️⃣ Data Cleaning & Preprocessing")
st.markdown("""
- Convert birth and death dates to datetime.
- Drop NaN in 'NOC' column (only 1 row).
- Create 'birth_year' and 'death_year' columns for easier analysis.
- Drop redundant date columns.
""")

olympdata['born_date'] = pd.to_datetime(olympdata['born_date'])
olympdata['died_date'] = pd.to_datetime(olympdata['died_date'])

olympdata.dropna(subset=['NOC'], inplace=True)

olympdata['birth_year'] = olympdata['born_date'].dt.year
olympdata['death_year'] = olympdata['died_date'].dt.year

olympdata.drop(columns=['born_date', 'died_date'], inplace=True)

st.markdown("""
- Create an 'age' column to estimate athlete age in 2025.
- Age = -1 if death_year is given (athlete has passed away).
""")
current_year = 2025
olympdata['age'] = np.where(
    (olympdata['death_year'].notna()) | (olympdata['death_year'] == 0),
    -1,
    current_year - olympdata['birth_year']
)
st.dataframe(olympdata.head())

st.header("3️⃣ Merge Results Dataset")
st.markdown("""
- Drop 'team' column.
- Fill missing 'medal' values with 'No Medal'.
- Merge with main dataset on 'athlete_id'.
""")
results = pd.read_csv('Olymps_results.csv')
results = results.drop(columns=['team'])
results['medal'] = results['medal'].fillna('No Medal')

combined = pd.merge(olympdata, results, on='athlete_id', how='left')
combined.drop(columns=['as', 'born_city', 'born_region'], inplace=True)
combined[["year", "birth_year", "death_year"]] = combined[["year", "birth_year", "death_year"]].fillna(0).astype(int)
combined = combined[~combined['year'].isin([1940, 1944])]  # remove WWII years
combined.drop_duplicates(subset=['athlete_id', 'year', 'event'], inplace=True)
combined.rename(columns={'discipline':'sport'}, inplace=True)
combined.drop(columns=['noc'], inplace=True)
st.dataframe(combined.head())

st.header("4️⃣ Handling Height & Weight")
st.markdown("""
- Impute missing values using median height/weight grouped by sport and gender.
- For same athlete, keep maximum height/weight to avoid duplicates.
""")
combined['sex'] = combined['event'].apply(lambda x: '-' if pd.isna(x) else ('Men' if 'Men' in x else ('Women' if 'Women' in x else '-')))
combined['height_cm'] = combined.groupby(['sport','sex'])['height_cm'].transform(lambda x: x.fillna(x.median()))
combined['weight_kg'] = combined.groupby(['sport','sex'])['weight_kg'].transform(lambda x: x.fillna(x.median()))
combined['height_cm'] = combined.groupby('name')['height_cm'].transform('max')
combined['weight_kg'] = combined.groupby('name')['weight_kg'].transform('max')
st.dataframe(combined.head())

st.header("5️⃣ Merge Host Countries")
st.markdown("""
- Merge host country dataset based on year.
""")
host = pd.read_excel('Host_Countries.xlsx')
host.rename(columns={'Year':'year','Country':'host_country'}, inplace=True)
combined = pd.merge(combined, host, on='year', how='left')

st.subheader("Combined Dataset Preview")
st.dataframe(combined.head())

st.header("6️⃣ Quick Summary of Pre-Processed Data")
st.markdown("""
- Dataset cleaned and standardized for analysis.
- Missing values handled and redundant columns removed.
- Age, height, weight columns prepared for fair comparison.
- Ready for deeper analysis of athlete longevity, consistency, and performance trends.
""")



# Part 1: What makes a sport successful
st.header("Part 1: What Makes a Sport Successful?")

st.subheader("1. Which sport and event had the most winners?")
st.markdown("""
- We group medal-winning athletes and count unique participants per sport and event.
- Top 5 sports/events plotted below.
""")

def summarize_max(frame, name = 'None', val = 'None', description = 'Maximum'):
    if isinstance(frame, pd.Series):
        max_ind = frame.idxmax()
        max_val = frame[max_ind]
        print(f'{description}: {max_ind} with a value of {max_val} athletes')
    else:
        # For DataFrame 
        max_row = frame[val].idxmax()
        max_label = frame.loc[max_row, name]
        max_val = frame.loc[max_row, val]
        print(f'{description}: {max_label} with a value of {max_val} athletes')
    return max_ind, max_val
                
winners = combined[combined['medal'].isin(['Gold','Silver','Bronze'])]
winners_counts = winners.groupby(['sport', 'event'])['name'].nunique().sort_values(ascending = False)
max_event, max_winners = summarize_max(winners_counts, description = 'The Sport and The Event with The Most Winners')
st.write("The sport with the most winners is: " ,max_event, "with ", max_winners, " athletes.")
top_events = winners_counts.head(5)

explodes = [0.1 if i == top_events.max() else 0 for i in top_events]
labels = [f'{events.split(',')[0:2]}' for sport, events in top_events.index]

winners = combined[combined['medal'].isin(['Gold','Silver','Bronze'])]
winners_counts = winners.groupby(['sport', 'event'])['name'].nunique().sort_values(ascending = False)
max_event, max_winners = summarize_max(winners_counts, description = 'The Sport and The Event with The Most Winners')
top_events = winners_counts.head(5)
explodes = [0.1 if i == top_events.max() else 0 for i in top_events]
labels = [f'{events.split(",")[0:2]}' for sport, events in top_events.index]

fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.pie(top_events, labels=labels, explode=explodes, autopct='%.2f%%', pctdistance=0.7)
ax1.set_title('Top 5 Sports / Events by the Percentage of Winners')
st.pyplot(fig1)


# Which sports dominate Summer and Winter seasons
st.subheader("2. Which Sports Dominate Summer vs Winter?")
st.markdown("Top 5 sports for each season based on medal count:")

medal_events = combined.drop_duplicates(['year', 'event', 'medal', 'type'])
count = medal_events.groupby(['sport', 'type'])['medal'].count().reset_index()
sport_top = count.loc[count.groupby('type')['medal'].idxmax()]
st.write("Top sports by season:")
st.dataframe(sport_top)

for season in count['type'].unique():
    season_data = count[count['type'] == season].sort_values('medal', ascending=False)[:5]
    fig, ax = plt.subplots()
    ax.barh(season_data['sport'], season_data['medal'], color='skyblue')
    ax.set_xlabel('Number of Medals')
    ax.set_ylabel(f'{season} Sports')
    ax.set_title(f'Top 5 {season} Sports')
    st.pyplot(fig)



st.subheader("3. Height and Weight Outliers: Did They Win More Events?")
st.markdown("""
- Compare win rates between outliers (extreme height/weight) vs normal athletes.
- Plotted below for visualization.
""")

# 3. checking for Height outliers

Q1 = combined['height_cm'].quantile(0.25)
Q3 = combined['height_cm'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

height_outliers = combined[(combined['height_cm'] < lower_bound) | (combined['height_cm'] > upper_bound)]

# checking for weight outliers

Q1_w = combined['weight_kg'].quantile(0.25)
Q3_w = combined['weight_kg'].quantile(0.75)

IQR_w = Q3_w - Q1_w

lower_bound_w = Q1_w - 1.5 * IQR_w
upper_bound_w = Q3_w + 1.5 * IQR_w

weight_outliers = combined[(combined['weight_kg'] < lower_bound_w) | (combined['weight_kg'] > upper_bound_w)]

st.markdown("""
**Comparison Methodology:**
1. Keep height/weight outlier athletes who won a medal.
2. Non-outlier dataset = clean data without outliers.
3. Normal winners = non-outlier athletes who won a medal.
4. Calculate win rate for both height and weight categories.
""")

height_outliers_winners = height_outliers[height_outliers['medal'].isin(['Gold', 'Silver', 'Bronze'])]
non_outlier = combined[~combined.index.isin(height_outliers.index)]
normal_winners = non_outlier[non_outlier['medal'].isin(['Gold', 'Silver', 'Bronze'])]

weight_outliers_winners = weight_outliers[weight_outliers['medal'].isin(['Gold','Silver','Bronze'])]
non_outlier_w = combined[~combined.index.isin(weight_outliers.index)]
normal_winners_w = non_outlier_w[non_outlier_w['medal'].isin(['Gold', 'Silver', 'Bronze'])]

Height_Outlier_win_rate =  (len(height_outliers_winners) / len(height_outliers))*100
Height_NonOutlier_win_rate = (len(normal_winners) / len(non_outlier))*100
Weight_Outlier_win_rate = (len(weight_outliers_winners) / len(weight_outliers))*100
Weight_NonOutlier_win_rate = (len(normal_winners_w) / len(non_outlier_w))*100

# converting our data to a table
data = {
    'Category' : ['Height', 'Weight'],
    'Outlier Win Rate' : [Height_Outlier_win_rate, Weight_Outlier_win_rate],
    'Non-outlier Win Rate' : [Height_NonOutlier_win_rate, Weight_NonOutlier_win_rate],
    'Difference(%)' : [(Height_Outlier_win_rate - Height_NonOutlier_win_rate), (Weight_Outlier_win_rate - Weight_NonOutlier_win_rate)]
}

win_table = pd.DataFrame(data)   
ans = win_table.round(2)

# Plotting the graph 
metrics = ['Height', 'Weight']
outlier_rates = [Height_Outlier_win_rate, Weight_Outlier_win_rate]
non_outlier_rates = [Height_NonOutlier_win_rate, Weight_NonOutlier_win_rate]


fig, ax = plt.subplots()
x = range(len(metrics))
ax.bar(x, outlier_rates, color='brown', width=0.3, label='Outliers')
ax.bar([i+0.3 for i in x], non_outlier_rates, color='grey', width=0.3, label='Non-Outliers')
ax.set_title('Win Rate: Outliers vs Non-Outliers')
ax.set_ylabel('Win Rate')
ax.set_xticks([i+0.15 for i in x])
ax.set_xticklabels(metrics)
ax.legend()
st.pyplot(fig)
st.dataframe(ans)

# Observations
st.markdown("""
- **Participation & Popularity:** Football and Ice Hockey dominate in terms of athlete participation and medal counts.
- **Physical Attributes Advantage:** Taller athletes had a 7.9% higher win rate; heavier athletes had a 5.6% higher win rate.
- **Season & Suitability:** Summer → Athletics & Swimming; Winter → Skating & Skiing.
- **Conclusion:** Most successful sports have high global participation, align with optimal physical attributes, and fit climate/season conditions.
""")

st.header("Part 2: The role of age in performance")
st.subheader("1. How has the average age of medalists changed by each decade?")

combined['Age_of_Competition'] = np.where(
    (combined['year'] == 0) | (combined['birth_year'] == 0),
    -1,
    combined['year'] - combined['birth_year']
)

# valid ages: only keep realistic competition ages
valid_data = combined[
    (combined['year'] > 0) &
    (combined['Age_of_Competition'] > 15 ) & 
    (combined['Age_of_Competition'] < 60)  
]

avg_age_year = valid_data.groupby('year')['Age_of_Competition'].mean().reset_index()

# Plot in Streamlit
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(avg_age_year['year'], avg_age_year['Age_of_Competition'], marker='o', color='green')
ax.set_title('Average Age of Medalists over Time', fontsize=14)
ax.set_xlabel('Year')
ax.set_ylabel('Average Age')
ax.grid(True)

st.pyplot(fig)


st.markdown("""
### The average age of Olympic medalists has shown some variation across decades:
- In the 1890s, athletes were generally the youngest, averaging around 25 years old.
- By the 1920s, the trend shifted upward, with the highest average age touching nearly 30 years old.

- Across other decades, the age fluctuated but consistently centered around 27 years, which appears to be the biological “peak” performance age for Olympic athletes.
  
- This observation aligns with external studies analyzing Olympic track and field data, which also identified 27 years as the median peak performance age.
  
- The fact that Athletics dominates the Summer Olympics—and has the highest medal counts—further reinforces this age clustering, as most medalists in Athletics reach their prime during their mid to late 20s

While athlete ages have shifted slightly between decades, the overall pattern confirms that the mid-to-late 20s is the prime age for Olympic success, with the 1920s being the notable outlier where average medalist age peaked close to 30.
""")

# ### Part 3: Longevity & Consistency

# 6. How do repeat athletes perform over multiple olympics?

st.header("Part 3: Longevity & Consistency")
st.subheader("1. How do repeat athletes perform over multiple olympics?")

repeat_athletes =  combined.groupby('name')['year'].nunique().reset_index()
repeat_athletes = repeat_athletes[repeat_athletes['year'] > 1]

# getting all the data of repeat athletes
repeat_winners = combined.merge(
    repeat_athletes[['name']], on='name', how='inner'
)[['name', 'year', 'sport', 'event', 'medal','place']]


plt.figure(figsize=(21,20))

# taking the best place per year per athlete 
best_place_per_year = repeat_winners.groupby(['name','year'])['place'].min().reset_index()

fig1, ax1 = plt.subplots(figsize=(21, 20))
sns.boxplot(data=best_place_per_year, x='year', y='place', ax=ax1)
ax1.set_ylabel("Place (lower = better)")
ax1.set_title("Distribution of Repeat Athletes' Performance by Year")
st.pyplot(fig1)

st.markdown("""
### Observations

- **Repeat Athletes’ Performance:**  
  - Early Olympics (till 1940s): repeat athletes clustered in top 1–5 ranks.  
  - After 1950s: range widened, with outliers reaching beyond 100th place (peaking ~140th in 2016).  
  - Median consistently stayed between 1st–3rd place → repeat athletes usually sustain medal-winning form.  

Repeat athletes show both **career longevity** and **longer lifespans**, proving consistent training brings medals *and* health.
""")


# 7. Does sport type affect athlete lifespan?

st.subheader("2. Does sport type affect athlete lifespan?")

valid_results = combined[
    (combined['sport'].notna()) &
    (combined['birth_year'].notna()) &
    (combined['death_year'].notna()) &
    (combined['birth_year'] > 0) &
    (combined['death_year'] > 0)
].copy()

valid_results['lifespan'] = valid_results['death_year'] - valid_results['birth_year']

lifespan_by_sport = valid_results.groupby('sport')['lifespan'].mean().reset_index()

lifespan_by_sport.sort_values('lifespan', ascending = False)

fig2, ax2 = plt.subplots(figsize=(14, 28))
sns.boxplot(
    data=lifespan_by_sport, 
    y='sport', 
    x='lifespan', 
    hue='sport',
    dodge=False,
    legend=False,
    ax=ax2
)
ax2.set_xlabel('Age at Death')
ax2.set_ylabel('Sport')
ax2.set_title('Distribution of Athlete Lifespans by Sport')
ax2.grid(axis='y', linestyle='--', alpha=0.8)
st.pyplot(fig2)


# Observations
st.markdown("""
### Observations
- **Lifespan by Sport:**  
  - Average athlete lifespan ≈ **70 years**, higher than global average.  
  - Winter Pentathlon athletes lived longest (~90 years), possibly due to endurance training and active lifestyle.  

Repeat athletes show both **career longevity** and **longer lifespans**, proving consistent training brings medals *and* health.
""")


# ### Part 4: Countries and Dominance

# 8. Are some countries most successful in certain sports?

st.header("Part 4: Countries and Dominance")
st.subheader("1. Are some countries most successful in certain sports?")

# 8️⃣ Grouping medals by NOC and sport (corrected)

# 1. Keep only medal winners
winners = combined[combined['medal'].isin(['Gold','Silver','Bronze'])]

# 2. Drop duplicates so each medal event counts only once per NOC per sport
unique_medals = winners.drop_duplicates(
    subset=['year', 'event', 'medal', 'sport', 'NOC']
).copy()  

# 3. Handle multi-NOC entries (split and explode)
unique_medals['NOC'] = unique_medals['NOC'].str.split(',')
unique_medals = unique_medals.explode('NOC')

# 4. Group by NOC and sport
success = unique_medals.groupby(['NOC','sport'])['medal'].count().reset_index()

# 5. Sort by total medals
success = success.sort_values(by='medal', ascending=False)

top10 = success.head(10)
fig1, ax1 = plt.subplots(figsize=(10,6))
sns.barplot(data=top10, x='medal', y='sport', hue='NOC', ax=ax1)
ax1.set_title("Top 10 Country-Sport Successes (Unique Medal Events)")
ax1.set_xlabel("Number of Medals")
ax1.set_ylabel("Sport")
st.pyplot(fig1)

# Explanation
st.markdown("""
### Country dominance
- **USA** → Athletics, Swimming, Diving, Wrestling  
- **Soviet Union** → Gymnastics  
- **Norway** → Cross-country skiing  
- **Austria** → Alpine skiing  

**Insight:** Success often reflects culture, training, and geography.
""")

# 9. How has performance shifted in Summer vs. Winter Olympics over decades?

st.subheader("2. How has performance shifted in Summer vs. Winter Olympics over decades?")


# 9.
valid_place_data = combined[
    (combined['place'].notna()) & 
    (combined['place'] > 0) & 
    (combined['year'] > 0)
]

valid_place_data = valid_place_data.copy()

valid_place_data['decade'] = (valid_place_data['year'] // 10) * 10

performance_by_decade = valid_place_data.groupby(['type', 'decade'])['place'].mean().reset_index()

fig2, ax2 = plt.subplots(figsize=(12,6))
sns.lineplot(data=performance_by_decade, x='decade', y='place', hue='type', marker='o', ax=ax2)
ax2.set_title('Average Place in Summer vs Winter Olympics Over Decades')
ax2.set_ylabel('Average Place (lower is better)')
ax2.set_xlabel('Decade')
ax2.grid(True)
st.pyplot(fig2)

# Explanation
st.markdown("""
### Performance trends
- **Summer Olympics**: Rose (1880s–1950s), dipped in 1980s, steady since (~13th place today vs 3rd in 1890s).  
- **Winter Olympics**: Climbed alongside Summer but ~4 places behind since 1980s (2020 → Summer 13th, Winter 19th).  

**Insight:** Countries excel in sports aligned with environment and resources.  
Summer remains slightly stronger overall; Winter reflects limited global access.


---


""")

st.header("Conclusion")


st.markdown("""
-The Olympic data reveals that success in sports is not random—it is shaped by participation, physical attributes, age, and country-specific advantages.

### Participation & Events
- Sports with more events and global reach (Athletics, Football) naturally dominate medal counts.  

### Biology
- Peak athletic performance clusters around **age 27**, with taller/heavier outliers excelling in specific events.  

### Consistency
- Repeat athletes prove that long-term training sustains top performance and contributes to longer lifespans.  

### Geography & Culture
- Countries dominate where climate, resources, and culture align with a sport (e.g., Norway in skiing, USA in athletics).  

---

**Overall:** A sport is “successful” when it attracts high participation, aligns with optimal physical/biological traits, and is backed by strong cultural or national support.
""")







