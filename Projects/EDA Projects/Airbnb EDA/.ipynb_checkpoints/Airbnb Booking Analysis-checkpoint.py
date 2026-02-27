#!/usr/bin/env python
# coding: utf-8

# <img src="airbnb.png">

# ![AirbnbAirBnbGIF.gif](attachment:bc7e286c-8cfe-4fbe-841f-b2c71ae96bca.gif)

# # Lets get started😁

# ## Problem Statements

# ### 🏙️ **Neighborhood & Location Analysis**
# 
# 1. **What are the most popular neighborhoods for Airbnb rentals in NYC?**
#    *(based on listing count, reviews, availability)*
# 
# 2. **How do average prices and availability vary by neighborhood and neighborhood group (borough)?**
# 
# 3. **Which neighborhoods have the highest and lowest occupancy potential (based on availability & reviews)?**
# 
# 4. **What are the top listing locations for travelers in terms of price, convenience, and availability?**
# 
# ---
# 
# ### 📈 **Market Trends & Time Analysis**
# 
# 5. **How has the Airbnb market changed over time in terms of number of listings, price trends, and reviews?**
# 
# 6. **What are the busiest months for Airbnb in NYC (seasonal trends)?**
# 
# 7. **Are certain boroughs showing increasing or decreasing trends in listing counts or popularity?**
# 
# ---
# 
# ### 🛏️ **Property Type & Room Type Analysis**
# 
# 8. **Which room types (Entire home, Private room, Shared room) are most common in each borough?**
# 
# 9. **Are certain property/room types more expensive or popular than others?**
# 
# 10. **Which room types are most reviewed in each neighborhood group per month?**
# 
# ---
# 
# ### 💰 **Price & Review Correlation**
# 
# 11. **Which features (e.g., room type, availability, location, reviews) are correlated with Airbnb prices?**
# 
# ---
# 
# ### 🧳 **Booking Behavior & Stay Duration**
# 
# 12. **How do minimum night stays vary by neighborhood and room type?**
# 
# 13. **Which neighborhoods attract longer or shorter stays on average?**
# 
# ---
# 
# ### 💬 **Review Insights**
# 
# 14. **What’s the total number of reviews per neighborhood group?**
# 
# 15. **Which listings have the maximum number of reviews by neighborhood?**
# 
# 16. **How does number of reviews correlate with availability and price?**
# 
# ---
# 
# ### 🌟 **Bonus Advanced Questions**
# 
# 17. **Which hosts own the most properties — are they professional operators?**
# 
# 18. **Can we recommend top 5 listings for a traveler based on reviews, price, and availability?**
# 
# ---

# ## Import Necessary Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# ## Load the Dataset

# In[2]:


df = pd.read_csv("Airbnb NYC 2019 dataset.csv")
df


# ## About the Dataset – Airbnb NYC Bookings (2019)
# 
# This dataset comprises approximately **49,000 Airbnb listings** from **New York City**, collected in 2019. It includes **16 diverse columns** spanning both **categorical** and **numerical** features — offering detailed insights into each listing's characteristics such as price, availability, room type, and user engagement through reviews.
# 
# The richness of the dataset makes it highly suitable for exploring:
# 
# * **Market trends and seasonal patterns**
# * **Geographic and pricing distribution**
# * **User preferences and booking behavior**
# 
# By analyzing this data, we can uncover meaningful patterns in Airbnb usage across different NYC boroughs and gain actionable insights into the dynamics of short-term rentals in one of the world’s most visited cities.
# 

# ## Understanding the given columns

# In[3]:


df.columns


# In[4]:


# renaming the column name as it is too long
df = df.rename(columns={'id':'listing_id', 'name':'listing_name', 'calculated_host_listings_count':'host_listings_count'})
df


# - 🆔 **`listing_id`**: Unique identifier assigned to each listing in the dataset.
# 
# - 🏠 **`listing_name`**: Title of the Airbnb listing as shown on the platform.
# 
# - 🧑‍💼 **`host_id`**: Unique identifier for the host of the listing.
# 
# - 👤 **`host_name`**: Name of the host as displayed publicly.
# 
# - 🌆 **`neighbourhood_group`**: NYC borough where the listing is located (e.g., Manhattan, Brooklyn).
# 
# - 🏘️ **`neighbourhood`**: Specific neighborhood or local area within the borough.
# 
# - 📍 **`latitude`**: Geographic latitude coordinate of the listing.
# 
# - 📍 **`longitude`**: Geographic longitude coordinate of the listing.
# 
# - 🛏️ **`room_type`**: Type of accommodation offered (e.g., Entire home, Private room, Shared room).
# 
# - 💵 **`price`**: Nightly rental price in USD.
# 
# - 📅 **`minimum_nights`**: Minimum number of nights required per booking.
# 
# - ✍️ **`number_of_reviews`**: Total number of reviews received by the listing.
# 
# - 🗓️ **`reviews_per_month`**: Average number of reviews the listing receives each month.
# 
# - 🧾 **`host_listings_count`**: Total number of listings owned by the host.
# 
# - 📆 **`availability_365`**: Number of days in a year that the listing is available for booking.
# 

# In[5]:


df.shape


# In[6]:


df.info()


# So we have null values in name, host_name, last_review and reviews_per_month columns

# In[7]:


df.dtypes


# So, host_name, neighbourhood_group, neighbourhood and room_type fall into categorical variable category.
# 
# While host_id, latitude, longitude, price, minimum_nights, number_of_reviews, last_review, reviews_per_month, host_listings_count, availability_365 are numerical variables

# ## Summary Statistics

# In[8]:


df.describe()


# 🌍 **latitude & longitude**
# - Values are valid (NYC’s lat-long range: ~40.5 to 40.9, -74.25 to -73.70)
# 
# 💰 **price**
# - Mean price: $152.72
# 
# - Max price: $10,000 → Likely an outlier 😬
# 
# 📅 **minimum_nights**
# - Mean: ~7 nights
# 
# - (but max = 1250) → this is very high! -> This can be an outlier
# 
# ✍️ **number_of_reviews**
# - Mean: ~23 reviews per listing
# 
# - Some listings have 0 reviews (new or inactive), and some have up to 629 reviews → high engagement

# ## Missing Values & Duplicates

# In[9]:


print("Number of duplicates:", df.duplicated().sum())


# In[10]:


df.isna().sum()


# So we have missing values in **name, host_name, last_review** and **reviews_per_month** columns

# In[11]:


df['listing_name'].fillna('unknown',inplace=True)
df['host_name'].fillna('unknown',inplace=True)


# In[12]:


df.isna().sum()


# For **reviews_per_month** column, listings can have 0 reviews per month so best to fill it with 0.

# In[13]:


df['reviews_per_month'].fillna(0, inplace=True)


# In[14]:


# the null values are replaced by 0 value
df['reviews_per_month'].isnull().sum()


# In[15]:


df.sample(5)


# In[16]:


# there are some listings where the listing names and host names are equal
df[df['listing_name']==df['host_name']].head()


# ## Outlier Detection

# In[17]:


df.describe()


# ### 💰 price
# Max = 10,000
# 
# Mean = 152.7, 75th percentile = 175, Median = 106
# 
# 🔥 Evidence of outliers:
# 
# Listings priced > $500–1000 are extremely rare and unrealistic
# 
# A few luxury or test listings with inflated prices
# 
# ### 📅 minimum_nights
# Max = 1250, but median = 3, 75th percentile = 5
# 
# ❗ Evidence:
# 
# 99% of listings require under 30 nights
# 
# minimum_nights > 365 = not practical (e.g., test data, long-term stays)
# 
# ### 🧾 host_listings_count
# Max = 327, Mean = 7
# 
# 👀 Most hosts own 1–2 listings; 327 suggests a commercial host

# In[18]:


cols = ['price', 'minimum_nights', 'host_listings_count']
for col in cols:
    sns.boxplot(x=df[col])
    plt.show()


# **price, minimum_nights, number_of_reviews,** and **host_listings_count** have outliers

# ## Removing Outliers using IQR method

# In[19]:


def remove_outliers(df, cols):
    df_clean = df.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        LB = Q1 - (1.5 * IQR)
        UB = Q3 + (1.5 * IQR)
        df_clean = df_clean[(df_clean[col] >= LB) & (df_clean[col] <= UB)]
    return df_clean


# In[20]:


cols = ['price', 'minimum_nights', 'number_of_reviews', 'host_listings_count']

df_cleaned = remove_outliers(df, cols)

# Check shapes
print("Original dataset:", df.shape)
print("After outlier removal:", df_cleaned.shape)


# In[21]:


cols = ['price', 'minimum_nights', 'host_listings_count']
for col in cols:
    sns.boxplot(x=df_cleaned[col])
    plt.show()


# # 📊Data Analysis

# ## Distribution of Price Range

# In[22]:


plt.figure(figsize=(15,5))
sns.histplot(
            data=df_cleaned,
            x = 'price',
            bins=30,
            kde=True,
            color='salmon'
)
plt.title('Distribution of Airbnb Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# ### Observations
# - The curve is heavily skewed to the right i.e. Most listings are on the lower price side and a few very expensive listings pull the tail far right.
# - Prices charged by Airbnb ranges from \\$20 to \\$330.
# - Most of the prices are falling between \\$50 to \\$150.
# - The number of listings significantly drops after \\$150 – \\$200.
# - Few luxury listings go up to ~\\$300+, but they're rare.
# - This suggests affordability is a key factor in listing popularity.

# ## Total Listing counts for each neighbourhood group

# In[23]:


total_listings = df_cleaned['neighbourhood_group'].value_counts()
total_listings


# In[24]:


plt.figure(figsize=(15,5))
sns.barplot(x=total_listings.index, y= total_listings.values, palette='flare')
plt.title("Listing Counts for Neighbourhood Group in NYC", fontsize=24)
plt.xlabel("Neighbourhood Group", fontsize=14)
plt.ylabel("Total Listing Counts", fontsize=14)
plt.tight_layout()
plt.show()


# ### Observations
# 
# - Brookyln shows highest number of listings (~14,000) — indicating a high density of Airbnb activity
# - Manhattan Comes Second ~13,000 listings — slightly behind Brooklyn
# - Queens and Bronx have fewer listings with 3670 and 754 respectively. Likely due to lower tourist activity and accessibility challenges

# In[25]:


df_cleaned.to_csv("Airbnb dataset.csv")


# ## 1. Most popular neighborhoods for Airbnb rentals in NYC? (based on listing count, reviews, availability)

# In[26]:


# Based on Listing count
listing_counts = df_cleaned.groupby('neighbourhood')['listing_id'].count().sort_values(ascending=False).reset_index().rename(columns={"listing_id": "listing_count"})
listing_counts


# In[27]:


top_10 = listing_counts.head(10)
top_10


# In[28]:


plt.figure(figsize=(12,6))
sns.barplot(data=top_10, x='listing_count', y='neighbourhood', palette='flare')
plt.title('Top 10 Neighborhoods in NYC by Listing Count')
plt.xlabel('Number of Listings')
plt.ylabel('Neighborhood')
plt.show()


# ### Observations

# - 🏆 Williamsburg tops the chart with nearly 3,000 listings — it's a cultural hotspot in Brooklyn known for its nightlife and arts scene.
# 
# - Bedford-Stuyvesant, Harlem, and Bushwick also show strong presence, suggesting high Airbnb activity in both Brooklyn and northern Manhattan.
# 
# - Neighborhoods like East Village, Upper West Side, and Crown Heights represent high tourist interest or host availability in centrally located residential areas.

# In[29]:


# Other ways to showcase using aggregate function
# Group by neighborhood
neighborhood_stats = df_cleaned.groupby('neighbourhood').agg(
    total_listings=('listing_id', 'count'),
    total_reviews=('number_of_reviews', 'sum'),
    avg_availability=('availability_365', 'mean')
).sort_values(by='total_listings', ascending=False) #sort by total_listings or total_reviews or avg_availability

# View top 10 neighborhoods
neighborhood_stats.head(10)


# In[30]:


# based on reviews

reviews_count = df_cleaned.groupby('neighbourhood')['number_of_reviews'].sum().sort_values(ascending=False).reset_index().rename(columns={'number_of_reviews':'total_reviews'})
reviews_count


# In[31]:


top_10 = reviews_count.head(10)
top_10


# In[32]:


plt.figure(figsize=(12,6))
sns.barplot(data=top_10, x=top_10['total_reviews'], y=top_10['neighbourhood'], palette='flare')
plt.title('Top 10 Neighborhoods in NYC by Reviews Count')
plt.xlabel('Number of Reviews')
plt.ylabel('Neighborhood')
plt.show()


# ### Observations

# - 🏆 Bedford-Stuyvesant leads the pack. Even though Williamsburg topped by listings, Bedford-Stuyvesant slightly edges out in guest activity. This suggests high guest turnover or engagement in this neighborhood.
# 
# - Williamsburg stays strong and very close second — confirms its position as both a high-supply and high-demand neighborhood.
# 
# - Harlem, Bushwick, and East Village follow strong mix of cultural hotspots and affordable options. These neighborhoods combine a decent number of listings with high booking activity.
# 
# - Neighborhoods like Crown Heights, Upper East Side, and Hell’s Kitchen rank in reviews, even if they didn’t make the top of the listing count chart. This suggests they have fewer listings but highly active ones.

# In[33]:


# based on availabilty

availability_count = df_cleaned.groupby('neighbourhood').agg(
        average_availability = ('availability_365', 'mean')
).reset_index().sort_values(by='average_availability', ascending=False)
availability_count


# In[34]:


top_10 = availability_count.head(10)
top_10


# In[35]:


plt.figure(figsize=(12,6))
sns.barplot(data=top_10, x=top_10['average_availability'], y=top_10['neighbourhood'], palette='flare')
plt.title('Top 10 Neighborhoods in NYC by availability Count')
plt.xlabel('Average Availability')
plt.ylabel('Neighborhood')
plt.show()


# ### Observations

# 1. 🏆 Co-op City has the highest average availability, close to 365 days — indicating that listings here are open almost all year round.
# 
# 2. Willowbrook and Spuyten Duyvil also show exceptionally high availability, suggesting hosts in these areas are highly active or using Airbnb as a full-time rental platform.
# 
# 3. Most of these top neighborhoods did not appear in the top charts for listing count or review volume, implying:
# 
#     - Lower overall guest demand
#     - Possibly fewer total listings
#     - But consistently available for booking
# 
# 4. These areas may be less competitive, offering opportunities for:
#     - Travelers looking for flexibility
#     - New hosts entering the market
# 
# 5. The chart suggests that these neighborhoods are more supply-ready than demand-heavy — they have potential but might not be fully utilized yet.
# 6. These neighborhoods could be ideal for long-term or business travelers, given the high availability and less fluctuation in listing status.

# ## 2. How do average prices and availability vary by neighborhood and neighborhood group (borough)?

# ### By Neighbourhood

# In[36]:


# average prices by neighborhood
neighbourhood_summary = df_cleaned.groupby('neighbourhood').agg(
        average_prices = ('price', 'mean'),
        average_availability = ('availability_365', 'mean')
).reset_index().sort_values(by='average_prices', ascending=False)

neighbourhood_summary


# In[37]:


plt.figure(figsize=(17, 5))
sns.barplot(data=neighbourhood_summary.head(10), x='neighbourhood', y='average_prices', palette='flare')
plt.title('Average Price by Neighbourhood')
plt.show()


# ### Observations

# - 🏆 Eltingville has the highest average price, around \\$300+, making it the most expensive neighborhood for Airbnb listings in this chart.
# - Willowbrook and Neponsit follow closely with average prices above $240.
# - Neighborhoods like Breezy Point and NoHo also show higher-than-expected prices, indicating small but high-value markets.

# In[38]:


# availability vary by neighborhood
neighbourhood_summary = df_cleaned.groupby('neighbourhood').agg(
        average_prices = ('price', 'mean'),
        average_availability = ('availability_365', 'mean')
).reset_index().sort_values(by='average_availability', ascending=False)

neighbourhood_summary


# In[39]:


plt.figure(figsize=(15, 5))
sns.barplot(data=neighbourhood_summary.head(10), x='neighbourhood', y='average_availability', palette='flare')
plt.title('Average Availability by Neighbourhood')
plt.show()


# ### Observations

# - 🏆 Co-op City, Willowbrook, and Spuyten Duyvil have the highest average availability (300+ days), indicating year-round open listings.
# 
# - All top 10 neighborhoods show consistently high availability (>240 days), making them attractive for long-term or flexible stays.
# 
# - These areas are mostly non-touristy, residential zones — likely with lower competition and more full-time hosts.
# 
# Willowbrook stands out for appearing in both high price and high availability charts — ideal for premium, committed hosting.

# ### By Neighbourhood Group

# In[40]:


# average prices by neighborhood_group
neighbourhood_group_summary = df_cleaned.groupby('neighbourhood_group').agg(
        average_prices = ('price', 'mean'),
        average_availability = ('availability_365', 'mean')
).reset_index().sort_values(by='average_prices', ascending=False)

neighbourhood_group_summary


# In[41]:


plt.figure(figsize=(10, 5))
sns.barplot(data=neighbourhood_group_summary, x='neighbourhood_group', y='average_prices', palette='flare')
plt.title('Average Price by Neighbourhood Group')
plt.show()


# ### Observations

# - Manhattan has the highest average Airbnb price (~$145), reflecting its premium status and high demand.
# 
# - Brooklyn ranks second, offering a balance of price and popularity.
# 
# - Staten Island, Queens, and Bronx have lower average prices, with Bronx being the most affordable.
# 
# - Price generally decreases as you move away from central/tourist-heavy neighbourhoods.

# In[42]:


# average availability by neighborhood_group
neighbourhood_group_summary = df_cleaned.groupby('neighbourhood_group').agg(
        average_prices = ('price', 'mean'),
        average_availability = ('availability_365', 'mean')
).reset_index().sort_values(by='average_availability', ascending=False)

neighbourhood_group_summary


# In[43]:


plt.figure(figsize=(10, 5))
sns.barplot(data=neighbourhood_group_summary, x='neighbourhood_group', y='average_availability', palette='flare')
plt.title('Average Availability by Neighbourhood Group')
plt.show()


# ### Observations

# - Staten Island has the highest average availability (~180 days), showing listings are open for nearly half the year or more.
# 
# - Bronx and Queens follow with moderate availability (~150 and ~120 days respectively).
# 
# - Brooklyn and Manhattan have the lowest availability — despite being the most expensive and in-demand boroughs.
# 
# - This suggests that while Manhattan and Brooklyn listings are pricey, they’re often less available — likely due to:
# 
#     - High booking frequency
# 
#     - Regulatory limits
# 
#     - Hosts offering listings seasonally

# ## 3. Which neighborhoods have the highest and lowest occupancy potential (based on availability & reviews)?

# Occupancy potential is a proxy metric you calculated as:
# 
# $$ Occupancy\space Potential = \frac{Total\space Reviews}{Total\space Availability (days)} $$
#  
# This tells you how many reviews a neighborhood gets per available day across all listings. This gives a sense of how frequently guests are reviewing relative to how often listings are available.

# In[44]:


occupancy_df = df_cleaned.groupby('neighbourhood').agg(
    total_reviews=('number_of_reviews', 'sum'),
    total_availability=('availability_365', 'sum')
).reset_index()
occupancy_df


# In[45]:


# Step 2: Calculate occupancy potential
occupancy_df['occupancy_potential'] = occupancy_df['total_reviews'] / occupancy_df['total_availability']
occupancy_df


# In[46]:


occupancy_df = occupancy_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['occupancy_potential'])
occupancy_df


# In[47]:


highest_occupancy = occupancy_df.sort_values(by='occupancy_potential', ascending=False).head(10)
highest_occupancy


# In[48]:


plt.figure(figsize=(19,7))
sns.barplot(data = highest_occupancy, y='occupancy_potential', x='neighbourhood', palette='flare')
plt.title('Highest Occupancy Potential')
plt.xlabel('Neighbourhood')
plt.ylabel('Occupancy Potential')
plt.show()


# ### Observations

# - Castle Hill leads with 0.76 occupancy potential — almost 1 review every 1.3 days available! That’s incredibly high engagement.
# 
# - Navy Yard, Arden Heights, and Cobble Hill also have strong demand vs availability, showing guests actively book and review properties there.
# 
# - Even neighborhoods with moderate availability (like Battery Park City or Downtown Brooklyn) perform very well — suggesting they efficiently convert their availability into bookings.
# 
# - Nolita has a high number of reviews and good potential despite high availability — a great indicator of steady traffic.

# In[49]:


lowest_occupancy = occupancy_df.sort_values(by='occupancy_potential').head(10)
lowest_occupancy


# In[50]:


plt.figure(figsize=(19,7))
sns.barplot(data = lowest_occupancy, y='occupancy_potential', x='neighbourhood', palette='flare')
plt.title('Lowest Occupancy Potential')
plt.xlabel('Neighbourhood')
plt.ylabel('Occupancy Potential')
plt.show()


# ### Observations

# - Eastchester and Eltingville had zero reviews despite being available for booking — indicating no guest engagement at all.
# 
# - Co-op City and Little Neck had less than 3 reviews while being available almost the full year — suggesting extremely poor utilization.
# 
# - Oakwood, West Farms, and Riverdale show very low booking frequency, with occupancy potential below 2%.
# 
# - Edgemere and Sea Gate had slightly better performance but still less than 1 review per 50 days available — pointing to weak guest demand.
# 
# - Overall, these areas are not ideal for hosting unless listing visibility, pricing, or marketing is significantly improved.

# ## 4. What are the top listing locations for travelers in terms of price, convenience, and availability?

# | Metric                            | Why it matters for travelers                       |
# | --------------------------------- | -------------------------------------------------- |
# | 💰 **Affordable Price**           | Budget-friendly stays                              |
# | 🚇 **Convenience (Review Count)** | More reviews → trusted, popular, likely accessible |
# | 📅 **High Availability**          | Easier to book anytime (flexible planning)         |

# In[51]:


traveler_df = df_cleaned.groupby('neighbourhood').agg(
    avg_price=('price', 'mean'),
    total_reviews=('number_of_reviews', 'sum'),
    avg_availability=('availability_365', 'mean'),
    listing_count=('listing_id', 'count')
).reset_index()
traveler_df


# In[52]:


# Filter for enough listings to avoid outliers
traveler_df = traveler_df[traveler_df['listing_count'] >= 10]
traveler_df


# In[53]:


# Filter for reasonable price range (e.g., 50 to 150 USD)
traveler_df = traveler_df[(traveler_df['avg_price'] >= 50) & (traveler_df['avg_price'] <= 150)]
traveler_df


# In[54]:


top_travel_areas = traveler_df.sort_values(by=['total_reviews', 'avg_availability'], ascending=[False, False]).head(10)
top_travel_areas


# ### Top NYC Neighborhoods for Travelers(Based on Price and Reviews)

# In[55]:


fig, ax1 = plt.subplots(figsize=(15, 7))

# 🟦 Bar plot for avg_price using seaborn
sns.barplot(
    x='neighbourhood',
    y='avg_price',
    data=top_travel_areas,
    ax=ax1,
    palette='flare',
    label='Avg Price (USD)'
)

ax1.set_xlabel('Neighbourhood')
ax1.set_ylabel('Avg Price (USD)', color='purple')
ax1.tick_params(axis='y', labelcolor='purple')
ax1.set_xticklabels(top_travel_areas['neighbourhood'], rotation=45, ha='right')
ax1.set_title('Top NYC Neighborhoods for Travelers\n(Based on Price and Reviews)')

# 🟧 Line plot for total_reviews using seaborn
ax2 = ax1.twinx()
sns.lineplot(
    x='neighbourhood',
    y='total_reviews',
    data=top_travel_areas,
    ax=ax2,
    color='orange',
    marker='o',
    label='Total Reviews'
)

ax2.set_ylabel('Total Reviews / Avg Availability', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# 🧭 Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

plt.tight_layout()
plt.show()


# ### Observations

# 1. 📈 High Review Counts (Popularity):
# 
# - Bedford-Stuyvesant and Williamsburg have the highest number of total reviews (~30,000+), indicating they are very popular among travelers.
# 
# - Harlem and Bushwick also show strong review numbers, suggesting high footfall and visitor engagement.
# 
# 2. 💰 Price Trends:
# 
# - Upper West Side and Lower East Side have the highest average prices (around \\$145–$150 USD), yet they don't have the highest reviews.
# 
# - Washington Heights and Bushwick offer lower prices (~\\$80–$90 USD) while still maintaining decent popularity, making them potentially good budget-friendly options.
# 
# 3. 📊 Trade-off Between Price & Reviews:
# 
# - Neighborhoods like East Harlem and Astoria strike a balance — moderate prices (~\\$100–$120 USD) and steady review activity — which makes them solid choices for average travelers.
# 
# 4. 🎯 Best Overall Travel Areas:
# 
# - Bedford-Stuyvesant appears to be the best value for travelers: high review count, reasonable price, and good availability.
# 
# - Williamsburg is a close second — slightly more expensive, but extremely popular and well-reviewed.
# 
# 5. 🔸 Declining Trend:
# 
# - As you move rightward in the plot (from top neighborhoods to lower ones), total reviews decline, even if prices don’t drop proportionally — indicating those areas are less frequented.

# ### Top NYC Neighborhoods for Travelers(Based on Price and Average Availability)

# In[56]:


fig, ax1 = plt.subplots(figsize=(15,7))

# bar plot for Average Price using Seaborn

sns.barplot(
    x='neighbourhood',
    y='avg_price',
    data=top_travel_areas,
    ax=ax1,
    palette='flare',
    label='Avg Price (USD)'
)

ax1.set_xlabel('Neighbourhood')
ax1.set_ylabel('Avg Price (USD)', color='purple')
ax1.tick_params(axis='y', labelcolor='purple')
ax1.set_xticklabels(top_travel_areas['neighbourhood'], rotation=45, ha='right')
ax1.set_title('Top NYC Neighborhoods for Travelers\n(Based on Price and Average Availability)')

# 🟧 Line plot for total_reviews using seaborn
ax2 = ax1.twinx()
sns.lineplot(
    x='neighbourhood',
    y='avg_availability',
    data=top_travel_areas,
    ax=ax2,
    color='blue',
    marker='o',
    label='Total Reviews'
)

ax2.set_ylabel('Average Availability', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# 🧭 Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

plt.tight_layout()
plt.show()


# ### Observations

# 1. 🟫 Average Price (Bar Chart)
# - Upper West Side and Lower East Side have the highest average prices (~\\$147–$149 USD), suggesting premium listings.
# 
# - Bushwick and Washington Heights are the most affordable, with prices ~\\$80–$85 USD.
# 
# 2. 🔵 Average Availability (Line Plot in Blue)
# - Astoria, Harlem, and Bedford-Stuyvesant show high average availability (around 75–85), indicating listings are open for most of the year.
# 
# - Upper West Side and Williamsburg show lower availability (around 50 or less), possibly due to high booking frequency or hosts' preferences.

# ## 5. How has the Airbnb market changed over time in terms of number of listings, price trends, and reviews?

# ### In terms of Number of Listings

# In[57]:


df_cleaned['last_review']= pd.to_datetime(df['last_review'], errors='coerce')


# In[58]:


df_cleaned['year'] = df_cleaned['last_review'].dt.year


# In[59]:


df_cleaned


# In[60]:


listings_per_year = df_cleaned.groupby('year')['listing_id'].nunique().reset_index(name='listing_count')
plt.figure(figsize=(15,5))
sns.lineplot(data=listings_per_year, x='year', y='listing_count', marker='o')
plt.title('Number of Listings Over Time')
plt.xlabel('Year')
plt.ylabel('Unique Listings')
plt.grid(True)
plt.show()


# ### Observations

# - Listings began increasing significantly after 2014, crossing 1,000+ in 2015, and more than doubled in 2016.
# 
# - The year 2019 saw an explosive rise, reaching 15,000+ unique listings, indicating peak Airbnb activity in NYC just before the COVID-19 pandemic.
# 
# - Between 2011 and 2014, Airbnb listings were minimal, likely because the platform was still gaining traction in NYC.

# ### In terms of Price Trends

# In[61]:


price_trends = df_cleaned.groupby('year')['price'].mean().reset_index()
price_trends


# In[62]:


plt.figure(figsize=(15,5))
sns.lineplot(data=price_trends, x='year', y='price', marker='o')
plt.title('Average Price Trend Over Time')
plt.xlabel('Year')
plt.ylabel('Average Price (USD)')
plt.grid(True)
plt.show()


# ### Observations

# - After peaking in 2013 (~\\$149 USD), the average price steadily declined until 2017, indicating a possible increase in competition or market stabilization.
# 
# - From 2017 to 2019, there’s a modest upward trend, suggesting price correction or better-quality listings entering the market.
# 
# - The prices in 2011–2013 were highly volatile, likely due to fewer listings and outliers (e.g., luxury apartments skewing the average).

# ### In terms of Reviews

# In[63]:


total_reviews_per_year = df_cleaned.groupby('year')['number_of_reviews'].sum().reset_index()
total_reviews_per_year


# In[64]:


plt.figure(figsize=(15,5))
sns.lineplot(data=total_reviews_per_year, x='year', y='number_of_reviews', marker='o', color='green')
plt.title('Total Reviews Over Time')
plt.xlabel('Year')
plt.ylabel('Total Reviews')
plt.grid(True)
plt.show()


# ### Observations

# - From 2015 onward, total reviews show a steep upward trajectory, reflecting a rapid increase in bookings and guest engagement.
# 
# - In 2019, total reviews surged past 300,000, nearly a 7x jump from 2018. This may indicate:
# 
#     - A significant boom in platform usage
#     - More guests leaving reviews
#     - Enhanced review mechanisms on Airbnb.
# 
# - From 2011 to 2014, reviews remained low and stagnant, likely due to:
# 
#     - Fewer users
#     - Newer market presence
#     - Lower trust or awareness in Airbnb as a platform

# ## 6. What are the busiest months for Airbnb in NYC (seasonal trends) ?

# In[65]:


df_cleaned['last_review']= pd.to_datetime(df['last_review'], errors='coerce')
df_cleaned['month'] = df_cleaned['last_review'].dt.month_name()
df_cleaned


# In[66]:


# Group by month and count reviews
monthly_reviews = df_cleaned.groupby('month')['number_of_reviews'].sum().reset_index()

month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

monthly_reviews['month'] = pd.Categorical(monthly_reviews['month'], categories=month_order, ordered=True)
monthly_reviews = monthly_reviews.sort_values('month')
monthly_reviews


# In[67]:


# Plot using Seaborn
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.barplot(data=monthly_reviews, x='month', y='number_of_reviews', palette='flare')
plt.title('Busiest months for Airbnb in NYC as per Reviews')
plt.ylabel('Total Reviews')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### Observations

# - June – Highest number of reviews, indicating peak travel season. 🌞
# 
# - July – Very high activity, following June. 🏖️
# 
# - May – Sharp rise in bookings as summer begins. 🌼
# 
# - January – Post-holiday stays still bring in decent reviews.
# 
# - April, September, December – Moderate engagement, possibly due to spring breaks and holidays.
# 
# - February – Lowest number of reviews; likely due to cold weather. ❄️
# 
# - March, August, November – Noticeably less demand compared to summer months.

# ## 7. Are certain boroughs showing increasing or decreasing trends in listing counts or popularity?

# ### Listing trend per borough

# In[68]:


borough_trends = df_cleaned.groupby(['neighbourhood_group', 'year']).agg({
    'listing_id': 'nunique',  # unique listings
    'number_of_reviews': 'sum'
}).reset_index().rename(columns={'listing_id': 'listing_count'})
borough_trends


# In[69]:


plt.figure(figsize=(12, 6))
sns.lineplot(data=borough_trends, x='year', y='listing_count', hue='neighbourhood_group', marker='o')
plt.title('Listing Count Trend by Borough')
plt.ylabel('Number of Listings')
plt.xlabel('Year')
plt.legend(title='Borough')
plt.grid(True)
plt.tight_layout()
plt.show()


# ### Observations

# 🟠Brooklyn
# - Shows the steepest growth in listing count, especially from 2016 to 2019.
# 
# - Became the most dominant borough for listings by 2019.
# 
# - Suggests Brooklyn is a top choice for hosts and possibly tourists due to affordability and space.
# 
# 🟢Manhattan
# - Consistent growth, but slightly overtaken by Brooklyn in recent years.
# 
# - High listing counts reflect its status as NYC’s central tourism hub.
# 
# - Growth is strong, but the curve flattens slightly post-2016 → signs of market saturation?
# 
# 🔴Queens
# - Listing count has steadily increased, with a big jump in 2019.
# 
# - Queens is emerging as a rising market, possibly due to lower costs and growing tourism interest.
# 
# 🔵Bronx
# - Growth is gradual but visible, with a noticeable rise in 2019.
# 
# - Still has the lowest listing count among the major boroughs (excluding Staten Island).
# 
# - Could be an untapped market for investment if growth continues.
# 
# 🟣Staten Island
# - Remains flat and lowest throughout the years.
# 
# - Suggests low tourist demand or host interest, possibly due to location/remoteness.

# ### Review trend per borough

# In[70]:


plt.figure(figsize=(12, 6))
sns.lineplot(data=borough_trends, x='year', y='number_of_reviews', hue='neighbourhood_group', marker='o')
plt.title('Total Reviews Trend by Borough')
plt.ylabel('Total Reviews')
plt.xlabel('Year')
plt.legend(title='Borough')
plt.grid(True)
plt.tight_layout()
plt.show()


# ### Observations

# 🟠 Brooklyn
# - 🚀 Leads in total reviews by 2019, showing it’s highly popular with guests.
# 
# - The sharp rise post-2017 indicates a spike in demand and satisfaction, possibly due to increased hosting and better marketing.
# 
# - Brooklyn continues to solidify its status as the top borough for guests.
# 
# 🟢 Manhattan
# - Shows a strong upward trend, with a steady increase in reviews from 2015 to 2019.
# 
# - Slightly behind Brooklyn in 2019, suggesting fierce competition between these two hubs.
# 
# - Consistent performance reflects tourist density and high listing activity.
# 
# 🔴 Queens
# - Reviews grew moderately until 2018, followed by a massive spike in 2019.
# 
# - Indicates that Queens is gaining popularity quickly, possibly due to better affordability or spillover demand from central areas.
# 
# - One of the most improved boroughs in terms of guest engagement.
# 
# 🔵 Bronx
# - Review count remains relatively low until 2018, but then increases significantly by 2019.
# 
# - Reflects growing guest interest, but the borough still trails behind others in popularity.
# 
# 🟣 Staten Island
# - Remains the least reviewed borough throughout the years.
# 
# - Minor growth suggests limited guest traffic and interest compared to others.

# ## 8. Which room types (Entire home, Private room, Shared room) are most common in each borough?

# In[71]:


room_types_summary = df_cleaned.groupby(['neighbourhood_group','room_type'])['listing_id'].count().reset_index().rename(columns={'listing_id': 'listing_count'})
room_types_summary.sort_values(by='listing_count', ascending=False)


# In[72]:


plt.figure(figsize=(12, 6))
sns.barplot(data=room_types_summary, 
            x='neighbourhood_group', 
            y='listing_count', 
            hue='room_type',
            palette = 'flare')

plt.title('Room Type Distribution by Borough in NYC')
plt.xlabel('Borough')
plt.ylabel('Number of Listings')
plt.legend(title='Room Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### Observations

# 🗽 Manhattan
# 
# - Manhattan leads with the highest number of entire home listings, indicating high tourist demand and host preference for full-property rentals.
# 
# 🌳 Brooklyn
# 
# - Brooklyn shows a slight lead in private rooms over entire homes, suggesting a stronger community-driven or budget-conscious hosting culture.
# 
# 🏙️ Queens
# 
# - A clear majority of listings are private rooms, aligning with its residential nature and affordability.
# 
# 🏞️ Bronx
# 
# - Private rooms dominate here as well, with very limited shared or entire home listings.
# 
# 🏝️ Staten Island
# 
# - Mixed distribution, with a relatively balanced count between private rooms and entire homes (but overall very low volume).

# Shared rooms are minimal across all boroughs, showing low preference or demand post-pandemic.
# 
# Entire homes dominate in Manhattan, possibly due to higher income hosts or more tourist-centric neighborhoods.
# 
# Private rooms are more common in outer boroughs (Queens, Bronx, Brooklyn), likely due to residents renting spare space.

# ## 9. Are certain property/room types more expensive or popular than others?

# In[73]:


rooms_expense = df_cleaned.groupby('room_type').agg(
        avg_price = ('price','mean'),
        total_reviews=('number_of_reviews', 'sum'),
        listing_count=('listing_id','count')
).reset_index()
rooms_expense


# In[74]:


fig, ax1 = plt.subplots(figsize=(10,5))

# Avg Price per room type
sns.barplot(data=rooms_expense, x='room_type', y='avg_price', ax=ax1, palette='flare', label='Avg Price(USD)')
ax1.set_ylabel('Average Price (USD)', color='skyblue')
ax1.set_xlabel('Room Type')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.set_title('Room Types: Average Price & Popularity (Reviews)')

# Total Reviews per Room Type
ax2 = ax1.twinx()
sns.lineplot(data=rooms_expense, x='room_type', y='total_reviews', ax=ax2, color='orange', marker='o', linewidth=2, label='Total_Reviews')
ax2.set_ylabel('Total Reviews', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# # 🧭 Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

# Show both legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
plt.tight_layout()
plt.show()


# ### Observations

# 🏠 Entire Home/Apt:
# 
# - Highest average price (~$160).
# 
# - Most popular with the highest number of reviews (~200,000+).
# 
# - 🔥 This suggests travelers are willing to pay more for privacy and comfort.
# 
# 🚪 Private Room:
# 
# - Moderate average price (~$80).
# 
# - Second in popularity, still receiving a high number of reviews.
# 
# - 💡 Preferred by budget travelers seeking a balance between cost and privacy.
# 
# 🛏️ Shared Room:
# 
# - Lowest average price (~$70).
# 
# - Least popular with minimal reviews, indicating low demand.
# 
# - 🚫 Not favored much, possibly due to limited privacy or safety concerns.
# 
# 

# ## 10. Which room types are most reviewed in each neighborhood group per month?

# In[75]:


room_types = df_cleaned.groupby(['room_type','neighbourhood_group','month'])['reviews_per_month'].sum().reset_index()
room_types


# In[76]:


most_reviewed = room_types.sort_values(['month', 'reviews_per_month'], ascending=[True, False])
most_reviewed


# In[77]:


most_reviewed_top = most_reviewed.drop_duplicates(subset=['month']).copy()
most_reviewed_top


# In[78]:


# Display top reviewed room type per neighborhood group per month
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

most_reviewed_top['month'] = pd.Categorical(
    most_reviewed_top['month'], categories=month_order, ordered=True
)
most_reviewed_top = most_reviewed_top.sort_values('month')
most_reviewed_top


# In[79]:


most_reviewed_top['label'] = most_reviewed_top['neighbourhood_group'] + '\n(' + most_reviewed_top['room_type'] + ')'

# Plot
plt.figure(figsize=(14, 6))
sns.barplot(x='month', y='reviews_per_month', hue='label', data=most_reviewed_top, dodge=False)

plt.title('Top Reviewed Room Type per Month with Neighbourhood Group')
plt.xlabel('Month')
plt.ylabel('Reviews per Month')
plt.xticks(rotation=45)
plt.legend(title='Neighbourhood (Room Type)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# ### Observations

# - June (Manhattan, Entire home/apt) saw the highest reviews.
# 
# - July (Brooklyn, Entire home/apt) also experienced a significant peak.
# 
# - Brooklyn - Private room dominated most months except June, July, and December.
# 
# - In December, Manhattan - Entire home/apt regained the top spot.

# ## 11. Which features (e.g., room type, availability, location, reviews) are correlated with Airbnb prices?

# ### 📊 Numerical Features Correlated with Price

# In[80]:


numeric_cols = ['price', 'number_of_reviews', 'availability_365']

plt.figure(figsize=(8, 6))
sns.heatmap(df_cleaned[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation with Airbnb Price")
plt.show()


# ### Observations

# Correlation with Airbnb Price:
# 1. number_of_reviews = ~-0.0 i.e. No correlation. More or fewer reviews do not affect price directly.
# 2. availability_365 = ~+0.05 i.e. very weak positive. Slight tendency: higher availability might have slightly higher prices, but it's negligible.

# In[81]:


# Average price by room type
room_type_impact = df_cleaned.groupby('room_type').agg(
    avg_price=('price','mean')
).sort_values(by='avg_price',ascending=False)
room_type_impact


# In[82]:


# Average price by neighbourhood
location_impact = df_cleaned.groupby('neighbourhood_group').agg(
    avg_price=('price','mean')
).sort_values(by='avg_price',ascending=False)
location_impact


# ### Observations

# - room_type: Entire home/apt is most expensive, Shared room is cheapest
# - neighbourhood_group: Manhattan > Brooklyn > Staten Island > Queens/Bronx

# ### ✅ Final Observation:

# - No strong correlation is observed between these numerical features and price.
# 
# - This suggests that price is likely more influenced by categorical features like:
# 
#     - room_type (Entire home/apt vs Private room)
# 
#     - neighbourhood_group (Manhattan vs others)
# 
#     - property_type or location
# 
#     - Possibly amenities, if included

# ## 12. How do minimum night stays vary by neighborhood and room type?

# In[83]:


min_nights = df_cleaned.groupby(['neighbourhood_group','room_type'])['minimum_nights'].mean().reset_index()
min_nights


# In[84]:


plt.figure(figsize=(12, 6))
sns.barplot(
    data=min_nights,
    x='neighbourhood_group',
    y='minimum_nights',
    hue='room_type',
    palette='flare'
)
plt.title('Average Minimum Night Stays by Neighborhood and Room Type')
plt.ylabel('Average Minimum Nights')
plt.xlabel('Neighborhood Group')
plt.legend(title='Room Type')
plt.show()


# ### Observations

# 🏙️ Neighborhood-wise Patterns:
# 1. Brooklyn and Manhattan have the highest average minimum night requirements across room types.
# 
# 2. Staten Island has the lowest minimum nights for entire homes but relatively higher for shared rooms.
# 
# 🛏️ Room Type Trends:
# 1. Entire home/apt listings consistently require the longest stays in all boroughs.
# 
# 2. Private rooms usually fall in the mid-range, requiring fewer nights than entire homes.
# 
# 3. Shared rooms generally have the shortest minimum night stays—except in Staten Island, where it’s unexpectedly high.

# Queens offers more flexibility in minimum stays across room types.
# 
# Hosts in Manhattan and Brooklyn might prefer longer bookings, possibly to ensure higher revenue or reduce guest turnover.

# ## 13. Which neighborhoods attract longer or shorter stays on average?

# In[85]:


avg_stays = df_cleaned.groupby(['neighbourhood','room_type'])['minimum_nights'].mean().reset_index()
avg_stays


# In[86]:


# Longer stays
long_stays = avg_stays.sort_values(by='minimum_nights', ascending=False)
long_stays.head(20)


# In[87]:


# Shorter stays
short_stays = avg_stays.sort_values(by='minimum_nights', ascending=True)
short_stays.head(20)


# In[88]:


plt.figure(figsize=(15,5))
sns.barplot(data = long_stays.head(20), x='neighbourhood', y='minimum_nights', palette='magma')
plt.xticks(rotation=45, ha='right')
plt.title('Top 20 Neighborhoods with Longest Minimum Night Stays')
plt.ylabel('Avg Minimum Nights')
plt.xlabel('Neighborhood')
plt.tight_layout()
plt.show()


# ### Observations

# - Eltingville and Bensonhurst have the highest average minimum night stays — around 10 nights, indicating a preference or regulation for longer-term stays.
# 
# - Many of the top neighborhoods (e.g., Grant City, Riverdale, Sea Gate) are located in Staten Island and Brooklyn, which may have fewer tourist-focused short-term listings.
# 
# - These areas might cater more to:
# 
#     - Longer vacationers or families.
# 
#     - Temporary relocations.
# 
#     - Hosts preferring fewer guest turnovers.
# 
# - Listings in these neighborhoods may attract guests who plan extended stays, possibly due to suburban settings or housing type (e.g., entire homes).

# In[89]:


plt.figure(figsize=(15,5))
sns.barplot(data = short_stays.head(20), x='neighbourhood', y='minimum_nights', palette='crest')
plt.xticks(rotation=45, ha='right')
plt.title('Top 20 Neighborhoods with Shortest Minimum Night Stays')
plt.ylabel('Avg Minimum Nights')
plt.xlabel('Neighborhood')
plt.tight_layout()
plt.show()


# ### Observations

# - All top 20 neighborhoods for shortest stays have an average of 1 night — meaning many listings allow single-night bookings.
# - These neighborhoods are better suited for weekend getaways or overnight stays.

# ## 14. What’s the total number of reviews per neighborhood group?

# In[90]:


tot_reviews=df_cleaned.groupby('neighbourhood_group')['number_of_reviews'].sum().sort_values(ascending=False).reset_index().rename(columns={'number_of_reviews':'total_reviews'})
tot_reviews


# In[91]:


plt.figure(figsize=(10,5))
sns.barplot(data=tot_reviews, x='neighbourhood_group', y='total_reviews', palette='flare')
plt.title('Total Number of Reviews per Neighborhood Group')
plt.ylabel('Total Reviews')
plt.xlabel('Neighborhood Group')
plt.tight_layout()
plt.show()


# ### Observations

# 1. Brooklyn has the highest total number of reviews, indicating strong guest activity and listing popularity.
# 
# 2. Manhattan follows closely behind, reflecting its dense tourism and high listing volume.
# 
# 3. Queens shows moderate review activity, potentially due to its mix of residential and less tourist-heavy areas.
# 
# 4. Bronx has relatively low review counts, suggesting fewer listings or lower guest traffic.
# 
# 5. Staten Island has the least number of reviews, possibly due to limited accessibility or lower Airbnb presence.

# ## 15. Which listings have the maximum number of reviews by neighborhood?

# In[92]:


top_review_indices = df_cleaned.groupby('neighbourhood')['number_of_reviews'].idxmax()
top_review_indices


# In[93]:


top_reviewed_listings = df_cleaned.loc[top_review_indices].sort_values(by='number_of_reviews', ascending=False).reset_index(drop=True)
top_reviewed_listings


# In[94]:


top_reviewed_listings = top_reviewed_listings[['listing_id', 'neighbourhood', 'neighbourhood_group', 'listing_name','number_of_reviews','room_type']]
top_reviewed_listings.head(10)


# In[95]:


plt.figure(figsize=(14,6))
sns.barplot(data=top_reviewed_listings.head(10), 
            x='neighbourhood', 
            y='number_of_reviews',
            palette='flare'
           )
plt.title("Top Listings by Number of Reviews per Neighborhood")
plt.xticks(rotation=45)
plt.ylabel("Number of Reviews")
plt.xlabel("Neighborhood")
plt.tight_layout()
plt.show()


# ### Observations

# 1. Each neighborhood's top listing has 71 reviews, indicating a possible data cap, rounding, or filtering limit.
# 
# 2. The top-reviewed listings are spread across Brooklyn and Manhattan, showing high activity in these boroughs.
# 
# 3. Most of the top-reviewed listings are Private rooms, suggesting they are preferred or booked more frequently than entire apartments.
# 
# 4. Listings from popular Manhattan Areas like Upper East Side, Upper West Side, Chelsea, Chinatown, and Financial District are among the most reviewed.
# 
# 5. Listings from popular Brooklyn Areas like Flatbush, Brooklyn Heights, Bushwick, and Flatlands are also highly reviewed, indicating strong traveler interest.
# 
# 6. There’s no standout listing with drastically higher reviews than others — suggesting Airbnb popularity is relatively distributed across neighborhoods.

# ## 16. How does number of reviews correlate with availability and price?

# In[96]:


corr_data = df_cleaned[['number_of_reviews', 'availability_365', 'price']]

plt.figure(figsize=(8, 6))
sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation with Number Of Reviews")
plt.show()


# ### Observations

# | Feature Pair                          | Correlation Value         | Interpretation                                                                                                                                                                                                             |
# | ------------------------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **Number of Reviews vs Availability** | **0.25**                  | ✅ **Moderate Positive Correlation** — Listings available for more days in a year tend to receive more reviews. This makes intuitive sense since more availability offers more opportunities for bookings and thus reviews. |
# | **Number of Reviews vs Price**        | **\~0.00**                | 🚫 **No Correlation** — Price does not appear to influence the number of reviews. Higher or lower prices do not guarantee more or fewer reviews.                                                                           |
# | **Availability vs Price**             | **Low Positive (\~0.05)** | 🤏 **Negligible Correlation** — Slightly positive, but not significant enough to draw conclusions.                                                                                                                         |
# 

# ## 17. Which hosts own the most properties — are they professional operators?

# In[97]:


host_listing_counts = df_cleaned.groupby(['host_id', 'host_name'])['listing_id'].count().reset_index()

# Rename for clarity
host_listing_counts.columns = ['host_id', 'host_name', 'total_properties']

# Sort by most properties
top_hosts = host_listing_counts.sort_values(by='total_properties', ascending=False).head(10)

top_hosts


# ### Observations

# - All top hosts own exactly 3 properties.
# 
# - This does not strongly indicate professional operators, as they likely represent small-scale or semi-professional hosts.
# 
# - True professional operators often have 10+ properties and sometimes manage listings on behalf of others.

# ## 18. Can we recommend top 5 listings for a traveler based on reviews, price, and availability?

# In[103]:


traveler_df = df_cleaned.groupby(['listing_id', 'listing_name', 'host_name','neighbourhood_group'], as_index=False)[
    ['number_of_reviews', 'price', 'availability_365']
].mean()
traveler_df


# In[104]:


# Sort by:
# 1. Highest reviews
# 2. Lowest price
# 3. Highest availability
top_5 = traveler_df.sort_values(
    by=['number_of_reviews', 'availability_365', 'price'],
    ascending=[False, False, True]
).head(5)
top_5


# ### Observations

# - Queens dominates with budget-friendly, highly available listings.
# 
# - All listings have 71 reviews — indicating strong traveler trust.
# 
# - Prices are reasonable considering NYC standards, especially under $100.
# 
# - Great for solo travelers or short stays.

# # Summary Analysis

# ### 🏙️ Market Growth & Trends
# - Listings Growth: Number of Airbnb listings grew steadily from 2011 to 2019, with a sharp rise in 2019.
# 
# - Review Trends: Total reviews spiked alongside listings, showing increasing traveler engagement.
# 
# - Price Trends: Average prices remained mostly stable, with slight seasonal variations.
# 
# ### 🗓️ Seasonality
# - Busiest Months: June, July, and May had the highest number of reviews, indicating peak travel seasons.
# 
# - Quiet Months: February and January had the lowest activity.
# 
# ### 🌍 Top Locations
# - Top for Travelers: Neighborhoods like Bedford-Stuyvesant, Williamsburg, and Harlem had high availability, review volume, and moderate prices.
# 
# - High Occupancy: Areas like Castle Hill and Navy Yard showed strong occupancy potential.
# 
# - Low Occupancy: Neighborhoods such as Eastchester and Eltingville underperformed.
# 
# ### 💸 Price vs Features
# - Room Type Impact: Entire homes/apartments were the most expensive and most reviewed.
# 
# - Availability: Listings with higher availability tended to get more reviews, though correlation with price was weak.
# 
# - Outliers: Some listings were identified as overpriced based on their neighborhood averages.
# 
# ### 🏡 Room & Property Analysis
# - Most Common Room Types: Private rooms dominated in most boroughs, especially in Manhattan and Brooklyn.
# 
# - Minimum Night Stays: Manhattan and Brooklyn listings had higher average minimum night requirements. Staten Island had the lowest.
# 
# ### 🧑‍💼 Host Insights
# - Most hosts managed only 1–2 listings, but a few were professional operators with multiple properties.
# 
# ### 🌟 Top Recommendations
# - Based on number of reviews, price, and availability, we recommended top listings that offer the best value and experience to travelers.

# In[ ]:




