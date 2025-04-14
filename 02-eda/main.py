import pandas as pd
import seaborn as sns
import numpy as np
from django.contrib.admin import display
import matplotlib.pyplot as plt
import unidecode
import re
import urllib
from matplotlib import image as mpimg
import folium
from folium.plugins import HeatMap
import pickle

# import dataset from preprocessing
with open('../01-preprocessing/customers_dataset.pkl', 'rb') as file:
    customers_dataset = pickle.load(file)
with open('../01-preprocessing/geolocation_dataset.pkl', 'rb') as file:
    geolocation_dataset = pickle.load(file)
with open('../01-preprocessing/orders_dataset.pkl', 'rb') as file:
    orders_dataset = pickle.load(file)
with open('../01-preprocessing/order_items_dataset.pkl', 'rb') as file:
    order_items_dataset = pickle.load(file)
with open('../01-preprocessing/order_payments_dataset.pkl', 'rb') as file:
    order_payments_dataset = pickle.load(file)

# top 20 customer cities

print("null on customer city: ", customers_dataset['customer_city'].isna().mean())

# karena kota di brazil kemungkinan ada huruf dengan aksen dan ada kemungkinan duplikat dengan non-aksen, periksa terlebih dahulu 
def has_accented_characters(s):
    return bool(re.search(r'[^\x00-\x7F]', s))

print("accent on customer city: ", customers_dataset['customer_city'].apply(has_accented_characters).sum())

customer_city_counts = customers_dataset['customer_city'].value_counts().reset_index()
customer_city_counts.columns = ['City', 'Count']

top_20_cities = customer_city_counts.head(20)
# Plot using bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x='Count', y='City', data=top_20_cities)
plt.title('Top 20 customer cities')
plt.xlabel('Number of Customers')
plt.ylabel('City')
plt.tight_layout()
# plt.show()
plt.savefig('top_20_cities.png', dpi=300, bbox_inches='tight')

# heatmap customer
print("null in geolocation dataset: \n", geolocation_dataset.isna().mean())
print("unique val in geoloc: ", geolocation_dataset['geolocation_city'].unique())
def has_accented_characters(s):
    return bool(re.search(r'[^\x00-\x7F]', s))

print("accent on geolocation city: ", geolocation_dataset['geolocation_city'].apply(has_accented_characters).sum())
# karena ada kota dengan huruf aksen dan non-aksen duplikat, perlu dilakukan preprocessing terlebih dahulu
def pretty_string(column):
    column_space = ' '.join(column.split())
    return unidecode.unidecode(column_space.lower())
geolocation_dataset['geolocation_city'] = geolocation_dataset['geolocation_city'].apply(pretty_string)
print("unique val in geoloc: ", geolocation_dataset['geolocation_city'].unique())
print("accent on geolocation city: ", geolocation_dataset['geolocation_city'].apply(has_accented_characters).sum())

# heatmap customer lanjut part 2

# cek zipcode
print(geolocation_dataset.groupby('geolocation_zip_code_prefix').size().sort_values(ascending=False))
# karena zipcode pada data di-hide sehingga hanya prefix nya saja yang dimunculkan, ada kemungkinan satu zipcode prefix memiliki beberapa koordinat, sesuai dengan zipcode lengkapnya
print(geolocation_dataset[geolocation_dataset['geolocation_zip_code_prefix'] == 24220].head())
# oleh karena itu perlu dilakukan data processing dengan mengambil titik tengah dari masing2 zipcode prefix
state_unique_geolocation = geolocation_dataset.groupby(['geolocation_zip_code_prefix'])['geolocation_state'].nunique().reset_index(name='count')
print(state_unique_geolocation[state_unique_geolocation['count']>= 2].shape)
max_state = geolocation_dataset.groupby(['geolocation_zip_code_prefix','geolocation_state']).size().reset_index(name='count').drop_duplicates(subset = 'geolocation_zip_code_prefix').drop('count',axis=1)
geolocation_coords = geolocation_dataset.groupby(['geolocation_zip_code_prefix','geolocation_city','geolocation_state'])[['geolocation_lat','geolocation_lng']].median().reset_index()
geolocation_coords = geolocation_coords.merge(max_state,on=['geolocation_zip_code_prefix','geolocation_state'],how='inner')
customers_coords = customers_dataset.merge(geolocation_coords,left_on='customer_zip_code_prefix',right_on='geolocation_zip_code_prefix',how='inner')
# mari mapping 
brazil = mpimg.imread(urllib.request.urlopen('https://i.pinimg.com/originals/3a/0c/e1/3a0ce18b3c842748c255bc0aa445ad41.jpg'),'jpg')
ax = customers_coords.drop_duplicates(subset='customer_unique_id').plot(kind="scatter", x="geolocation_lng", y="geolocation_lat", figsize=(10,10), alpha=0.3,s=0.3,c='red')
plt.axis('off')
plt.imshow(brazil, extent=[-73.98283055, -33.8,-33.75116944,5.4])
# plt.show()
plt.savefig('customer_heatmap.png', dpi=300, bbox_inches='tight')

# alternatif heatmap yang sinan pengen

customer_counts = customers_coords.groupby(['geolocation_lat', 'geolocation_lng']).size().reset_index(name='customer_count')

# Create a map centered on Brazil
brazil_map = folium.Map(location=[-15.77972, -47.92972], zoom_start=4)

# Add heatmap layer
heat_data = [[row['geolocation_lat'], row['geolocation_lng'], row['customer_count']]
             for index, row in customer_counts.iterrows()]
HeatMap(heat_data).add_to(brazil_map)
# display(brazil_map)
brazil_map.save('customer_heatmap_v2.html')

# average expense of a customer
orders_with_customers = pd.merge(orders_dataset, customers_dataset, on='customer_id')
orders_with_items = pd.merge(orders_with_customers, order_items_dataset, on='order_id')
print(orders_with_items.head())

# total expense per customer (unique_id)
orders_with_items['total_value'] = orders_with_items['price'] + orders_with_items['freight_value']
customer_expenses = orders_with_items.groupby('customer_unique_id')['total_value'].sum().reset_index()
customer_expenses.columns = ['Customer ID', 'Total Expense']
print(customer_expenses.head())

# average customer expense
avg_expense = customer_expenses['Total Expense'].mean()
print(f"Average customer expense: R$ {avg_expense:.2f}")

# some additional statistics
perc_99 = np.percentile(customer_expenses['Total Expense'], 99)
print(f"Median customer expense: R$ {customer_expenses['Total Expense'].median():.2f}")
print(f"Maximum customer expense: R$ {customer_expenses['Total Expense'].max():.2f}")
print(f"99th percentile expense: R$ {perc_99:.2f}")

# histogram
# karena adanya outlier yang membuat histogram sebelumnya terlihat terlalu lebar, maka range histogram akan di set ke max 99th percentile saja
plt.figure(figsize=(10, 6))
plt.hist(customer_expenses['Total Expense'], bins=50, range=(0, perc_99))
plt.axvline(avg_expense, color='red', linestyle='dashed', linewidth=1)
plt.text(avg_expense*1.1, plt.ylim()[1]*0.9, f'Mean: R$ {avg_expense:.2f}')
plt.title('Distribution of Customer Expenses')
plt.xlabel('Total Expense (R$)')
plt.ylabel('Number of Customers')
plt.xlim(0, perc_99)
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()
plt.savefig('customer_expenses_histogram.png', dpi=300, bbox_inches='tight')

# average order in a year
orders_dataset['year'] = orders_dataset['order_purchase_timestamp'].dt.year
orders_dataset['month'] = orders_dataset['order_purchase_timestamp'].dt.month

monthly_orders = orders_dataset.groupby(['year', 'month']).size().reset_index(name='order_count')
monthly_orders['date'] = pd.to_datetime(monthly_orders['year'].astype(str) + '-' + monthly_orders['month'].astype(str) + '-01')
monthly_orders = monthly_orders.sort_values('date')

yearly_orders = orders_dataset.groupby(['year']).size().reset_index(name='order_count')
print("Order count per year")
print(yearly_orders)

avg_monthly_orders = monthly_orders['order_count'].mean()
print(f"Average monthly orders: {avg_monthly_orders:.2f}")
avg_yearly_orders = yearly_orders['order_count'].mean()
print(f"Average yearly orders: {avg_yearly_orders:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(monthly_orders['date'], monthly_orders['order_count'])
plt.axhline(avg_monthly_orders, color='red', linestyle='dashed')
plt.title('Number of Orders per Month')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()
plt.savefig('order_monthly.png', dpi=300, bbox_inches='tight')

payment_counts = order_payments_dataset['payment_type'].value_counts()
print(payment_counts)

payment_percentages = (payment_counts / payment_counts.sum() * 100).reset_index()
payment_percentages.columns = ['Payment Method', 'Percentage']

plt.figure(figsize=(10, 6))
plt.pie(payment_percentages['Percentage'], labels=payment_percentages['Payment Method'],
        autopct='%1.1f%%', startangle=90, shadow=True)
plt.axis('equal')
plt.title('Payment Methods Distribution')
plt.tight_layout()
# plt.show()
plt.savefig('payment_methods_distribution.png', dpi=300, bbox_inches='tight')

print(payment_percentages)