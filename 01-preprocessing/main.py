import os
import pandas as pd
import numpy as np
from django.contrib.admin import display
from sklearn.preprocessing import LabelEncoder

# Load and read all dataset
is_in_kaggle = True if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") else False
print("Is running inside kaggle:", is_in_kaggle)
data_path = "/kaggle/input/brazilian-ecommerce" if is_in_kaggle else "../datasets"

files = []
for dirname, _, filenames in os.walk(data_path):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))

customers_dataset = pd.read_csv(f'{data_path}/olist_customers_dataset.csv')
geolocation_dataset = pd.read_csv(f'{data_path}/olist_geolocation_dataset.csv')
order_items_dataset = pd.read_csv(f'{data_path}/olist_order_items_dataset.csv')
order_payments_dataset = pd.read_csv(f'{data_path}/olist_order_payments_dataset.csv')
order_reviews_dataset = pd.read_csv(f'{data_path}/olist_order_reviews_dataset.csv')
orders_dataset = pd.read_csv(f'{data_path}/olist_orders_dataset.csv')
products_dataset = pd.read_csv(f'{data_path}/olist_products_dataset.csv')
sellers_dataset = pd.read_csv(f'{data_path}/olist_sellers_dataset.csv')

# Check data sanity (entries and columns)
print("Customers Dataset: ", customers_dataset.shape)
print("Geolocation Dataset: ", geolocation_dataset.shape)
print("Order Items Dataset: ", order_items_dataset.shape)
print("Order Payments Dataset: ", order_payments_dataset.shape)
print("Order Reviews Dataset: ", order_reviews_dataset.shape)
print("Orders Dataset: ", orders_dataset.shape)
print("Products Dataset: ", products_dataset.shape)
print("Sellers Dataset: ", sellers_dataset.shape)

def check_info(title, data):
    print(title)
    display(data.head())
    print(data.info())
    display(data.describe(include='all'))
    print(f"Missing values count: {data.isnull().sum().sum()}\n")

check_info("Customer dataset:", customers_dataset)
check_info("Order items dataset:", order_items_dataset)
check_info("Order payments dataset:", order_payments_dataset)
check_info("Orders dataset:", orders_dataset)

# check null values
print("Column name | Count of missing values | Total row | Percentage")
null = orders_dataset.isnull().sum()[orders_dataset.isnull().sum() > 0]
print(np.array(
    [null.index, null.values, [orders_dataset.shape[0]] * len(null), null.values / orders_dataset.shape[0] * 100]).T)
# check the data
display(orders_dataset[orders_dataset.isnull().any(axis=1)].head())
# check order status of missing values
print(orders_dataset[orders_dataset.isnull().any(axis=1)]['order_status'].value_counts())

print("total entry before delete: ", orders_dataset.shape[0])
orders_dataset = orders_dataset.dropna(subset=['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date'])
print("total row after delete: ", orders_dataset.shape[0])
display(orders_dataset.info())
display(orders_dataset.isnull().sum())

# convert datetime string on orders_dataset dataset to datetime type
orders_dataset['order_purchase_timestamp'] = pd.to_datetime(orders_dataset['order_purchase_timestamp'])
orders_dataset['order_approved_at'] = pd.to_datetime(orders_dataset['order_approved_at'])
orders_dataset['order_delivered_carrier_date'] = pd.to_datetime(orders_dataset['order_delivered_carrier_date'])
orders_dataset['order_delivered_customer_date'] = pd.to_datetime(orders_dataset['order_delivered_customer_date'])
orders_dataset['order_estimated_delivery_date'] = pd.to_datetime(orders_dataset['order_estimated_delivery_date'])
display(orders_dataset.head())
display(orders_dataset.info())

# convert datetime string on order_item dataset to datetime type
order_items_dataset['shipping_limit_date'] = pd.to_datetime(order_items_dataset['shipping_limit_date'])
print(order_items_dataset.info())

# convert order_status on order dataset to category type
orders_dataset['order_status'] = orders_dataset['order_status'].astype('category')
orders_dataset['order_status'].value_counts()

# standardize order status using label encoding

le = LabelEncoder()
orders_dataset['order_status_encoded'] = le.fit_transform(orders_dataset['order_status'])

status_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"Order status mapping: {status_mapping}")

# standardize payment type using one hot encoding
payment_dummies = pd.get_dummies(order_payments_dataset['payment_type'], prefix='payment')
payments_encoded = pd.concat([order_payments_dataset, payment_dummies], axis=1)
# payments.drop(columns='payment_type', inplace=True)
display(payments_encoded.head())

# export the datasets to csv
orders_dataset.to_pickle('orders_dataset.pkl')
order_items_dataset.to_pickle('order_items_dataset.pkl')
status_mapping_df = pd.DataFrame(list(status_mapping.items()), columns=['order_status', 'order_status_encoded'])
status_mapping_df.to_pickle('order_status_mapping.pkl')
payments_encoded.to_pickle('payments_encoded_dataset.pkl')
# also convert unprocessed dataset to pkl
customers_dataset.to_pickle('customers_dataset.pkl')
geolocation_dataset.to_pickle('geolocation_dataset.pkl')
order_payments_dataset.to_pickle('order_payments_dataset.pkl')
order_reviews_dataset.to_pickle('order_reviews_dataset.pkl')
products_dataset.to_pickle('products_dataset.pkl')
sellers_dataset.to_pickle('sellers_dataset.pkl')


#Merging all relevant datasets
merged_ds = pd.merge(customers_dataset, orders_dataset, on='customer_id', how='inner')
merged_ds = pd.merge(merged_ds, order_items_dataset, on='order_id', how='inner')
merged_ds = pd.merge(merged_ds, order_payments_dataset, on='order_id', how='inner')

# Shape 
print(merged_ds.shape)

# Info
print(merged_ds.info())

# Convert datetime columns to datetime type
merged_ds['order_purchase_timestamp'] = pd.to_datetime(merged_ds['order_purchase_timestamp'])

# filter out rows that are not delivered (incomplete)
merged_ds = merged_ds[merged_ds['order_status'] == 'delivered'].copy()

# Count missing values
print("Missing values count: ")
print(merged_ds.isnull().sum())

# Drop missing values
merged_ds = merged_ds.dropna()

# Count duplicate values
print("Duplicate values count: ", merged_ds.duplicated().sum())

merged_ds.to_pickle('merged_dataset.pkl')