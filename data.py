import pandas as pd



customers_df= pd.read_csv('data/olist_customers_dataset.csv')
geolocation_df= pd.read_csv('data/olist_geolocation_dataset.csv')
items_df= pd.read_csv('data/olist_order_items_dataset.csv')
payments_df= pd.read_csv('data/olist_order_payments_dataset.csv')
reviews_df= pd.read_csv('data/olist_order_reviews_dataset.csv')
orders_df= pd.read_csv('data/olist_orders_dataset.csv')
products_df= pd.read_csv('data/olist_products_dataset.csv')
sellers_df= pd.read_csv('data/olist_sellers_dataset.csv')
category_translation_df= pd.read_csv('data/product_category_name_translation.csv')


df= pd.merge(customers_df, orders_df, on="customer_id", how='inner')
df= df.merge(reviews_df, on="order_id", how='inner')
df= df.merge(items_df, on="order_id", how='inner')
df= df.merge(products_df, on="product_id", how='inner')
df= df.merge(payments_df, on="order_id", how='inner')
df= df.merge(sellers_df, on='seller_id', how='inner')
df= df.merge(category_translation_df, on='product_category_name', how='inner')

df.to_csv('olist_customers_dataset.csv', index=False)