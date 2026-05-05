# data/generate_extended_data.py
import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
np.random.seed(42)

def generate_extended_dataset(n=5000):
    categories = ['Electronics', 'Sport', 'Home', 'Fashion', 'Beauty', 'Books', 'Toys', 'Automotive']
    subcategories = {
        'Electronics': ['Smartphones', 'Laptops', 'Audio', 'Wearables'],
        'Sport': ['Fitness', 'Outdoor', 'Team Sports', 'Yoga'],
        'Home': ['Kitchen', 'Furniture', 'Lighting', 'Decor'],
        'Fashion': ['Men', 'Women', 'Kids', 'Accessories'],
        'Beauty': ['Skincare', 'Makeup', 'Hair', 'Fragrance'],
        'Books': ['Fiction', 'Non-Fiction', 'Education', 'Comics'],
        'Toys': ['Action Figures', 'Puzzles', 'Educational', 'Outdoor'],
        'Automotive': ['Parts', 'Accessories', 'Tools', 'Electronics']
    }
    
    data = []
    for i in range(1, n+1):
        cat = random.choice(categories)
        sub = random.choice(subcategories[cat])
        price = np.round(np.random.lognormal(mean=np.log(40), sigma=0.8), 2)
        promo = random.choice([True, False])
        price_promo = round(price * random.uniform(0.5, 0.9), 2) if promo else price
        rating = np.clip(np.random.normal(4.2, 0.6), 1, 5)
        nb_reviews = int(np.random.lognormal(mean=5, sigma=1.2))
        stock = random.randint(0, 500)
        delivery = random.choice([1,2,3,5,7])
        shop = fake.company()
        country = fake.country()
        date_added = fake.date_between(start_date='-2y', end_date='today')
        
        data.append({
            'product_id': i,
            'name': fake.catch_phrase(),
            'description': fake.text(max_nb_chars=200),
            'category': cat,
            'subcategory': sub,
            'brand': fake.company(),
            'price': price,
            'price_promo': price_promo,
            'discount_pct': round((price - price_promo)/price*100, 1),
            'currency': 'EUR',
            'rating': round(rating, 1),
            'reviews': nb_reviews,
            'stock': stock,
            'delivery_days': delivery,
            'shop': shop,
            'country': country,
            'product_url': fake.url(),
            'date_scraped': fake.date_time_this_year()
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/extended_products.csv', index=False)
    print(f"Dataset étendu généré : {len(df)} produits.")
    return df

if __name__ == '__main__':
    generate_extended_dataset(5000)