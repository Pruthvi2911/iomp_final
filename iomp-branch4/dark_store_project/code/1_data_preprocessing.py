"""
Dark Store Project - Data Preprocessing
Step 1: Load Instacart data and prepare for ILP + PPO
"""

import pandas as pd
import pickle
from collections import Counter
from itertools import combinations
from pathlib import Path
import os

print("=" * 60)
print("DARK STORE DATA PREPROCESSING")
print("=" * 60)

# ============================================
# CONFIGURATION
# ============================================
BASE_DIR = Path(__file__).resolve().parents[1]
# Based on your folder structure, we ensure DATA_PATH points to D:\IOMP\dataset
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = WORKSPACE_ROOT / "dataset"
OUTPUT_PATH = BASE_DIR / "data"

# Create output directory if it doesn't exist to prevent errors
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ============================================
# STEP 1: Load Instacart Data
# ============================================
print(f"\n[1/5] Loading Instacart data from: {DATA_PATH}")

try:
    # Changed '+' to '/' for Path objects
    orders = pd.read_csv(DATA_PATH / "orders.csv", low_memory=False)
    products = pd.read_csv(DATA_PATH / "products.csv", low_memory=False)
    order_products = pd.read_csv(DATA_PATH / "order_products__train.csv", low_memory=False)
    
    print(f"✓ Loaded {len(orders):,} orders")
    print(f"✓ Loaded {len(products):,} products")
    print(f"✓ Loaded {len(order_products):,} order-product pairs")
except FileNotFoundError as e:
    print(f"❌ ERROR: Could not find data files!")
    print(f"   Target path: {DATA_PATH}")
    print("   Make sure these files exist in that folder:")
    print("   - orders.csv")
    print("   - products.csv")
    print("   - order_products__train.csv")
    exit()

# ============================================
# STEP 2: Calculate Pick Frequency
# ============================================
print("\n[2/5] Calculating pick frequency for each product...")

product_frequency = order_products['product_id'].value_counts()

product_freq_df = pd.DataFrame({
    'product_id': product_frequency.index,
    'frequency': product_frequency.values
})

product_freq_df = product_freq_df.merge(
    products[['product_id', 'product_name']], 
    on='product_id'
)

product_freq_df = product_freq_df.sort_values('frequency', ascending=False)

print(f"✓ Calculated frequency for {len(product_freq_df):,} products")
print(f"\nTop 5 most ordered items:")
for i, row in product_freq_df.head(5).iterrows():
    print(f"  {row['product_name']}: {row['frequency']:,} orders")

# ============================================
# STEP 3: Select Top 500 SKUs
# ============================================
print("\n[3/5] Selecting top 500 SKUs...")

top_500 = product_freq_df.head(500).copy()
top_500 = top_500.reset_index(drop=True)

print(f"✓ Selected top 500 products")
print(f"  Range: {top_500['frequency'].max():,} to {top_500['frequency'].min():,} orders")

# ============================================
# STEP 4: ABC Classification
# ============================================
print("\n[4/5] Performing ABC classification...")

top_500['abc_class'] = 'C'
top_500.loc[:99, 'abc_class'] = 'A'      # Top 100
top_500.loc[100:249, 'abc_class'] = 'B'  # Next 150

abc_counts = top_500['abc_class'].value_counts().sort_index()
print(f"✓ ABC Classification complete:")
print(f"  A-items (High freq): {abc_counts.get('A', 0)} products")
print(f"  B-items (Medium):    {abc_counts.get('B', 0)} products")
print(f"  C-items (Low freq):  {abc_counts.get('C', 0)} products")

# Changed '+' to '/'
abc_output = OUTPUT_PATH / "abc_classification.csv"
top_500.to_csv(abc_output, index=False)
print(f"✓ Saved to: {abc_output}")

# ============================================
# STEP 5: Find Top 50 Affinity Pairs
# ============================================
print("\n[5/5] Finding top 50 co-purchased item pairs...")

top_500_ids = set(top_500['product_id'].values)
relevant_orders = order_products[
    order_products['product_id'].isin(top_500_ids)
]

orders_grouped = relevant_orders.groupby('order_id')['product_id'].apply(list)

print("  Analyzing co-purchase patterns...")
pair_counter = Counter()
#for pair in combinations(sorted(order_items),2):pair_counter[pair]+=1 is Affinity Calculation

for order_items in orders_grouped:
    if len(order_items) >= 2:
        for pair in combinations(sorted(order_items), 2):
            pair_counter[pair] += 1

top_50_pairs = pair_counter.most_common(50)

print(f"✓ Found {len(pair_counter):,} unique item pairs")
print(f"✓ Selected top 50 pairs")

affinity_data = []
for (prod1, prod2), count in top_50_pairs:
    name1 = products[products['product_id'] == prod1]['product_name'].values[0]
    name2 = products[products['product_id'] == prod2]['product_name'].values[0]
    
    affinity_data.append({
        'product_1_id': prod1,
        'product_1_name': name1,
        'product_2_id': prod2,
        'product_2_name': name2,
        'co_purchase_count': count
    })

affinity_df = pd.DataFrame(affinity_data)

# Changed '+' to '/'
affinity_output = OUTPUT_PATH / "affinity_pairs.csv"
affinity_df.to_csv(affinity_output, index=False)
print(f"✓ Saved to: {affinity_output}")

# ============================================
# STEP 6: Save Processed Data
# ============================================
print("\n[6/6] Saving processed data for ILP and PPO...")

processed_data = {
    'products': top_500,
    'affinity_pairs': affinity_df,
    'metadata': {
        'total_products': 500,
        'a_items': 100,
        'b_items': 150,
        'c_items': 250,
        'affinity_pairs': 50
    }
}

# Changed '+' to '/'
pickle_output = OUTPUT_PATH / "processed_data.pkl"
with open(pickle_output, 'wb') as f:
    pickle.dump(processed_data, f)

print(f"✓ Saved to: {pickle_output}")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE! ✓")
print("=" * 60)