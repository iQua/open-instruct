#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

domain_names = {
    'github': 'GitHub',
    'arxiv': 'arXiv',
    'c4': 'C4',
    'book': 'Books',
    'wikipedia': 'Wikipedia',
    'tulu': "Tulu-v3.1",
}

def load_data(input_path):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_data_for_model(model):
    df = []
    for domain in domain_names.keys():
        input_path = f"{model}/expert_counts/{domain}.pkl"
        layer0, layer5, layer11 = load_data(input_path)
        counters = {
            0: layer0,
            5: layer5,
            11: layer11,
        }
        for i in [0, 5, 11]:
            total_count = sum(counters[i].values())
            for j, v in counters[i].items():
                df.append({
                    'Domain': domain_names[domain],
                    'Layer': i,
                    'Expert': j,
                    'Proportion': v / total_count * 100,
                })
    df = pd.DataFrame(df)
    return df


# In[2]:
df = load_data_for_model('mistral')
if df is None or df.empty:
    raise ValueError("Failed to load data for Mistral or DataFrame is empty.")

# Select a specific domain for the heatmap (e.g., 'Tulu-v3.1')
domain = 'GitHub'
df_domain = df[df['Domain'] == domain]

pivot_df = df_domain.pivot(index='Layer', columns='Expert', values='Proportion')

# Create the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_df, cmap='viridis', annot=False, vmin=0, vmax=100)  # Proportion is 0-100
plt.title(f'Expert Usage on {domain} for Mistral')
plt.xlabel('Expert ID')
plt.ylabel('Layer')
# Save as both PDF and JPG
plt.savefig(f'mistral_heatmap_{domain.lower().replace(" ", "_")}.pdf', bbox_inches='tight')
plt.savefig(f'mistral_heatmap_{domain.lower().replace(" ", "_")}.jpg', bbox_inches='tight')
plt.close()

