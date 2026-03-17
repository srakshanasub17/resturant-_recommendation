import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
df = pd.read_csv(r"C:\Users\Hxtreme\Downloads\Dataset .csv", encoding="utf-8-sig")
print("Rows loaded:", len(df))
print("Columns:", df.columns.tolist())
# -------------------------------------------------
# Preprocess Dataset
# -------------------------------------------------
# Clean column names
df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)

# Handle missing values (drop rows with missing critical info)
df = df.dropna(subset=['Restaurant Name', 'Cuisines', 'Average Cost for two', 'Price range'])
print("Rows after cleaning:", len(df))
print("Number of unique cuisines:", df['Cuisines'].nunique())

sample_cuisines = df['Cuisines'].unique()[:10]
print("Sample cuisines:")
for c in sample_cuisines:
    print(" -", c)

print("\nSample restaurants with cuisines:")
print(df[['Restaurant Name', 'Cuisines']].head(5))
# Split multi-cuisine strings into separate rows
df['Cuisines'] = df['Cuisines'].str.split(', ')
df = df.explode('Cuisines').reset_index(drop=True)
# Encode cuisines
encoder = OneHotEncoder(handle_unknown='ignore')
cuisine_encoded = encoder.fit_transform(df[['Cuisines']]).toarray()# Use automatic feature names (no mismatch error)
cuisine_df = pd.DataFrame(
    cuisine_encoded,
    columns=encoder.get_feature_names_out())

# Combine encoded cuisines with numeric features
features = pd.concat([
    df[['Average Cost for two', 'Price range']],
    cuisine_df
], axis=1)

# -------------------------------------------------
# Recommendation Function
# -------------------------------------------------
def recommend_restaurants(user_pref, top_n=5):
    """
    Recommend restaurants based on user preferences using content-based filtering.
    user_pref: dict with keys 'Cuisines', 'Price range', 'Average Cost for two'
    """
    # Encode user cuisine preference
    user_cuisine_encoded = encoder.transform([[user_pref['Cuisines']]]).toarray()
    user_features = pd.DataFrame(
        user_cuisine_encoded,
        columns=encoder.get_feature_names_out()
    )
    
    # Add numeric preferences
    user_features['Average Cost for two'] = user_pref['Average Cost for two']
    user_features['Price range'] = user_pref['Price range']
    
    # Compute similarity
    similarity_scores = cosine_similarity(features, user_features)
    
    # Attach similarity scores to dataframe
    df['Similarity'] = similarity_scores.flatten()
    
    # Get top N recommendations
    recommendations = df.sort_values(by='Similarity', ascending=False).head(top_n)
    
    return recommendations
