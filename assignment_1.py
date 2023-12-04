#Let's import all the necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict
from termcolor import colored
from IPython.display import display, HTML
import warnings
warnings.filterwarnings("ignore")

# Style for dataframes to look better in the notebook

def style_dataframe(df):
    df_html = df.to_html(index=False)

    css = """
    <style type=\"text/css\">
    table {
        color: #333; /* Lighten up font color */
        font-family: Helvetica, Arial, sans-serif;
        width: 640px;
        border-collapse:
        collapse; 
        border-spacing: 0;
        margin: 20px auto;
    }

    td, th {
        border: 1px solid transparent; /* No more visible border */
        height: 30px;
        transition: all 0.3s;  /* Simple transition for hover effect */
    }

    th {
        background: #DFDFDF;  /* Darken header a bit */
        font-weight: bold;
    }

    td {
        background: #FAFAFA;
        text-align: center;
    }

    /* Cells in even rows (2,4,6...) are one color */
    tr:nth-child(even) td { background: #F1F1F1; }

    /* Cells in odd rows (1,3,5...) are another (excludes header cells)  */
    tr:nth-child(odd) td { background: #FEFEFE; }

    tr td:hover { background: #666; color: #FFF; } /* Hover cell effect! */
    </style>
    """

    # Return the styled HTML
    return HTML(css + df_html)


# Load datasets
animes = pd.read_csv('animes.csv')
user_scores_train = pd.read_csv('user_rates_train.csv')
user_scores_test = pd.read_csv('user_rates_test.csv')

# Merging data
anime_fulldata_train = pd.merge(animes, user_scores_train, on='anime_id', suffixes= ['', '_user'])
anime_fulldata_test = pd.merge(animes, user_scores_test, on='anime_id', suffixes= ['', '_user'])

# Renaming columns
anime_fulldata_train = anime_fulldata_train.rename(columns={'Name': 'anime_title', 'rating': 'user_rating'})
anime_fulldata_test = anime_fulldata_test.rename(columns={'Name': 'anime_title', 'rating': 'user_rating'})

# Drop unnecessary columns
anime_fulldata_train = anime_fulldata_train.drop(['Anime Title'], axis=1)
anime_fulldata_test = anime_fulldata_test.drop(['Anime Title'], axis=1)

# Combine train and test data for visualization
anime_fulldata = pd.concat([anime_fulldata_train, anime_fulldata_test], axis=0, ignore_index=True)

for i in anime_fulldata_train.isnull().sum() + anime_fulldata_test.isnull().sum():
    if(i>0):
        print(colored("There are null values in the dataset","red"))
        break
    else:
        print(colored("There are no null values in the dataset","green"))
        break


#Preprocessing the data

def convert_to_minutes(duration_str):
    minutes = -1  # Default value for unparseable strings or unknowns
    # Check if duration is in hours and minutes
    if 'hr' in duration_str and 'min' in duration_str:
        time = re.findall(r'(\d+)\s*hr\s*(\d+)\s*min', duration_str)
        minutes = int(time[0][0]) * 60 + int(time[0][1])
    # Check if duration is in hours
    elif 'hr' in duration_str:
        time = re.findall(r'(\d+)\s*hr', duration_str)
        minutes = int(time[0]) * 60
    # Check if duration is in minutes
    elif 'min' in duration_str:
        time = re.findall(r'(\d+)\s*min', duration_str)
        minutes = int(time[0])
    # Check if duration is in seconds
    elif 'sec' in duration_str:
        time = re.findall(r'(\d+)\s*sec', duration_str)
        # Convert seconds to minutes;
        minutes = int(time[0]) / 60.0  
    # For 'Unknown' or other unhandled cases, we keep the default value of -1

    return minutes

def process_combined_one_hot_encoding(train_df, test_df, columns):
    combined_df = pd.concat([train_df, test_df], axis=0)
    
    for column in columns:
        dummies = pd.get_dummies(combined_df[column], prefix=column)
        combined_df = pd.concat([combined_df, dummies], axis=1)
        combined_df = combined_df.drop(column, axis=1)
        combined_df[dummies.columns] = combined_df[dummies.columns].astype(int)

    # Splitting the datasets back to train and test
    train_df = combined_df.iloc[:len(train_df)]
    test_df = combined_df.iloc[len(train_df):]
    
    return train_df, test_df


def process_combined_features(train_df, test_df, feature_name):
    combined_df = pd.concat([train_df, test_df], axis=0)
    combined_df[feature_name] = combined_df[feature_name].apply(lambda x: x.split(', ') if x != 'UNKNOWN' else [])
    
    mlb = MultiLabelBinarizer()
    binary_data = pd.DataFrame(mlb.fit_transform(combined_df[feature_name]), columns=mlb.classes_, index=combined_df.index)
    
    # Splitting the datasets back to train and test
    train_binary_data = binary_data.iloc[:len(train_df)]
    test_binary_data = binary_data.iloc[len(train_df):]
    
    train_df = train_df.drop(feature_name, axis=1)
    test_df = test_df.drop(feature_name, axis=1)

    train_df = pd.concat([train_df, train_binary_data], axis=1)
    test_df = pd.concat([test_df, test_binary_data], axis=1)

    return train_df, test_df

# Data preprocessing pipeline
def data_preprocessing_pipeline(train_df, test_df):
    # Convert duration to minutes
    train_df['Duration'] = train_df['Duration'].apply(convert_to_minutes)
    test_df['Duration'] = test_df['Duration'].apply(convert_to_minutes)
    
    # Process genres and studios on combined datasets
    train_df, test_df = process_combined_features(train_df, test_df, 'Genres')

    # One-hot encode necessary columns
    columns_to_encode = ['Type', 'Source']
    train_df, test_df = process_combined_one_hot_encoding(train_df, test_df, columns_to_encode)
    
    return train_df, test_df

anime_fulldata_train, anime_fulldata_test = data_preprocessing_pipeline(anime_fulldata_train, anime_fulldata_test)

# Drop unnecessary columns for training and testing. Username, Anime Title and Image URL
anime_fulldata_train = anime_fulldata_train.drop(columns=['Username', 'Image URL','Studios'], axis=1)
anime_fulldata_test = anime_fulldata_test.drop(columns=['Username', 'Image URL','Studios'], axis=1)

# Print number of rows and columns
print(colored(f"Number of rows in the train dataset: {anime_fulldata_train.shape[0]}", "green"))
print(colored(f"Number of columns in the train dataset: {anime_fulldata_train.shape[1]}", "green"))

print(colored(f"Number of rows in the test dataset: {anime_fulldata_test.shape[0]}", "green"))
print(colored(f"Number of columns in the test dataset: {anime_fulldata_test.shape[1]}", "green"))

# print first 5 rows of the dataset
display(style_dataframe(anime_fulldata.head(5)))


#Implementing necessary functions

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_product / (norm1 * norm2)

def mean_absolute_error(predictions, actual):
    absolute_errors = [abs(x - y) for x, y in zip(predictions, actual)]
    return sum(absolute_errors) / len(absolute_errors)

def k_nearest_neighbors(X_train, X_test, y_train, k, similarity_func):
    predictions = []
    
    for i in range(len(X_test)):
        distances = []

        for j in range(len(X_train)):
            dist = similarity_func(X_test[i], X_train[j])
            # prediction rating, distance
            distances.append((y_train[j], dist))
        
        distances.sort(key=lambda x: x[1], reverse=True)
        neighbors = distances[:k]
        
        predicted_class = np.mean([neighbor[0] for neighbor in neighbors])
        predictions.append(predicted_class)
    
    return predictions


def weighted_k_nearest_neighbors(X_train, X_test, y_train, k, similarity_func):
    predictions = []
    
    for i in range(len(X_test)):
        distances = []

        for j in range(len(X_train)):
            dist = similarity_func(X_test[i], X_train[j])
            distances.append((y_train[j], dist))
        
        distances.sort(key=lambda x: x[1], reverse=True)
        neighbors = distances[:k]
        
        sum_weights = sum([neighbor[1] for neighbor in neighbors])
        predicted_value = sum([neighbor[0] * neighbor[1] for neighbor in neighbors]) / sum_weights if sum_weights > 0 else 0
        predictions.append(predicted_value)
    
    return predictions


# Prepare data for KNN

# Normalize the features for anime_fulldata_train
features = anime_fulldata_train.drop(['anime_id', 'user_id', 'anime_title', 'user_rating'], axis=1)
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)
normalized_df_train = pd.DataFrame(normalized_features, columns=features.columns)

normalized_df_train['user_id'] = anime_fulldata_train['user_id']
normalized_df_train['user_rating'] = anime_fulldata_train['user_rating']
normalized_df_train['anime_id'] = anime_fulldata_train['anime_id']

X_train = normalized_df_train.drop(['user_rating'], axis=1)  # features
y_train = normalized_df_train['user_rating']  # target variable

# Normalize the features for anime_fulldata_test
features = anime_fulldata_test.drop(['anime_id', 'user_id', 'anime_title', 'user_rating'], axis=1)
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)
normalized_df_test = pd.DataFrame(normalized_features, columns=features.columns)

normalized_df_test['user_id'] = anime_fulldata_test['user_id']
normalized_df_test['user_rating'] = anime_fulldata_test['user_rating']
normalized_df_test['anime_id'] = anime_fulldata_test['anime_id']

X_test = normalized_df_test.drop(['user_rating'], axis=1)  # features
y_test = normalized_df_test['user_rating']  # target variable


# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize a list to store the results as dictionaries
results = []

# Loop over different values of k
for k in [3, 5, 7]:
    knn_mae, weighted_knn_mae = [], []

    # Perform k-fold cross-validation
    for train_idx, test_idx in kf.split(X_train):
        Xtrain, Xval = X_train.iloc[train_idx], X_train.iloc[test_idx]
        ytrain, yval = y_train.iloc[train_idx], y_train.iloc[test_idx]

        # Get predictions from the models
        knn_preds = k_nearest_neighbors(Xtrain.to_numpy(), Xval.to_numpy(), ytrain.to_numpy(), k, cosine_similarity)
        weighted_knn_preds = weighted_k_nearest_neighbors(Xtrain.to_numpy(), Xval.to_numpy(), ytrain.to_numpy(), k, cosine_similarity)

        # Calculate the MAE for this fold
        knn_mae.append(mean_absolute_error(knn_preds, yval))
        weighted_knn_mae.append(mean_absolute_error(weighted_knn_preds, yval))

    # Record the average MAE for this value of k
    results.append({
        'K Value': k,
        'MAE (KNN)': np.mean(knn_mae),
        'MAE (Weighted KNN)': np.mean(weighted_knn_mae)
    })

# Convert the list of dictionaries to a DataFrame
results_df = pd.DataFrame(results)

# Plot the MAE for different k values
plt.figure(figsize=(10, 6))
plt.plot(results_df['K Value'], results_df['MAE (KNN)'], marker='o', linestyle='-', label='KNN')
plt.plot(results_df['K Value'], results_df['MAE (Weighted KNN)'], marker='s', linestyle='--', label='Weighted KNN')
plt.title('K Value vs. Mean Absolute Error')
plt.xlabel('K Value')
plt.ylabel('Mean Absolute Error')
plt.xticks(results_df['K Value'])
plt.legend()
plt.grid(True)
plt.show()

display(HTML('<h2 style="color: green; text-align: center;">Results:</h2>'))

styled_df = style_dataframe(results_df)
display(styled_df)

# Initialize a list to store the results as dictionaries
results = []

# Loop over different values of k
for k in [3, 5, 7]:

    print(colored(f"KNN Predictions for k = {k}:", "green"))

    knn_mae, weighted_knn_mae = [], []

    # Get predictions from the models
    knn_preds = k_nearest_neighbors(X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), k, cosine_similarity)
    weighted_knn_preds = weighted_k_nearest_neighbors(X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), k, cosine_similarity)

    # Calculate and store MAE
    knn_mae.append(mean_absolute_error(y_test, knn_preds))
    weighted_knn_mae.append(mean_absolute_error(y_test, weighted_knn_preds))

    # Record the average MAE for this value of k
    results.append({
        'K Value': k,
        'MAE (KNN)': np.mean(knn_mae),
        'MAE (Weighted KNN)': np.mean(weighted_knn_mae)
    })

# Convert the list of dictionaries to a DataFrame
results_df = pd.DataFrame(results)

# Plot the MAE for different k values
plt.figure(figsize=(10, 6))
plt.plot(results_df['K Value'], results_df['MAE (KNN)'], marker='o', linestyle='-', label='KNN')
plt.plot(results_df['K Value'], results_df['MAE (Weighted KNN)'], marker='s', linestyle='--', label='Weighted KNN')
plt.title('K Value vs. Mean Absolute Error')
plt.xlabel('K Value')
plt.ylabel('Mean Absolute Error')
plt.xticks(results_df['K Value'])
plt.legend()
plt.grid(True)
plt.show()
    

display(HTML('<h2 style="color: green; text-align: center;">Results for test:</h2>'))

styled_df = style_dataframe(results_df)
display(styled_df)

def k_nearest_neighbors_anime_recommendation(data, query_index, k, similarity_func):
    distances = []

    for j in range(len(data)):
        if j == query_index:
            continue  # Skip the query index
        dist = similarity_func(data[query_index], data[j])
        distances.append((j, dist))

    # Exclude the query index
    distances = [(j, dist) for j, dist in distances if j != query_index]

    # Sort by similarity
    distances.sort(key=lambda x: x[1], reverse=True)

    # Get the top k most similar indices
    neighbors = distances[:k]


    return neighbors  # Returns a list of tuples (index, similarity)

# Combine train and test data for anime recommendation
anime_fulldata = pd.concat([anime_fulldata_train, anime_fulldata_test], axis=0, ignore_index=True)

# Group the data by 'anime_title' and calculate the mean 'user_rating' for each anime.
anime_ratings = anime_fulldata.groupby('anime_title')['user_rating'].mean().reset_index()

# Drop duplicate rows based on 'anime_title', keeping only the first occurrence.
anime_features = anime_fulldata.drop_duplicates(subset='anime_title', keep='first')

# Merge the 'anime_features' DataFrame with the 'anime_ratings' DataFrame on the 'anime_title' column.
anime_features = pd.merge(anime_features, anime_ratings, on='anime_title', how='left', suffixes=('', '_mean'))

# Rename the 'user_rating_mean' column to 'mean_user_rating' for clarity.
anime_features.rename(columns={'user_rating_mean': 'mean_user_rating'}, inplace=True)

# Drop unnecessary columns 'anime_id', 'user_id', and 'user_rating' as they are not needed for recommendations.
anime_features.drop(['anime_id', 'user_id', 'user_rating'], axis=1, inplace=True)

anime_feature= anime_features.copy()

# Anime Index for which you want to get recommendations
target_index = 0

# Get the title of the anime for which you're making recommendations
target_anime_title = anime_feature.iloc[target_index]['anime_title']

# Keep a copy of anime titles
anime_titles = anime_feature['anime_title'].copy()

# Predicting the top 10 animes similar to the target anime
preds = k_nearest_neighbors_anime_recommendation(anime_feature.drop(['anime_title'], axis=1).to_numpy(), target_index, 10, cosine_similarity)

# Extract the indices of the recommended animes from the preds
indices = [index for index, _ in preds]

# Get unique titles from the indices, preserving the order
titles_seen = set()
unique_titles = []
for i in indices:
    title = anime_titles.iloc[i]
    if title not in titles_seen:
        unique_titles.append(title)
        titles_seen.add(title)


# Create the HTML content for styling
html_content = f'''
<div style="font-family: Arial, sans-serif;">
    <h2 style="color: #4CAF50; text-align: center;">🌟 Recommendations for "{target_anime_title}" 🌟</h2>
    <ul style="background-color: #f2f2f2; border: 1px solid #e0e0e0; border-radius: 5px; padding: 20px;">
'''

for i, title in enumerate(titles_seen, 1):
    html_content += f'''
    <li style="margin-bottom: 10px; font-size: 18px;">
        <span style="color: #555555;"><strong>{i}.</strong> {title}</span>
    </li>
    '''

html_content += '''
    </ul>
</div>
'''

# Display the HTML content
display(HTML(html_content))