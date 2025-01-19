import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The '%matplotlib inline' magic command is specific to Jupyter Notebook; comment it out for VS Code
# %matplotlib inline
plt.style.use("ggplot")

from sklearn.decomposition import TruncatedSVD

# Load dataset
amazon_ratings = pd.read_csv('ratings_Beauty.csv')
amazon_ratings = amazon_ratings.dropna()
print(amazon_ratings.head())

print("Shape of amazon_ratings:", amazon_ratings.shape)

# Group by ProductId and count ratings
popular_products = pd.DataFrame(amazon_ratings.groupby('ProductId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
print("Top 10 most popular products:\n", most_popular.head(10))

# Plot top 30 popular products
most_popular.head(30).plot(kind="bar")
plt.show()

# Take the first 10,000 rows for processing
amazon_ratings1 = amazon_ratings.head(10000)

# Create a pivot table
ratings_utility_matrix = amazon_ratings1.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)
print("Utility Matrix Head:\n", ratings_utility_matrix.head())

print("Shape of Utility Matrix:", ratings_utility_matrix.shape)
X = ratings_utility_matrix.T
print("Shape of Transposed Matrix:", X.shape)

# Perform Truncated SVD
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
print("Shape of Decomposed Matrix:", decomposed_matrix.shape)

# Compute the correlation matrix
correlation_matrix = np.corrcoef(decomposed_matrix)
print("Shape of Correlation Matrix:", correlation_matrix.shape)

# Example product index
i = "6117036094"
product_names = list(X.index)
product_ID = product_names.index(i)

# Correlations for the product
correlation_product_ID = correlation_matrix[product_ID]
print("Shape of Correlation for product ID:", correlation_product_ID.shape)

# Recommend products with high correlation
Recommend = list(X.index[correlation_product_ID > 0.90])

# Remove the item itself from recommendations
if i in Recommend:
    Recommend.remove(i)

print("Recommended Products:", Recommend[0:9])

# Feature extraction and clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Update the file path for the product descriptions dataset
product_descriptions = pd.read_csv('product_descriptions.csv')  # Adjust the path as needed
print("Shape of product_descriptions:", product_descriptions.shape)

product_descriptions = product_descriptions.dropna()
product_descriptions1 = product_descriptions.head(500)

vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(product_descriptions1["product_description"])

# K-Means clustering
kmeans = KMeans(n_clusters=10, init='k-means++')
y_kmeans = kmeans.fit_predict(X1)
plt.plot(y_kmeans, ".")
plt.show()

# Define a function to print cluster terms
def print_cluster(i):
    print(f"Cluster {i}:")
    for ind in order_centroids[i, :10]:
        print(f' {terms[ind]}')
    print()

# Set the number of clusters and fit the model
true_k = 10
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X1)

# Print cluster terms
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(true_k):
    print_cluster(i)

# Recommendation function
def show_recommendations(product):
    Y = vectorizer.transform([product])
    prediction = model.predict(Y)
    print_cluster(prediction[0])

# Example recommendations
show_recommendations("cutting tool")
show_recommendations("spray paint")
show_recommendations("steel drill")
show_recommendations("water")
