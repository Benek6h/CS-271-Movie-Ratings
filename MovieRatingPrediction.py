import pandas as pd
import numpy as np
import networkx as nx
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

# Load and pre-process data ---------------------------------------------------------------------
# Ratings: user_id, movie_id, rating, timestamp
ratings = pd.read_csv("data/ratings.dat", sep="::", names=["user_id", "movie_id", "rating", "timestamp"], engine="python", encoding="ISO-8859-1")

# Users: user_id, gender, age, occupation, zip_code
users = pd.read_csv("data/users.dat", sep="::", names=["user_id", "gender", "age", "occupation", "zip_code"], engine="python", encoding="ISO-8859-1")

# Movies: movie_id, title, genres
movies = pd.read_csv("data/movies.dat", sep="::", names=["movie_id", "title", "genres"], engine="python", encoding="ISO-8859-1")

# Merge all the data together
df = ratings.merge(users, on="user_id").merge(movies, on="movie_id")

# Drop any rows with missing values
df.dropna(inplace=True)

# Build bipartide graph ---------------------------------------------------------------------
# The graph will contain:
#   - User partition
#   - Movie partition
#   - Edges connect user->movie with 'weight' = rating
#       - Higher rating == thicker line

# Blank graph
B = nx.Graph()

# Add user nodes
# Need to add a prefix because users and movies can share the same ID
user_nodes = df["user_id"].unique()
prefixed_user_nodes = [f"U_{u}" for u in user_nodes]
B.add_nodes_from(prefixed_user_nodes, bipartite="user")

# Add movie nodes
# Need to add a prefix because users and movies can share the same ID
movie_nodes = df["movie_id"].unique()
prefixed_movie_nodes = [f"M_{m}" for m in movie_nodes]
B.add_nodes_from(prefixed_movie_nodes, bipartite="movie")

# Add edges (user -> movie) with weight = rating
for row in df.itertuples(index=False):
    user = f"U_{row.user_id}"
    movie = f"M_{row.movie_id}"
    B.add_edge(user, movie, weight=row.rating)

# Compute degree dictionaries with numeric keys by stripping prefixes
user_degree = {int(u[2:]): deg for u, deg in B.degree() if u.startswith("U_")}
movie_degree = {int(m[2:]): deg for m, deg in B.degree() if m.startswith("M_")}

# extract features ---------------------------------------------------------------------
#   - degree
#   - average rating
#   - user demographics: age, gender, occupation
#   - Centrality (WIP: NOT ADDED YET)

# User-level features
user_avg_rating = df.groupby("user_id")["rating"].mean().to_dict()
user_age = df.groupby("user_id")["age"].first().to_dict()
user_gender = df.groupby("user_id")["gender"].first().to_dict()
user_occupation = df.groupby("user_id")["occupation"].first().to_dict()

# Movie-level features
movie_avg_rating = df.groupby("movie_id")["rating"].mean().to_dict()

# Build feature matrix and target vector ---------------------------------------------------------------------
# For each row (user_id, movie_id, rating), combine user features
# and movie features into a single feature vector

# Returns a list of numeric features for a user movie pair (user, movie)
def build_feature_vector(u_id, m_id):
    # Combine user and movie features
    features = [
        user_degree.get(u_id, 0),                  # Number of movies the person watched
        user_avg_rating.get(u_id, 0),              # Average rating the user gives
        user_age.get(u_id, 0),                     # Age of user
        1 if user_gender[u_id] == "M" else 0,      # Change gender to numeric M-->1, F-->0
        user_occupation.get(u_id, 0),              # Occupation of user
        movie_degree.get(m_id, 0),                 # Number of people that watched the movie
        movie_avg_rating.get(m_id, 0)              # Average rating of the movie
    ]
    return features

# Feature Matrix
X = []

# Target Vector
y = []

# Put these features all into a feature maxtrix and target vector
for row in df.itertuples(index=False):
    u = row.user_id
    m = row.movie_id
    rating = row.rating
    feat_vec = build_feature_vector(u, m)
    X.append(feat_vec)
    y.append(rating)

X = np.array(X)
y = np.array(y)


kfold = KFold(n_splits=5, shuffle=True)
scores = []

# Regression Model
for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.01, max_depth=2,
                             colsample_bytree=1, gamma=0)

    # Train the model on training data
    model.fit(X_train, y_train)


    # Predict on the test set
    y_pred = model.predict(X_test)

    # Model evaulation---------------------------------------------------------------------
    # Evaulation is made using RMSE
    rmse = mean_squared_error(y_test, y_pred)
    scores.append(rmse)

for i in range(0, len(scores)):
    print(f"Prediction RMSE - model_{i}: {scores[i]:.2f}")

# Classification Model
kfold_classification = KFold(n_splits=5, shuffle=True)
scores_f1 = []
scores_roc = []
y= y-1

for train_index, test_index in kfold_classification.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = xgb.XGBClassifier(objective='multi:softprob', num_class=5, n_estimators=500, learning_rate=0.02, max_depth=8,
                             colsample_bytree=0.8, gamma=0.2)

    # Train the model on training data
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred_roc = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    scores_f1.append(f1_score(y_test, y_pred, average="macro"))
    scores_roc.append(roc_auc_score(y_test, y_pred_roc, average="macro", multi_class="ovr"))

for i in range(0, len(scores_f1)):
    print(f"ROC-score - model_{i}: {scores_roc[i]:.2f}")

for i in range(0, len(scores_f1)):
    print(f"F1-score - model_{i}: {scores_f1[i]:.2f}")

# Make graph from a smal subset of the data (bipartide graph)------------------------------
# Get a subset size that is <= 1 million
subset_size = 100

# Use prefix user id's to grab the correct nodes
subset_users = [f"U_{uid}" for uid in list(user_nodes)[:subset_size]]
subset_movies = [f"M_{mid}" for mid in list(movie_nodes)[:subset_size]]
sub_nodes = subset_users + subset_movies
subgraph = B.subgraph(sub_nodes)
plt.figure(figsize=(10, 8))

# Use bipartite_layout
pos = nx.bipartite_layout(subgraph, subset_users)

nx.draw_networkx_nodes(subgraph, pos, nodelist=subset_users, node_color='green', label='Users', node_size=100)
nx.draw_networkx_nodes(subgraph, pos, nodelist=subset_movies, node_color='red', label='Movies', node_size=100)
nx.draw_networkx_edges(subgraph, pos, node_size=100)
plt.title("Bipartite Graph Subset")
plt.legend()
plt.savefig('Bipartite Graph Subset.png')

# Degree distribution graph (histogram)------------------------------
# User and movie degrees from the bipartite graph
user_degrees = list(user_degree.values())
movie_degrees = list(movie_degree.values())

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(user_degrees, bins=20, color='skyblue', edgecolor='black')
plt.title("User Degree Distribution")
plt.xlabel("Degree (Number of Connections)")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(movie_degrees, bins=20, color='salmon', edgecolor='black')
plt.title("Movie Degree Distribution")
plt.xlabel("Degree (Number of Connections)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig('Histogram of Movie Degree Distribution.png')


# Plot User Degree vs. Average rating (Scatter plot)------------------------------
# Create df that has the user, their degree, and the average rating they give
user_stats = pd.DataFrame({
    "user_id": list(user_degree.keys()),
    "degree": list(user_degree.values()),
    "avg_rating": [user_avg_rating[u] for u in user_degree.keys()]
})

plt.figure(figsize=(8, 6))
plt.scatter(user_stats["degree"], user_stats["avg_rating"], alpha=0.5, color='purple')
plt.title("User Degree vs. Average Rating")
plt.xlabel("Degree (Number of Connections)")
plt.ylabel("Average Rating Given")
plt.grid(True)
plt.savefig('Scatter plot of User vs Average Rating.png')
