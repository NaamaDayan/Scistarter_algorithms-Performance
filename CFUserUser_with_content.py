from Strategy import Strategy
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np


class CFUserUser_with_content(Strategy):

    def __init__(self, data_items_train, data_matrix):
        self.data_items_train = data_items_train
        self.data_matrix = data_matrix

    def findksimilarusers(self, user_id, metric='cosine', k=50):
        similarities = []
        indices = []
        model_knn = NearestNeighbors(5, 1.0, 'brute', 30, metric)
        model_knn.fit(self.data_items_train)
        distances, indices = model_knn.kneighbors(
            self.data_items_train.iloc[user_id, :].values.reshape(1, -1), n_neighbors=k + 1)
        similarities = 1 - distances.flatten()
        return pd.Series(similarities, indices[0])

    def get_user_projects(self, user_id):
        known_user_likes = self.data_items_train.loc[user_id]
        known_user_likes = known_user_likes[known_user_likes > 0].index.values
        return known_user_likes

    def get_recommendations(self, user_index, known_user_likes_train, k):
        similarUsers = self.findksimilarusers(user_index).drop(user_index, 0)
        similar_projects = [self.get_user_projects(user) for user in similarUsers.index]
        similar_projects = list(set([item for sublist in similar_projects for item in sublist]))
        projects_scores = dict.fromkeys(similar_projects, 0)
        for s_project in similar_projects:
            for user in similarUsers.index:
                projects_scores[s_project] += similarUsers.loc[user] * self.data_items_train.loc[user][s_project]
        projects_scores = sorted(projects_scores.items(), key=lambda x: x[1], reverse=True)  # sort
        recommended_projects = [i[0] for i in projects_scores]
        recommended_projects = list(set(recommended_projects) - set(known_user_likes_train))[:200]

        projects_predicted_ratings = [[project, np.mean(self.data_matrix.loc[project][known_user_likes_train])] for project in recommended_projects]
        projects_predicted_ratings = sorted(projects_predicted_ratings, key=lambda i: i[1])
        projects_predicted_ratings = [i[0] for i in projects_predicted_ratings]  # take the projects ids
        return projects_predicted_ratings[-k:]