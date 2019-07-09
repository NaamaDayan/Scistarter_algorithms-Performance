from Strategy import Strategy
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class CFUserUser(Strategy):

    def __init__(self, data_items_train):
        self.data_items_train = data_items_train

    def findksimilarusers(self, user_id, k, metric='cosine'):
        model_knn = NearestNeighbors(k, 1.0, 'brute', 30, metric)
        model_knn.fit(self.data_items_train)
        distances, indices = model_knn.kneighbors(
            self.data_items_train.iloc[user_id, :].values.reshape(1, -1), n_neighbors=k + 1)
        similarities = 1 - distances.flatten()
        return pd.Series(similarities, indices[0])

    def get_user_projects(self, user_id):
        known_user_likes = self.data_items_train.loc[user_id]
        known_user_likes = known_user_likes[known_user_likes > 0].index.values
        return known_user_likes

    def get_recommendations_helper(self, user_index, known_user_likes_train, k, knn_var):
        similar_users = self.findksimilarusers(user_index, k=knn_var)
        if user_index in similar_users.index:
            similar_users = similar_users.drop(user_index, 0)
        similar_projects = [self.get_user_projects(user) for user in similar_users.index]
        similar_projects = list(set([item for sublist in similar_projects for item in sublist]))
        projects_scores = dict.fromkeys(similar_projects, 0)
        for s_project in similar_projects:
            for user in similar_users.index:
                projects_scores[s_project] += similar_users.loc[user] * self.data_items_train.loc[user][s_project]
        projects_scores = sorted(projects_scores.items(), key=lambda x: x[1], reverse=True)  # sort
        recommended_projects = [i[0] for i in projects_scores]
        recommended_projects = list(filter(lambda x: x not in known_user_likes_train, recommended_projects))
        while len(recommended_projects) < k:
            recommended_projects = self.get_recommendations_helper(user_index, known_user_likes_train, k, knn_var+100)  #increase knn_var until sufficient variety of projects
        return recommended_projects[:k]

    def get_recommendations(self, user_index, known_user_likes_train, k):
        return self.get_recommendations_helper(user_index, known_user_likes_train, k, 100)

