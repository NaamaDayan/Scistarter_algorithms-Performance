from Strategy import Strategy
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


class CFItemItem(Strategy):

    def __init__(self, data_items_train):
        self.data_items_train = data_items_train
        self.data_matrix = self.calculate_similarity()

    def calculate_similarity(self):
        data_sparse = sparse.csr_matrix(self.data_items_train)
        similarities = cosine_similarity(data_sparse.transpose())
        sim = pd.DataFrame(similarities, self.data_items_train.columns, self.data_items_train.columns)
        return sim

    def get_recommendations(self, user_index, known_user_likes_train, k):
        user_projects = self.data_matrix[known_user_likes_train]  # without ratings!!
        neighbourhood_size = 10
        data_neighbours = pd.DataFrame(0, user_projects.columns, range(1, neighbourhood_size + 1))
        for i in range(0, len(user_projects.columns)):
            data_neighbours.iloc[i, :neighbourhood_size] = user_projects.iloc[0:, i].sort_values(0, False)[
                                                           :neighbourhood_size].index

        # Construct the neighbourhood from the most similar items to the
        # ones our user has already liked.
        most_similar_to_likes = data_neighbours.loc[known_user_likes_train]
        similar_list = most_similar_to_likes.values.tolist()
        similar_list = list(set([item for sublist in similar_list for item in sublist]))
        similar_list = list(set(similar_list) - set(known_user_likes_train))
        neighbourhood = self.data_matrix[similar_list].loc[similar_list]

        user_vector = self.data_items_train.loc[user_index].loc[similar_list]

        score = neighbourhood.dot(user_vector).div(neighbourhood.sum(1))
        relevant_projects = score.nlargest(k).index.tolist()
        return relevant_projects