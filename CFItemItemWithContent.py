from Strategy import Strategy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

NUM_OF_ATTR = 4

class CFItemItemWithContent(Strategy):

    def __init__(self, data_items_train):
        self.data_items_train = data_items_train
        projects_data = pd.read_csv('projects_meta_data_4_attr.csv', index_col=0)
        self.projects_data = projects_data.fillna('')
        self.data_matrix = self.calculate_similarity_by_content()

    def calculate_similarity_by_content(self):
        tf_idf = TfidfVectorizer()
        total_similarity = pd.DataFrame(0, index=range(len(self.projects_data)), columns=range(len(self.projects_data)))
        for feature in self.projects_data.columns:
            feature_similarity = tf_idf.fit_transform(self.projects_data[feature])
            feature_similarity = pd.DataFrame(cosine_similarity(feature_similarity))
            total_similarity = total_similarity + feature_similarity / NUM_OF_ATTR  # check the division!!!
        total_similarity.index = [str(i) for i in self.projects_data.index]
        total_similarity.columns = [str(i) for i in self.projects_data.index]
        return total_similarity

    def get_recommendations(self, user_index, known_user_likes_train, k):
        user_projects = self.data_matrix[known_user_likes_train]  # without ratings!!

        neighbourhood_size = 10
        data_neighbours = pd.DataFrame(0,user_projects.columns, range(1, neighbourhood_size + 1))
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