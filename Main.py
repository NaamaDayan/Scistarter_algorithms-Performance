from CFItemItem import CFItemItem
from CFUserUser import CFUserUser
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from more_itertools import unique_everseen
from random import randint
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from PopularityBased import PopularityBased
from SVD import SVD


def get_data():
    data1 = pd.read_csv('users_projects_full.csv')  # this file is without testers users
    data1 = data1.dropna(0, 'all')
    data1 = data1.set_index(pd.Index(list(range(data1.shape[0]))))
    data_items1 = data1.drop('user', 1)
    return data_items1


def calculate_similarity(train):
    data_sparse = sparse.csr_matrix(train)
    similarities = cosine_similarity(data_sparse.transpose())
    sim = pd.DataFrame(similarities, train.columns, train.columns)
    return sim


def get_precision_and_recall(user_index, k, algorithm):
    # Get the projects the user has participated in.
    known_user_likes = data_items.loc[user_index]
    known_user_likes = known_user_likes[known_user_likes >0].index.values
    if len(known_user_likes) > 2:
        known_user_likes_train, known_user_likes_test = train_test_split(known_user_likes)
        relevant_projects = algorithm.get_recommendations(user_index, known_user_likes_train, k)

        # calculate recall and precision - this is the same value since the sets are the same size
        precision = np.intersect1d(relevant_projects, known_user_likes_test).size / len(relevant_projects)
        recall = np.intersect1d(relevant_projects, known_user_likes_test).size / known_user_likes_test.size
        return [precision, recall]
    return [-1, -1]


def get_precision_and_recall_by_time_split(user_index, k, algorithm):
    user = data.loc[user_index][0]
    user_data = records[records.profile == user]
    projects_list = list(user_data['project'].values)
    projects_list = [str(int(x)) for x in projects_list if x is not None and x==x]
    projects_list = list(unique_everseen(projects_list))
    projects_list = [int(x) for x in projects_list]
    if len(projects_list)>2:
        splitter_index = max(1, int(0.9*len(projects_list)))
        # split to train and test by timeline!!
        known_user_likes_train = projects_list[:splitter_index]
        known_user_likes_test = projects_list[splitter_index:]
        relevant_projects = algorithm.get_recommendations(user_index, known_user_likes_train, k)
        print (relevant_projects)
        # calculate recall and precision - this is the same value since the sets are the same size
        precision = np.intersect1d(relevant_projects, known_user_likes_test).size / len(relevant_projects)
        special_percision = calc_special_precision(relevant_projects, known_user_likes_test)
        recall = np.intersect1d(relevant_projects, known_user_likes_test).size / len(known_user_likes_test)
        return [precision,recall, special_percision]
    return [-1, -1, -1]


def calc_special_precision(relevant_projects, known_user_likes_test):
    known_user_likes_test = [int(x) for x in known_user_likes_test]
    precision = np.intersect1d(relevant_projects, known_user_likes_test).size / len(relevant_projects)
    rejected_recs = list(set(relevant_projects) - set(known_user_likes_test))
    similarity_sums = np.sum([calc_max_sim(rp, known_user_likes_test) for rp in rejected_recs])
    return precision + similarity_sums / len(relevant_projects)


def calc_max_sim(rejected_project, chosen_projects):
    return np.max([data_matrix.loc[str(cp)][str(rejected_project)] for cp in chosen_projects])


def precision_recall_at_k(k_values, test_users, algorithm):
    for k in k_values:
        results = []
        for user in test_users:
            results.append(get_precision_and_recall_by_time_split(user, k, algorithm))
        precisions = np.mean([i[0] for i in results if i>=0])
        recalls = np.mean([i[1] for i in results if i>=0])
        special_precisions = np.mean([i[2] for i in results if i>=0])
        print (precisions, recalls, special_precisions)

def calculate_similarity_by_content():
    tf_idf = TfidfVectorizer()
    total_similarity = pd.DataFrame(0, index=range(len(projects_data)), columns=range(len(projects_data)))
    for feature in projects_data.columns:
        feature_similarity = tf_idf.fit_transform(projects_data[feature])
        feature_similarity = pd.DataFrame(cosine_similarity(feature_similarity))
        total_similarity = total_similarity + feature_similarity / 4  # check the division!!!
    total_similarity.index = [str(i) for i in projects_data.index]
    total_similarity.columns = [str(i) for i in projects_data.index]
    return total_similarity


data = pd.read_pickle('data.pkl')
data_items = data.drop('user', 1)
# records = pd.read_pickle('records.pkl')#rec.pkl
records = pd.read_pickle('rec.pkl')
data_items_train = pd.read_pickle('data_items_train.pkl')
# projects_data = pd.read_csv('projects_meta_data_4_attr.csv', index_col=0) #dev.csv
projects_data = pd.read_csv('dev.csv', index_col=0)
projects_data = projects_data.fillna('')
data_matrix = calculate_similarity_by_content()
#data_matrix = calculate_similarity(data_items_train)


def main():
    #example
    cf = SVD(data_items_train)
    print(get_precision_and_recall_by_time_split(32, 3, cf))



def get_result(user_index, k):
    cf = CFUserUser(data_items_train)
    return get_precision_and_recall_by_time_split(user_index, k, cf)

if __name__ == "__main__":
    main()
