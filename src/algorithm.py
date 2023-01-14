import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

NUM_CLUSTER = 10
NUM_RECOMMENDATIONS = 100
NUM_QUESTIONS = 10


def random_suggestion(df, prob_list):
    cluster = np.random.choice(range(NUM_CLUSTER), p=np.asarray(prob_list) / sum(prob_list))
    return df[df["cluster"] == cluster].sample(1).iloc[0] #returns all info of the book in pd format


def get_rec_tfidf(df, book, cluster):

  same_cluster_df = df.loc[df["cluster"]==cluster].reset_index()
  
  tfidf = TfidfVectorizer(stop_words='english')
  tfidf_matrix = tfidf.fit_transform(same_cluster_df['description'])

  cosine_sim_tfidf = linear_kernel(tfidf_matrix, tfidf_matrix)
  indices = pd.Series(same_cluster_df.index, same_cluster_df['title']).drop_duplicates()

  sim_scores = sorted(enumerate(cosine_sim_tfidf[indices[book]]), key=lambda x: x[1], reverse=True)[1:1+NUM_RECOMMENDATIONS]
  movie_indices = [i[0] for i in sim_scores]

  return same_cluster_df.iloc[movie_indices]


def get_rec_CV(df, book, cluster):
    same_cluster_df = df.loc[df["cluster"] == cluster].reset_index()
    
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(same_cluster_df['metadata'])

    cosine_sim_CV = cosine_similarity(count_matrix, count_matrix)
    indices = pd.Series(same_cluster_df.index, same_cluster_df['title']).drop_duplicates()

    idx = indices[book]
    sim_scores = sorted(enumerate(cosine_sim_CV[idx]), key=lambda x: x[1], reverse=True)[1:1+NUM_RECOMMENDATIONS]

    movie_indices = [i[0] for i in sim_scores]

    return same_cluster_df.iloc[movie_indices]


def get_final_recs(df, book, cluster, num):  #for a book
    suggestions_1 = get_rec_CV(df, book, cluster)
    suggestions_2 = get_rec_tfidf(df, book, cluster)

    all_recs = pd.concat([suggestions_1, suggestions_2]).drop_duplicates()
    probs = all_recs["average_rating"] / all_recs["average_rating"].sum()
    # indices_list=np.arange(len(all_recs))
    # chosen_index = np.random.choice(indices_list, p=)

    final_recs = all_recs.sample(n=num, weights=probs)
    # print(all_recs["average_rating"], all_recs["average_rating"].sum())

    #
    # final_recs = [np.random.choice(all_recs, p=probs) for _ in range(ratio)]

    return final_recs


def get_all_suggestions(df, prob_list, n_q=NUM_QUESTIONS):
    all_records = pd.DataFrame(random_suggestion(df, prob_list=prob_list) for _ in range(n_q))

    return all_records

from itertools import compress
def final_recs_based_on_answers(df, prob_list, binary_list, suggestions):
    num_likes = sum(binary_list)
    if not num_likes:
        print("nothing was liked, serving random books")
        return get_all_suggestions(df, prob_list, NUM_RECOMMENDATIONS)

    num_per_book = NUM_RECOMMENDATIONS//num_likes

    final = []
    for book in compress(suggestions.iloc, binary_list):
        final.append(get_final_recs(df, book["title"], book["cluster"], num_per_book))
        prob_list[book["cluster"]] += 1

    return pd.concat(final)
