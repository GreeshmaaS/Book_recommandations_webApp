import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

books = pd.read_csv(r"D:\python\flask\static\allbooks.csv", encoding="UTF-8")
rows, cols = books.shape


indx = pd.Series(books.index, index=books["name"])

# Function that get book recommendations based on the cosine similarity score of book genre
def genre_recommendations(title):
    books["corpus"] = pd.Series(
        books[["genre1", "genre2", "genre3", "genre4", "genre5", "genre6"]]
        .fillna("")
        .values.tolist()
    ).str.join(" ")

    tf1 = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=0, stop_words="english"
    )
    tfidf_matrix1 = tf1.fit_transform(books["corpus"].head(rows - 1))
    cosine_sim1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)
    indices1 = pd.Series(books.index, index=books["name"])
    idx = indices1[title]
    sim_scores = list(enumerate(cosine_sim1[idx]))
    # print(sim_scores)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return books.iloc[book_indices]


# df = genre_recommendations("Truly Devious")
# print(df["name"])

####
# bookarray=list(books['name'])

# book recommendations based on one main genre


def genre_books(title):
    books["corpus"] = pd.Series(
        books[["genre1", "genre2", "genre3", "genre4", "genre5", "genre6"]]
        .fillna("")
        .values.tolist()
    ).str.join(" ")
    tf1 = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=0, stop_words="english"
    )
    tfidf_matrix1 = tf1.fit_transform(books["corpus"].head(rows - 1))
    cosine_sim1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)
    indices1 = pd.Series(books.index, index=books["genre1"])
    # print(indices1)
    idx = indices1[title]
    # print(idx)  # .lower()
    # sim_scores = list(enumerate(cosine_sim1[idx]))
    # print(list(sim_scores[0][1]))
    # sim_scores = sorted(sim_scores[1], key=lambda x: x[1], reverse=True)
    # sim_scores = sim_scores[1:21]
    # book_indices = [i[0] for i in sim_scores]
    return books.iloc[idx]


# print(genre_books("Classics"))


# book recommendations based on author
def author_books(title):
    tf1 = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=0, stop_words="english"
    )
    tfidf_matrix1 = tf1.fit_transform(books["author"].head(rows - 1))
    cosine_sim1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)
    indices1 = pd.Series(books.index, index=books["author"])
    idx = indices1[title]
    return books.iloc[idx]


# author_books('Chimamanda Ngozi Adichie')
