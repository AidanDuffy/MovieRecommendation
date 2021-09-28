"""
Author: Aidan Duffy
Creation Date: April 10, 2021
Last Updated: April 29, 2021
Description: This is the main program file for the Movie Recommender system. I
used several tutorials which provided me with the dataset in order to complete
the task. These helped me find functions from modules we had covered in class,
but not these specific functions, ie vectorization fucntions from sklearn.
ADD: I would also like for users to be able to store their own ratings and based
off of a generated profile, the system will recommend to them.
"""
import os
import string
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata

movies_df_index = 0
movies_metadata_df_index = 1
tags_index, movies_index, ratings_index, links_index = 0,1,2,3
genome_tags_index, genome_scores_index, credits_index, metadata_index = 4,5,6,7

here = os.path.abspath(__file__)
input_dir = os.path.abspath(os.path.join(here, os.pardir + r"/Data"))


def parse_files(init = False, nums = None):
    """
    This provides dataframes from the given data csv files.
    :param init: if this is the first run
    :param nums:
    :return: the dataframes from the parsed files.
    """
    dfs = list()
    file_names = [r"tags", r"movies", r"ratings_small", r"links_small",
                  r"genome-tags", r"genome-scores", r'credits',
                  r"movies_metadata"]
    used_files = list()
    if init:
        nums = [movies_index,metadata_index]
    if nums is None:
        print("Error, no list of numbers provided for the files!")
        return
    for num in nums:
        if num < 0 or num > 7:
            print("Error! Bad number given!")
            return
        used_files.append(file_names[num])
    for file in used_files:
        df = pd.read_csv(os.path.join(input_dir, file + ".csv"), delimiter=",")
        dfs.append(df)
    if init:
        # Elminate all the generic words
        tf_vect = TfidfVectorizer(stop_words='english')
        # Replaces all null entries with the empty string
        dfs[movies_metadata_df_index]['overview'] = \
            dfs[movies_metadata_df_index]['overview'].fillna('')
        # Transform and fit our data into this matrix
        tf_matrix = tf_vect.fit_transform(
            dfs[movies_metadata_df_index]['overview'])
        return dfs[0], dfs[1], tf_matrix
    else:
        return dfs


def format_titles(title1, title2=None, single_title=None):
    stop = ["the", 'The', "a", 'A', 'an', 'An', "Les", "La"]
    title1 = title1.str.replace('[{}]'.format(string.punctuation), '')
    title1 = title1.str.lower()
    new_title1 = list()
    count = 0
    for title in title1:
        if title is None or title != title:
            continue
        words = title.split()
        if words[0] in stop and title2 is not None:
            words = words[1:]
        final_title = " ".join(words)
        new_title1.append(final_title)
        count += 1
    if title2 is not None:
        title2 = title2.str.replace('[{}]'.format(string.punctuation), '')
        title2 = title2.str.replace(" ", "")
        title1 = title1.str.replace(" ", "")
        title2 = title2.str.lower()
        return title1, title2
    elif single_title is not None:
        single_title = single_title.replace('[{}]'.format(
            string.punctuation), '').lower()
        return new_title1, single_title
    else:
        return new_title1


def remove_accents(title):
    """
    I encountered some miscellaneous accent chars like in Les Mis, so
    this is to remove those.
    :param title: given movie title
    :return: Normalized title
    """
    nfkd_form = unicodedata.normalize('NFKD', title)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


def fix_years_add_movie_id(metadata, movies):
    """
    WORK IN PROGRESS
    I am trying to add the proper year of release as well as the movieId to the
    metadata file, though the formatting of the titles is somewhat hindering
    this as it is not consistent in formatting or sometimes the language.
    :param metadata: movie metadata dataframe
    :param movies: movie csv dataframe
    :return: metadata dataframe with the proper years and movieId
    ADD: Need to find a way to cross-reference the movieId with another file.
    """
    years, ids = list(), list()
    metadata_index = 0
    meta_titles_df = metadata["title"]
    movie_titles_df = movies["title"]
    meta_titles_df, movie_titles_df = format_titles(meta_titles_df,
                                                    movie_titles_df)
    for i in range(movies.shape[0]):
        meta_title = meta_titles_df[metadata_index].lower().replace(" ", "")
        current_title = movie_titles_df[i]
        year = current_title[-4:]
        current_title = current_title[:-5].lower().replace(" ", "")
        current_title = remove_accents(current_title)
        if check_title(meta_title, current_title):
            years.append(year)
            ids.append(movies['movieId'][i])
            metadata_index += 1
    metadata['movieId'] = ids
    metadata['year'] = years
    return metadata


def calculate_weighted_rating(df, min_vote, avg):
    """
    IMDB's formula for calculated a weighted rating.
    :param df: The movie metadata dataframe.
    :param min_vote: the minimum number of votes to be considered
    :param avg: the average vote score
    :return: the weighted rating.
    """
    average_rating = df['vote_average']
    vote_count = df['vote_count']
    denominator = min_vote + vote_count
    weight = ((vote_count / denominator) * average_rating) \
             + ((min_vote / denominator) * avg)
    return weight


def weighted_rating(movie_data, percentile):
    """
    This will help create a top X% chart of movies.
    :param percentile: user's desired percentile of popularity
    :param movie_data: movie metadata dataframe
    :return:
    """
    mean_vote = movie_data['vote_average'].mean()
    min_vote = movie_data['vote_count'].quantile(percentile)
    print("\n\nFor the entered percentile(" + str(
        100 * percentile) + ") the film " +
          "requires at least " + str(round(min_vote)) + " votes.\n")
    top_x_percent = movie_data.copy().loc[movie_data['vote_count'] >= min_vote]
    top_x_percent['weighted_score'] = top_x_percent.apply(
        calculate_weighted_rating, axis=1, args=(min_vote, mean_vote))
    top_x_percent = top_x_percent.sort_values('weighted_score',
                                              ascending=False)
    return top_x_percent


def generic_recommendations(title, metadata, indexes, scores):
    """
    This recommends the 10 movies with the most similar plots.
    :param title: Given title
    :param metadata: dataframe
    :param indexes: all the indexes
    :param scores: similarity scores
    :return:
    Add: allow users to alter the number of films recommended as well as add a
    popularity filter. Perhaps a filter for year as well? And language?
    """
    index = indexes[title]
    # Get all the sim scores for this title and sort them
    similarity = list(enumerate(scores[index]))
    similarity = sorted(similarity, key=lambda x: x[1],
                        reverse=True)  # We want the score, not the movie id being the weight
    ten_best = similarity[1:11]  # [0] would be the film itself!
    movie_indexes = [i[0] for i in ten_best]
    print("Based on the plot of " + title + ", we recommend: ")
    print(metadata['title'].iloc[movie_indexes])
    return


def get_list_of_features(data):
    if isinstance(data, list):
        names = [i['name'] for i in data]
        if len(names) > 3:
            names = names[:3]
        return names
    else:
        return []


def get_genre(data):
    data['genres'] = data['genres'].apply(get_list_of_features)
    return


def format_data(data):
    if isinstance(data,list):
        result = list()
        for movie in data:
            result.append(movie.replace(" ","").lower())
        return result
    elif isinstance(data, str):
        return data.lower().replace(" ","")
    else:
        return ""


def extra_recommender_data(data,genre=False,credit=False):
    metadata_vector = ""
    if genre:
        metadata_vector = " ".join(data['genres'])
    elif credit:
        metadata_vector = " ".join(data['genres']) + " " +\
                          " ".join(data['director']) + " " +\
                          " ".join(data['cast'])
    return metadata_vector


def finish_extra_recommend(title, metadata):
    count_matrix = CountVectorizer(stop_words='english').fit_transform(
        metadata['vector_data'])
    count_similarity_matrix = cosine_similarity(count_matrix, count_matrix)
    data = metadata.reset_index()
    indexes = pd.Series(data.index, index=metadata['title'])
    generic_recommendations(title,metadata, indexes,count_similarity_matrix)


def genre_based_recommendations(title, metadata):
    feature = 'genres'
    #These were causing errors and needed to be removed
    bad_ids = [19730,29503, 35587]
    metadata = metadata.drop(bad_ids)
    metadata[feature] = metadata[feature].apply(literal_eval)
    get_genre(metadata)
    metadata[feature] = metadata[feature].apply(format_data)
    metadata['vector_data'] = metadata.apply(extra_recommender_data, axis = 1,
                                             args = (True, False))
    finish_extra_recommend(title,metadata)
    return


def get_director(data):
    for crew in data:
        if crew['job'] == 'Director':
            return crew['name']
    return np.nan


def get_cast_and_crew(data):
    data['director'] = data['crew'].apply(get_director)
    data['cast'] = data['cast'].apply(get_list_of_features)
    return


def credits_based_recommendations(title, metadata):
    credits_df = parse_files(nums=[credits_index])[0]
    bad_ids = [19730, 29503, 35587]
    metadata = metadata.drop(bad_ids)
    features = ['cast', 'crew', 'genres']
    credits_df['id'] = credits_df['id'].astype('int')
    metadata['id'] = metadata['id'].astype('int')
    metadata = metadata.merge(credits_df, on='id')
    for feature in features:
        metadata[feature] = metadata[feature].apply(literal_eval)
    get_genre(metadata)
    get_cast_and_crew(metadata)
    features = ['cast', 'director', 'genres']
    for feature in features:
        metadata[feature] = metadata[feature].apply(format_data)
    metadata['vector_data'] = metadata.apply(extra_recommender_data, axis=1,
                                             args=(False, True))
    finish_extra_recommend(title, metadata)
    return


def content_based_recommender(title, metadata, indexes, scores,
                              plot_based=True, genre_based=False,
                              credits_based=False):
    """
    This is the main pipeline for the subtypes of the recommendation system.
    :param title: film title
    :param metadata: movie metadata dataframe
    :param indexes:
    :param scores: is the similary score matrix between movies
    :param plot_based: should we issue a plot based recommendation? default yes
    :param genre_based: should it issues a plot and genre based rec? default no
    :param credits_based: should it issue a plot, genre, and cast/crew
    based rec? default no
    :return:
    """
    if plot_based:
        generic_recommendations(title, metadata, indexes, scores)
    elif genre_based:
        genre_based_recommendations(title, metadata)
    elif credits_based:
        credits_based_recommendations(title, metadata)
    return


def check_title(title, metadata):
    """
    Checks if the given name is in the metadata file.
    :param title: user-given title
    :param metadata: movie metadata dataframe
    :return: True or False based on if the title is valid or not.
    TO ADD: Possibly check if it is possible to add a 'did you mean' for typos
    and what not.
    """
    movie_titles, given_title = format_titles(metadata['title'],
                                              single_title=title)
    for movie_title in movie_titles:
        if given_title == movie_title:
            return True
    return False


def popularity_weighted_ratings(movie_meta):
    percentile_q = "What percentile, in terms of votes received, would you " \
                   "like to see the top films for?\nEx: 25th percentile only " \
                   "needs 3 votes, but 90th needs 160 and 95th needs " \
                   "over 400!\nEnter as a float, so \'0.90\' for the 90th" \
                   " percentile: "
    top_x = "How many popular films would you like displayed? Top 20? 50?" \
            "(Enter a valid int below 500 and above 0): "
    while True:
        try:
            percentile = float(input(percentile_q))
            top_num = int(input(top_x))
            if top_num < 500 and top_num > 0:
                break
            else:
                print("Invalid integer!")
        except:
            print("Enter a valid number!")
    top_x_percent = weighted_rating(movie_meta, percentile)
    print(top_x_percent[
        ['title', 'vote_count', 'vote_average', 'weighted_score']].head(
        top_num))


def check_rec_types():
    check_yes = input("\nWould you like to have plot-based recommendations? ")
    yes = ['yes', 'Yes', 'y', 'Y']
    plot, genre, credit = False, False, False
    if check_yes in yes:
        plot = True
        check_yes = input("\nWould you like to also factor in genre-based "
                          "recommendations? ")
        if check_yes in yes:
            plot = False
            genre = True
            check_yes = input("\nWould you like to also factor in the cast and"
                              " crew into your recommendations? ")
            if check_yes in yes:
                genre = False
                credit = True
    return plot, genre, credit


def get_user_title(movie_meta):
    title = input("What movie do you want recommendations from? ")
    valid_title = check_title(title, movie_meta)
    while valid_title is False:
        title = input("Invalid! What movie do you want recommendations from? ")
        valid_title = check_title(title, movie_meta)
    print("Valid title! Please wait while the similarity matrix is created...")
    return title


def main():
    movies, movie_meta, movie_meta_matrix = parse_files(init = True)
    # movie_meta = fix_years_add_movie_id(movie_meta,movies)
    indexes = pd.Series(movie_meta.index,
                        index=movie_meta['title']).drop_duplicates()
    print("NOTIFICATION: Anytime there is a yes/no question, type"
          " \'yes\', \'Yes\', \'y\', or\'Y\' for yes, anything else will be"
          " read as a no.\n\n")
    check_yes = input("Would you like to see the most highly rated films given"
                      " certain popularities? \n(Ex: see top 10 from the 65th "
                      "percentile, in terms of popularity.)\n")
    yes = ['yes', 'Yes', 'y', 'Y']
    if check_yes in yes:
        popularity_weighted_ratings(movie_meta)
    check_yes = input(
        "\nWould you like to use the movie recommendation system?\n")
    if check_yes in yes:
        plot, genre, credit = check_rec_types()
        title = get_user_title(movie_meta)
        if plot:
            similarity_matrix = linear_kernel(movie_meta_matrix,
                                              movie_meta_matrix)
        else:
            similarity_matrix = None
        content_based_recommender(title, movie_meta, indexes,
                                  similarity_matrix,
                                  plot, genre, credit)
    print("Thanks for using the program!")


if __name__ == '__main__':
    main()
