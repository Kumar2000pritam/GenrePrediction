import pandas as pd
from preprocess import create_combined_text
from encoding import add_text_embedding
def merge_tabular_and_text(X_train, X_test, train_emb, test_emb):
    """
    Concatenates tabular features with text embeddings.
    """

    X_train_tab = X_train.reset_index(drop=True)
    X_test_tab = X_test.reset_index(drop=True)

    train_text_df = pd.DataFrame(train_emb).reset_index(drop=True)
    test_text_df = pd.DataFrame(test_emb).reset_index(drop=True)

    X_train_final = pd.concat([X_train_tab, train_text_df], axis=1)
    X_test_final = pd.concat([X_test_tab, test_text_df], axis=1)

    return X_train_final, X_test_final

def final_cleanup(X_train, X_test, drop_cols):
    """
    Drops unused / leakage / raw columns before training.
    """

    X_train = X_train.drop(columns=drop_cols, errors='ignore')
    X_test = X_test.drop(columns=drop_cols, errors='ignore')

    return X_train, X_test

def genre_maping():
    genre_map = {
    "Comedy": "Comedies",
    "Comedies": "Comedies",
    "Dramas": "Dramas",
    "Drama": "Dramas",
    
    "International Movies": "International",
    "International TV Shows": "International",

    "Action & Adventure": "Action",
    "Action-Adventure": "Action",

    "Romantic Movies": "Romance",
    "Romantic TV Shows": "Romance",

    "TV Dramas": "Dramas",
    "TV Comedies": "Comedies",
    "TV Action & Adventure": "Action",
    "TV Sci-Fi & Fantasy": "SciFi & Fantasy",
    "TV Thrillers": "Thrillers",
    "TV Mysteries": "Mystery",

    "Sci-Fi & Fantasy": "SciFi & Fantasy",
    "Science Fiction": "SciFi & Fantasy",

    "Children & Family Movies": "Family",
    "Family": "Family",

    "Kids TV": "Kids",
    "Kids' TV": "Kids",

    "Docuseries": "Documentary",
    "Documentaries": "Documentary",
    "Documentary": "Documentary",
     "Documentaries": "Documentary",
    "Docuseries": "Documentary",
    "Horror Movies": "Horror",
    "[Thrillers]": "Thriller",
    "['Thrillers']": "Thriller",
    "Sports Movies": "Sports",

    "Independent Movies": "Independent",

    "Anime Series": "Anime",
    "Animation": "Animation",
    "Musical":"Music",
    "Crime TV Shows": "Crime",
    "Reality TV": "Reality",

    "LGBTQ Movies": "LGBTQ",
    "Music & Musicals": "Music",
    "Thrillers": "Thriller",
    "TV Horror": "Horror",
    "Anime Features": "Anime"
}
    return genre_map

def build_features(df, artifacts):

    df = df.copy()

    rating_encoder = artifacts["rating_encoder"]
    duration_encoder = artifacts["duration_encoder"]
    freq_map = artifacts["freq_map"]
    text_model = artifacts.get("text_model")
    feature_columns = artifacts["feature_columns"]

    # 1. text
    df = create_combined_text(df)

    # 2. encodings
    df["rating_encoded"] = rating_encoder.transform(df[["rating_cleaned"]])
    df["duration_category_encoded"] = duration_encoder.transform(df[["duration_category"]])

    df["release_year_bin_encoded"] = (
        df["release_year_bin"]
        .map(freq_map)
        .astype(float)
        .fillna(0.0)
    )

    # 3. embeddings
    if text_model is not None:
        df = add_text_embedding(df, text_model)

    # 4. DROP RAW / INTERMEDIATE COLUMNS
    drop_cols = [
        'title','description','director','cast',
        'rating','duration',
        'release_year',
        'duration_num','is_season',
        'movie_duration','num_seasons',
        'movie_duration_bin','season_bin',
        'duration_category','release_year_bin',
        'rating_cleaned'
    ]

    df = df.drop(columns=drop_cols, errors='ignore')
    # print(df.columns)
    # 5. FINAL ALIGNMENT WITH TRAINING FEATURES
    # keep only what exists in feature_columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df