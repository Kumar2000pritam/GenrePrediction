import pandas as pd
import numpy as np
import re



def handle_missing_values(df):
    """
    Handles missing values in key columns:
    - Fills 'rating' using mode per genre (listed_in)
    - Fills 'director' and 'cast' with 'Unknown'

    Parameters:
    df (pd.DataFrame): Raw input dataframe

    Returns:
    pd.DataFrame: Cleaned dataframe with missing values handled
    """

    # Fill rating based on genre-level mode
    df['rating'] = df.groupby('listed_in')['rating'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else df['rating'].mode()[0])
    )

    # Fill categorical missing values
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['country'] = df['country'].fillna('Unknown')
    df.dropna(subset=['duration'], inplace=True)
    df = df.loc[:, df.notna().sum() >= 5]
    # print(df.isnull().sum())
    return df

def create_combined_text(df):
    """
    Creates a unified text feature combining multiple metadata fields.

    Useful for NLP models like TF-IDF, BERT, embeddings, etc.

    Parameters:
    df (pd.DataFrame)

    Returns:
    pd.DataFrame with new column 'combined_text'
    """

    df['combined_text'] = (
        "Title: " + df['title'].fillna('') + ". " +
        "Description: " + df['description'].fillna('') + ". " +
        "Director: " + df['director'].fillna('Unknown') + ". " +
        "Cast: " + df['cast'].fillna('Unknown')
    )

    return df

def bin_release_year(df):
    """
    Converts release year into decade-based categorical bins.

    Parameters:
    df (pd.DataFrame)

    Returns:
    pd.DataFrame with 'release_year_bin'
    """

    bins = [1900, 1980, 1990, 2000, 2010, 2020, 2030]
    labels = ['<1980', '1980s', '1990s', '2000s', '2010s', '2020s']

    df['release_year_bin'] = pd.cut(df['release_year'], bins=bins, labels=labels)

    return df
def extract_and_filter_genres(df, min_count=50):
    """
    Extracts genres from 'listed_in', cleans them, and filters rare genres.

    Steps:
    1. Split listed_in into list of genres
    2. Flatten genres to compute frequency
    3. Keep only genres with frequency >= min_count
    4. Filter dataframe accordingly

    Parameters:
    df (pd.DataFrame)
    min_count (int): Minimum frequency threshold for genre retention

    Returns:
    pd.DataFrame with cleaned 'genre' column
    pd.Series genre_counts
    """

    # Step 1: split genres
    df['genre'] = df['listed_in'].fillna("").apply(
        lambda x: [g.strip() for g in x.split(",") if g.strip()]
    )

    # Step 2: flatten
    all_genres = [g for sublist in df['genre'] for g in sublist]

    # Step 3: frequency count
    genre_counts = pd.Series(all_genres).value_counts()

    # Step 4: keep top genres
    top_genres = set(genre_counts[genre_counts >= min_count].index)

    # Step 5: filter df
    df = df[
        df['genre'].apply(lambda x: any(g in top_genres for g in x))
    ].reset_index(drop=True)

    return df, genre_counts



def clean_genres(df, col="listed_in"):
    """
    Cleans genre strings and converts them into list format.

    Steps:
    - Handles missing values
    - Removes brackets/quotes
    - Splits by comma
    - Normalizes spacing

    Returns:
    df with new column 'genre'
    """

    def clean_genre(g):
        g = str(g)
        g = re.sub(r"[\[\]\(\)\{\}\'\"]", "", g)
        g = re.sub(r"\s+", " ", g).strip()
        return g

    df = df.copy()

    df["genre"] = df[col].fillna("").apply(
        lambda x: [clean_genre(g) for g in x.split(",") if g.strip()]
    )

    return df

def map_genres(df, genre_map):
    """
    Maps raw genres into standardized categories.

    Example:
    'Comedy' → 'Comedies'
    'Sci-Fi' → 'SciFi & Fantasy'

    Parameters:
    df (pd.DataFrame)
    genre_map (dict)

    Returns:
    pd.DataFrame
    """

    df['genre'] = df['genre'].apply(
        lambda x: list(set([genre_map.get(g, g) for g in x]))
    )

    return df



def filter_genres_by_min_count(df, min_count=50, col="genre"):
    """
    Filters rare genres based on frequency threshold.

    Steps:
    - Flatten genres
    - Compute frequency
    - Keep only genres >= min_count
    - Filter dataframe rows
    """

    all_genres = [g for sublist in df[col] for g in sublist]
    genre_counts = pd.Series(all_genres).value_counts()

    valid_genres = set(genre_counts[genre_counts >= min_count].index)

    df = df.copy()

    df[col] = df[col].apply(
        lambda genres: [g for g in genres if g in valid_genres]
    )

    df = df[df[col].map(len) > 0].reset_index(drop=True)

    return df, genre_counts


def process_duration_features(df):
    """
    Extracts numeric duration features and separates:
    - Movie duration
    - TV seasons

    Also creates bins for both.

    Parameters:
    df (pd.DataFrame)

    Returns:
    pd.DataFrame
    """
    df.dropna(subset=['duration'],inplace=True)
    # Extract numeric part
    df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)

    # Flag TV vs Movie
    df['is_season'] = df['duration'].str.contains('Season', na=False).astype(int)

    # Split features
    import numpy as np

    df['movie_duration'] = df.apply(
        lambda x: x['duration_num'] if x['is_season'] == 0 else np.nan,
        axis=1
    )

    df['num_seasons'] = df.apply(
        lambda x: x['duration_num'] if x['is_season'] == 1 else np.nan,
        axis=1
    )

    # Binning movie duration
    df['movie_duration_bin'] = pd.cut(
        df['movie_duration'],
        bins=[0,60,90,120,150,300],
        labels=['<1hr','1-1.5hr','1.5-2hr','2-2.5hr','2.5hr+']
    )

    # Binning seasons
    df['season_bin'] = pd.cut(
        df['num_seasons'],
        bins=[0,1,3,5,10,50],
        labels=['1','2-3','4-5','6-10','10+']
    )

    # Final combined category
    df['duration_category'] = np.where(
        df['is_season'] == 1,
        'TV_' + df['season_bin'].astype(str),
        'Movie_' + df['movie_duration_bin'].astype(str)
    )

    return df
def clean_rating_column(df):
    """
    Cleans noisy / rare rating labels into standardized categories.

    Rules:
    - NR, UR → Not Rated
    - TV-Y7-FV → TV-Y7
    - NC-17 → R (merged due to rarity)

    Parameters:
    df (pd.DataFrame)

    Returns:
    pd.DataFrame
    """

    def clean_rating(r):
        r = str(r).strip()

        if r in ['NR', 'UR']:
            return 'Not Rated'
        elif r == 'TV-Y7-FV':
            return 'TV-Y7'
        elif r == 'NC-17':
            return 'R'
        return r

    df['rating_cleaned'] = df['rating'].apply(clean_rating)
    return df