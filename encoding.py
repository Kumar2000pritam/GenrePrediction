import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sentence_transformers import SentenceTransformer


def add_text_embedding(df, text_model):
    """
    Adds sentence embeddings as numerical features to the dataframe.

    Steps:
    1. Fill missing text with empty string
    2. Generate embeddings using pre-trained model
    3. Convert embeddings into DataFrame
    4. Concatenate embeddings with original dataframe
    """

    # Handle missing text safely
    texts = df["combined_text"].fillna("").tolist()

    # Generate embeddings (shape: [n_samples, embedding_dim])
    emb = text_model.encode(texts)

    # Ensure numpy format
    emb = np.array(emb)

    # Convert embeddings to DataFrame with column names 0...n
    emb_df = pd.DataFrame(
        emb,
        columns=[f"{i}" for i in range(emb.shape[1])]
    )

    # Merge original data with embedding features
    return pd.concat([df.reset_index(drop=True), emb_df], axis=1)


def encode_rating(X_train, X_test):
    """
    Ordinally encodes 'rating_cleaned' using domain-specific order.

    Key Points:
    - Maintains logical ordering (G < PG < PG-13 < R ...)
    - Handles unseen categories in test set using value = -1
    - Prevents data leakage by fitting only on training data
    """

    # Domain-driven rating hierarchy
    rating_order = [[
        'Not Rated', 'G', 'TV-Y', 'TV-Y7', 'TV-G',
        'PG', 'TV-PG', 'PG-13', 'TV-14', 'R', 'TV-MA'
    ]]

    # Initialize encoder with safe unknown handling
    oe = OrdinalEncoder(
        categories=rating_order,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    # Fit only on train, transform both
    X_train['rating_encoded'] = oe.fit_transform(X_train[['rating_cleaned']])
    X_test['rating_encoded'] = oe.transform(X_test[['rating_cleaned']])

    return X_train, X_test, oe


def frequency_encode_release_year(X_train, X_test):
    """
    Applies frequency encoding to 'release_year_bin'.

    Idea:
    - Replace each category with its relative frequency in training data
    - Captures distribution importance instead of arbitrary numbers

    Important:
    - Test values not seen in train → filled with 0.0
    """

    # Compute normalized frequency from training data
    freq_map = X_train['release_year_bin'].value_counts(normalize=True)

    # Apply mapping on train
    X_train['release_year_bin_encoded'] = (
        X_train['release_year_bin'].map(freq_map).astype(float)
    )

    # Apply mapping on test (handle unseen values safely)
    X_test['release_year_bin_encoded'] = (
        X_test['release_year_bin']
        .map(freq_map)
        .astype(float)   # convert first
        .fillna(0.0)     # unseen categories → 0
    )

    return X_train, X_test, freq_map


def encode_duration_category(X_train, X_test):
    """
    Encodes 'duration_category' using ordinal encoding.

    Logic:
    - Movies ordered by increasing duration
    - TV shows ordered by number of seasons
    - Unknown categories handled as -1
    """

    # Logical ordering of duration categories
    duration_order = [[
        "Movie_<1hr",
        "Movie_1-1.5hr",
        "Movie_1.5-2hr",
        "Movie_2-2.5hr",
        "Movie_2.5hr+",
        "TV_1",
        "TV_2-3",
        "TV_4-5",
        "TV_6-10"
    ]]

    # Initialize encoder
    oe = OrdinalEncoder(
        categories=duration_order,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    # Fit on train, transform both datasets
    X_train['duration_category_encoded'] = oe.fit_transform(X_train[['duration_category']])
    X_test['duration_category_encoded'] = oe.transform(X_test[['duration_category']])

    return X_train, X_test, oe


def generate_text_embeddings(X_train, X_test, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    """
    Generates sentence embeddings for train and test datasets.

    Steps:
    1. Load pre-trained SentenceTransformer model
    2. Encode 'combined_text' column
    3. Save embeddings to disk (for reuse / speed optimization)

    Returns:
    - train embeddings
    - test embeddings
    - loaded model (for inference reuse)
    """

    # Load pre-trained transformer model
    model = SentenceTransformer(model_name)

    # Generate embeddings for train and test
    # train_emb = model.encode(
    #     X_train['combined_text'].tolist(),
    #     show_progress_bar=True
    # )

    # test_emb = model.encode(
    #     X_test['combined_text'].tolist(),
    #     show_progress_bar=True
    # )

    # Save embeddings (useful for large datasets to avoid recomputation)
    # np.save("x_train_temp_embeddings.npy", train_emb)
    # np.save("x_test_temp_embeddings.npy", test_emb)

    # Optional: Load instead of recomputing (commented)
    train_emb = np.load('x_train_temp_embeddings.npy')
    test_emb = np.load('x_test_temp_embeddings.npy')

    return train_emb, test_emb, model