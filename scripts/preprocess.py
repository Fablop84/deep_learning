# preprocess.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def load_dataset(path):
    """
    Carga el dataset desde un archivo CSV.
    """
    df = pd.read_csv(path)
    return df

def create_engagement_label(df, threshold=0.3):
    """
    Crea una etiqueta binaria de engagement alto/bajo.
    """
    df['engagement_ratio'] = (df['Likes'] + df['Bookmarks']) / df['Visits']
    df['label'] = (df['engagement_ratio'] > threshold).astype(int)
    return df

def preprocess_metadata(df):
    """
    Escala variables numéricas y codifica categorías.
    """
    scaler = MinMaxScaler()
    numeric_cols = ['tier', 'xps', 'Visits', 'Likes', 'Dislikes', 'Bookmarks']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    encoder = OneHotEncoder(sparse_output=False)
    categories_encoded = encoder.fit_transform(df[['categories']])
    categories_df = pd.DataFrame(categories_encoded, columns=encoder.get_feature_names_out(['categories']))
    
    df = pd.concat([df.reset_index(drop=True), categories_df.reset_index(drop=True)], axis=1)
    return df