import pandas as pd
from fastapi import FastAPI
#import pyarrow.parquet as pq
#from typing import List

#import numpy as np
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics.pairwise import cosine_similarity
# from pydantic import BaseModel
# from typing import Optional


app = FastAPI()

# Cargar el archivo CSV en un DataFrame
df_play_gnr = pd.read_csv('Datasets/df_play_gnr.csv')

# Cargar el archivo CSV en un DataFrame
df_usr_gnr_top = pd.read_csv('Datasets/df_usr_gnr_top.csv')

# Cargar el archivo CSV en un DataFrame
df_usr_gnr_top_year = pd.read_csv('Datasets/df_usr_gnr_top_year.csv')

# Cargar el archivo CSV en un DataFrame
df_games_rec_top_3 = pd.read_csv('Datasets/df_games_rec_top_3.csv')

# Cargar el archivo CSV en un DataFrame
df_dev_rec_worst_3 = pd.read_csv('Datasets/df_dev_rec_worst_3.csv')

# Cargar el archivo CSV en un DataFrame
df_dev_sent = pd.read_csv('Datasets/df_dev_sent.csv')

# Cargar el archivo CSV en un DataFrame
df_games_names = pd.read_csv('Datasets/df_games_names.csv')

# Cargar el archivo Parquet en un DataFrame
df_games_similarity = pd.read_parquet('Datasets/df_games_similarity.parquet')




# Cargar el archivo CSV en un DataFrame
#df_games_rec_ML = pd.read_csv('Datasets/df_games_rec_ML.csv')

# Cargar el archivo CSV en un DataFrame
#scaled_data = pd.read_csv('Datasets/scaled_data.csv')
# Importar scaled_data como numpy array
#scaled_data = np.genfromtxt('Datasets/scaled_data.csv', delimiter=',')


@app.get("/")
def index():
    return {"Proyecto Steam"}


@app.get("/PlayTimeGenre")
def PlayTimeGenre(genre: str):
    # Filtrar el DataFrame por el género proporcionado
    genre_data = df_play_gnr[df_play_gnr['genre'] == genre]

    if not genre_data.empty:
        # Encontrar el release_year con el mayor playtime
        max_playtime_row = genre_data.loc[genre_data['playtime'].idxmax()]
        max_release_year = max_playtime_row['release_year']
        max_playtime = max_playtime_row['playtime']

        return f"Para el género {genre}, el release_year con mayor playtime es {max_release_year} con un total de playtime de {max_playtime}."
    else:
        return f"No hay datos disponibles para el género {genre}."
    


@app.get("/UserForGenre")
def UserForGenre(genre: str):
    # Filtrar df_usr_gnr_top por el género proporcionado
    genre_data_top = df_usr_gnr_top[df_usr_gnr_top['genre'] == genre]

    if not genre_data_top.empty:
        # Encontrar el user_id con mayor playtime_forever para el género
        max_playtime_row = genre_data_top.loc[genre_data_top['playtime_forever'].idxmax()]
        max_user_id = max_playtime_row['user_id']
        max_playtime_forever = int(max_playtime_row['playtime_forever'])  # Convertir a int

        # Filtrar df_usr_gnr_top_year por el user_id y el género
        user_year_data = df_usr_gnr_top_year[(df_usr_gnr_top_year['user_id'] == max_user_id) & (df_usr_gnr_top_year['genre'] == genre)]

        # Preparar resultados
        result = {
            "genre": genre,
            "max_user_id": max_user_id,
            "max_playtime_forever": max_playtime_forever,
            "details": user_year_data[['release_year', 'playtime_forever', 'playtime_acumulated']].astype(int).to_dict(orient='records') if not user_year_data.empty else None
        }

        return result
    else:
        return {"error": f"No hay datos disponibles para el género {genre} en df_usr_gnr_top."}



@app.get("/UsersRecommend")
def UsersRecommend(posted_year: int):
    # Filtrar df_games_rec_top_3 por el posted_year proporcionado
    year_data = df_games_rec_top_3[df_games_rec_top_3['posted_year'] == posted_year]

    if not year_data.empty:
        # Presentar mensaje
        message = f"Juegos más recomendados para el año {posted_year}:"

        # Mostrar game_id, app_name y total_recommend para el posted_year dado
        result = year_data[['game_id', 'app_name', 'total_recommend']].to_dict(orient='records')

        return {"message": message, "result": result}
    else:
        return {"message": f"No hay datos disponibles para el posted_year {posted_year} en df_games_rec_top_3."}






    
@app.get("/UsersWorstDeveloper")
def UsersWorstDeveloper(posted_year: int):
    # Filtrar df_dev_rec_worst_3 por el posted_year proporcionado
    year_data = df_dev_rec_worst_3[df_dev_rec_worst_3['posted_year'] == posted_year]

    if not year_data.empty:
        # Presentar mensaje
        message = f"Desarrolladores con más reseñas negativas para el año {posted_year} son:"

        # Mostrar developer y neg_rec para el posted_year dado
        result = year_data[['developer', 'neg_rec']].to_dict(orient='records')

        return {"message": message, "result": result}
    else:
        return {"message": f"No hay datos disponibles para el posted_year {posted_year} en df_dev_rec_worst_3."}



@app.get("/sentiment_analysis/{developer}")
def sentiment_analysis(developer: str):
    # Filtrar registros con al menos un análisis de sentimiento para el desarrollador proporcionado
    df_filtered = df_dev_sent[(df_dev_sent['total_sentiment'] > 0) & (df_dev_sent['developer'] == developer)]

    if not df_filtered.empty:
        # Crear el diccionario con el desarrollador como llave y una lista de sentimientos como valor
        sentiment_dict = {developer: []}

        # Insertar valores de análisis de sentimientos en la lista
        for index, row in df_filtered.iterrows():
            sentiment_values = {
                "Negative": row['negative_sentiment'],
                "Neutral": row['neutral_sentiment'],
                "Positive": row['positive_sentiment']
            }
            sentiment_dict[developer].append(sentiment_values)

        # Presentar el diccionario resultante
        result = {"developer": developer, "sentiments": sentiment_dict[developer]}
    else:
        result = {"message": f"No hay datos disponibles para el desarrollador {developer} en df_dev_sent."}

    return result


@app.get("/get_recommendations/{game_id}")
def get_recommendations(game_id: str, num_recommendations: int = 5):
    game_id = int(game_id)
    
    game_row = df_games_similarity.loc[game_id]
    
    similar_games = game_row.sort_values(ascending=False).index.tolist()

    # Excluir el juego de entrada de la lista de recomendaciones
    similar_games = [game for game in similar_games if game != game_id]

    # Tomar las primeras num_recommendations recomendaciones
    recommendations = similar_games[:num_recommendations]

    return {"recommendations": recommendations}
