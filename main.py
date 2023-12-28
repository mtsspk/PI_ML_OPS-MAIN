import pandas as pd
from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import Optional

app = FastAPI()

#http://127.0.0.1:8000

'''
# crear clase BaseModel que garantiza los tipos de datos de las variables
class Libro(BaseModel):     
    titulo: str
    autor: str
    paginas: int
    editorial: str

@app.get("/")
def index():
    return {"hola": "alo"}

# ruta que contiene variables
@app.get("/libros/{id}")    
def mostrar_libro(id:int):
    return {"data": id}

@app.post("/libros")    # post sirve para 
def insertar_libro(libro: Libro):
    return {"message": f"libro {libro.titulo} insertado"}   # retorno de una clase
'''

# Cargar el archivo CSV en un DataFrame
df_play_gnr = pd.read_csv('../../Datasets/df_play_gnr.csv')

# Cargar el archivo CSV en un DataFrame
df_usr_gnr_top = pd.read_csv('../../Datasets/df_usr_gnr_top.csv')

# Cargar el archivo CSV en un DataFrame
df_usr_gnr_top_year = pd.read_csv('../../Datasets/df_usr_gnr_top_year.csv')

# Cargar el archivo CSV en un DataFrame
df_games_rec_top_3 = pd.read_csv('../../Datasets/df_games_rec_top_3.csv')

# Cargar el archivo CSV en un DataFrame
df_dev_rec_worst_3 = pd.read_csv('../../Datasets/df_dev_rec_worst_3.csv')

# Cargar el archivo CSV en un DataFrame
df_dev_sent = pd.read_csv('../../Datasets/df_dev_sent.csv')




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





@app.get("/sentiment_analysis")
def sentiment_analysis():
    # Filtrar registros con al menos un análisis de sentimiento
    df_filtered = df_dev_sent[df_dev_sent['total_sentiment'] > 0]

    # Obtener desarrolladores con total_sentiment mayor a 0
    developers_with_sentiment = df_filtered['developer'].unique()

    # Crear el diccionario con desarrolladores como llaves (sin datos en este paso)
    sentiment_dict = {developer: [] for developer in developers_with_sentiment}

    # Lista de listas para valores de negative, neutral y positive
    sentiment_values_list = []

    # Insertar valores de análisis de sentimientos en la lista
    for index, row in df_filtered.iterrows():
        sentiment_values = {
            "Negative": row['negative_sentiment'],
            "Neutral": row['neutral_sentiment'],
            "Positive": row['positive_sentiment']
        }
        sentiment_values_list.append(sentiment_values)

    # Insertar valores en el diccionario de desarrolladores
    for i, developer in enumerate(developers_with_sentiment):
        sentiment_dict[developer] = sentiment_values_list[i]

    # Presentar la cantidad de desarrolladores con registros de análisis de sentimientos
    message = f"Desarrolladores con registros de análisis de sentimientos: {len(sentiment_dict)}"

    # Presentar el diccionario resultante
    result = {"message": message, "sentiment_dict": sentiment_dict}
    
    return result




