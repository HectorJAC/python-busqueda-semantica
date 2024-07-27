# Funcion para hacer la limpieza del daataset de arxiv

import pandas as pd
import json

# Ruta del dataset
filename = '../../../../Descargas/archive/arxiv-metadata-oai-snapshot.json'  

# Leer el dataset línea por línea
articles = []
with open(filename, 'r') as f:
    for i, line in enumerate(f):
        if i >= 2000:  # Extraer el numero de articulos que se desean
            break
        article = json.loads(line)
        # Extraer los campos del titulo, del autor y el abstract
        articles.append({
            'title': article.get('title'),
            'authors': article.get('authors'),
            'abstract': article.get('abstract')
        })

# Convertir a un DataFrame de pandas
df = pd.DataFrame(articles)

# Guardar el DataFrame en un nuevo archivo CSV
df.to_csv('./data/arxiv_sample.csv', index=False)
