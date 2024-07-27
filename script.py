import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import torch
import time
import threading

# Configurar Matplotlib para usar el backend 'Agg'
import matplotlib
matplotlib.use('Agg')

print("Búsqueda Semántica de Artículos Científicos en ArXiv")

# Paso 1: Cargar el dataset
print("Carga de datos...")
df = pd.read_csv('./data/arxiv_sample.csv', encoding='utf-8')
print("Datos cargados")

# Paso 2: Preprocesamiento de Datos
print("Preprocesamiento de datos...")
def preprocess_text(text):
    text = text.lower()
    return text

df['processed_abstract'] = df['abstract'].apply(preprocess_text)
print("Preprocesamiento completo")

# Paso 3: Generación de Embeddings
print("Cargando el modelo y generando embeddings...")

# Función para mostrar el tiempo transcurrido
def show_elapsed_time(start_time):
    while not embeddings_generated:
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        time_str = f"Tiempo transcurrido: {int(minutes)} minutos y {int(seconds)} segundos"
        print(time_str, end='\r')
        time.sleep(1)

# Variable para controlar el estado de la generación de embeddings
embeddings_generated = False

# Iniciar el temporizador en un hilo separado
start_time = time.time()
timer_thread = threading.Thread(target=show_elapsed_time, args=(start_time,))
timer_thread.start()

# Cargar el modelo y tokenizer de BERT
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Función para generar embeddings
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Promediar los embeddings
    return embeddings.detach().numpy()

# Generar y almacenar embeddings para los abstracts
embeddings = np.array([generate_embeddings(text)[0] for text in df['processed_abstract']])
embeddings_generated = True
timer_thread.join()
print("\nEmbeddings generados y almacenados")

# Paso 4: Búsqueda Semántica
def search(query, embeddings, df, top_n=5):
    query = preprocess_text(query)  # Preprocesar la consulta
    query_embedding = generate_embeddings(query)[0]
    similarities = cosine_similarity([query_embedding], embeddings)[0]  # Calcular la similitud coseno
    top_indices = similarities.argsort()[-top_n:][::-1]  # Obtener los índices de los artículos más similares
    return df.iloc[top_indices]  # Devolver los artículos más relevantes

# Busqueda
query = input("Ingrese lo que desea buscar: ")
results = search(query, embeddings, df)
print(f"Resultados de búsqueda de : '{query}'")
if results.empty:
    print("No se encontraron resultados")
else:
    print(results[['title', 'authors', 'abstract']])

print("\nGenerando visualización de embeddings...")

# Visualización de resultados
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Función para limpiar el texto
def clean_text(text):
    return ''.join(char for char in text if char.isalnum() or char in [' ', '.'])

plt.figure(figsize=(10, 10))

# Mostrar solo un subconjunto de títulos para evitar superposición
# Esto porque se mostraban demasiados titulos y la imagen se veia con una mancha negra
subset_size = 10  # Número de títulos a mostrar
indices = np.random.choice(len(df['title']), subset_size, replace=False)

for i in indices:
    word, vector = df['title'].iloc[i], reduced_embeddings[i]
    clean_word = clean_text(word)  # Limpiar el texto. Daba error al hacer la grafica por los caracteres especiales
    plt.scatter(vector[0], vector[1], alpha=0.5)
    plt.text(vector[0] + 0.05, vector[1] + 0.05, clean_word, fontsize=9, alpha=0.7)

plt.title('Visualización de Embeddings usando PCA')

# Guardar el gráfico en un archivo
plt.savefig('embeddings_visualization.png')
print("Visualización guardada como 'embeddings_visualization.png'")

