#!/usr/bin/env python
# coding: utf-8

# In[90]:


import numpy as np
import pandas as pd

dataset = pd.read_csv("/Users/alexandremartinez/Desktop/Data_Science/Datasets/Youtube_dataset/CAvideos.csv")
dataset.head()


# In[91]:


#Observamos un resumen del dataset

dataset.describe()


# In[92]:


#Comenzamos el filtrado de datos. 

#Empezamos eliminando las columnas que no nos servirán para este estudio

lista_col = ["video_id", "trending_date","channel_title","category_id","comment_count","thumbnail_link","comments_disabled", "ratings_disabled","description", "tags"]

dataset = dataset.drop(lista_col, axis=1)
dataset.head(40)


# In[93]:


#Ahora veremos si hay algun vídeo que no se haya subido bien 

dataset[dataset["video_error_or_removed"]==True].head()

print(f"Longitud dataset inicial {len(dataset)}")

dataset = dataset[dataset["video_error_or_removed"] != True]

print(f"Longitud dataset final {len(dataset)}")


# In[94]:


#Ahora eleminaremos las filas de aquellos valores nulos en las columnas numéricas. 
#Primero debo verificar si no estamos ante objetos y confirmar que son datos numéricos.

dataset.dtypes


# In[95]:


dataset[pd.isna(dataset["views"])]
print(f"Longitud dataset {len(dataset)}")



# In[96]:


#Verificamos si hay visitas negativas, en búsqueda de un error 

dataset[dataset["views"]<0].head()


# In[97]:


print(dataset.columns)


# In[114]:


#Limpiamos los datos de texto

import re #Se importa la biblioteca re que proporciona funciones para trabajar con expresiones regulares en Python.

def clean(text): 
    text = re.sub(r'@[A-Z-z0-9]+', '', text) #Esto elimina menciones a usuarios en redes sociales. 
    #Busca patrones que comiencen con "@" seguido de letras mayúsculas minúsculas o números y los reemplaza con una 
    #cadena vacía, es decir, los elimina.
    
    text = re.sub(r'#', '', text) #Esto elimina los símbolos de numeral (#) en el texto.
    text = re.sub(r'https?:\/\/\S+', '', text) #Esto elimina las URLs en el texto. Busca patrones que comiencen con "http://" o "https://" 
    #seguido de cualquier secuencia de caracteres no espacios en blanco y los reemplaza con una cadena vacía.
    text = re.sub(r'[-?!¡¿&]', '', text)
    text = re.sub(r'["/|]', ' ', text)
    text = re.sub(r'[ft.]', '', text)
    #text = re.sub(r'[o]', '', text)
    #text = re.sub(r'[s]', '', text)
    prepositions = [
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "to",
    "from",
    "into",
    "during",
    "after",
    "before",
    "over",
    "under",
    "between",
    "among",
    "behind",
    "beside",
    "around",
    "through", "à", "après", "avant", "avec", "chez", "contre", "dans", "de", "depuis", "derrière",
    "devant", "en", "entre", "jusqu'à", "par", "pour", "sans", "sous", "sur", "vers", "le", "les", "des", "du", 
        "la", "the",
        "un", "une", "elle","lui","je","au","ils","il","elles", "a", "s", "he", "she", "i", "and","o"
]



    pattern = r'\b(?:' + '|'.join(map(re.escape, prepositions)) + r')\b'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    

    
    return text

#limpiamos los títulos
dataset["title"] = dataset["title"].apply(clean)
dataset[["title"]] = dataset[["title"]].applymap(str.lower)

dataset.head(40)


# In[115]:


#Vamos a unir las palabras de cada fila, perteneciente a tags y títulos comparando los resultados con las visitas

dataset['texto'] = dataset['title'] 


# In[116]:


import nltk
nltk.download('punkt')

# Convierte la serie 'texto' en una lista de cadenas de texto
text_list = dataset["texto"].tolist()

# Tokeniza las cadenas de texto en la lista
words = [nltk.word_tokenize(text) for text in text_list]


print(words[:20])



# In[117]:


dataset["words"]=words


# In[118]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Unir las listas en una sola lista de palabras
words_flat = [word for sublist in words for word in sublist]

# Convertir la lista de palabras en una cadena de texto
texto = ' '.join(words_flat)


# Configurar el objeto WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto)

# Mostrar el gráfico de nube de palabras
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Desactivar los ejes
plt.show()
        


# In[119]:


from collections import Counter

# Usar Counter para contar las repeticiones de cada palabra
contador = Counter(word for sublist in words for word in sublist)

# Mostrar las palabras más comunes y su conteo
palabras_mas_comunes = contador.most_common()

# Filtrar las palabras (elementos) que no sean números y conservar solo las palabras
palabras_filtradas = [(palabra, frecuencia) for palabra, frecuencia in palabras_mas_comunes if palabra.isalpha()]

umbral = 550

# Filtrar los datos para mantener solo aquellos mayores que el umbral
data_filtrados = [(palabra, frecuencia) for palabra, frecuencia in palabras_filtradas if frecuencia > umbral]

data_filtrados


# In[120]:


# Separar las palabras y las frecuencias en listas separadas
palabras, frecuencias = zip(*data_filtrados)

# Crear el gráfico de barras
plt.figure(figsize=(6, 12))
plt.barh(palabras, frecuencias, color='skyblue')
plt.xlabel('Frecuencia')
plt.ylabel('Palabra')
plt.title('Frecuencia de Palabras')
plt.gca().invert_yaxis()  # Invertir el eje y para mostrar las palabras más comunes en la parte superior
plt.show()


# In[121]:


#representamos en un diagrama circular la frecuencia y palabras más usadas.
#Haremos los mismo con una tabla 

tabla_palabras_destacadas = pd.DataFrame(data_filtrados)
col = ["Palabras", "Frecuencia"]
tabla_palabras_destacadas.columns = col
suma_total = tabla_palabras_destacadas["Frecuencia"].sum()
tabla_palabras_destacadas["Porcentaje"] = 100*(tabla_palabras_destacadas["Frecuencia"]/suma_total)

tabla_palabras_destacadas.head(20)


# In[122]:


# Inicializa un diccionario vacío
diccionario_nueva_lista = {}

# Usa un bucle para agregar cada par clave-valor al diccionario
for clave, valor in data_filtrados:
    diccionario_nueva_lista[clave] = valor

# El resultado es un diccionario
print(diccionario_nueva_lista)



# In[123]:


# Función para asignar visitas si una palabra está en palabras_destacadas
def asignar_visitas(words):
    visitas_asignadas = []
    for palabra in words:
        if palabra in diccionario_nueva_lista:
            visitas_asignadas.append((palabra, dataset['views'][dataset['words'].apply(lambda x: palabra in x)].values[0]))
    return visitas_asignadas

# Aplica la función a cada fila en el DataFrame
dataset['visitas_asignadas'] = dataset['words'].apply(asignar_visitas)

#Eliminamos las filas con un conjunto vacío 

dataset_visitas = dataset.copy()
dataset_visitas = dataset_visitas[dataset_visitas['visitas_asignadas'].apply(len) > 0]

# Restablecer el índice del DataFrame si es necesario
dataset_visitas = dataset_visitas.reset_index(drop=True)



dataset_visitas.head(20)


# In[124]:


# Extraer las visitas asignadas de tu DataFrame
visitas_asignadas = dataset_visitas['visitas_asignadas']

# Calcular el número total de visitas asignadas por palabra
total_visitas_por_palabra = {}
for visitas in visitas_asignadas:
    for palabra, visita in visitas:
        if palabra in total_visitas_por_palabra:
            total_visitas_por_palabra[palabra] += visita
        else:
            total_visitas_por_palabra[palabra] = visita

# Crear listas para las palabras y las visitas
palabras = list(total_visitas_por_palabra.keys())
visitas = list(total_visitas_por_palabra.values())

# Crear el gráfico de barras
plt.figure(figsize=(12, 6))
plt.bar(palabras, visitas)
plt.xlabel('Palabras')
plt.ylabel('Número de Visitas Asignadas')
plt.title('Número de Visitas Asignadas por Palabra')
plt.xticks(rotation=90)  # Rotar las etiquetas del eje x para mayor claridad

# Mostrar el gráfico
plt.tight_layout()
plt.show()


# In[125]:


# Calcular el recuento de palabras en el texto y agregarlo como una nueva columna 'word_count'

dataset['word_count'] = dataset['texto'].str.split().apply(len)

dataset.head()


# In[126]:


plt.plot(dataset["word_count"], dataset["views"], "v", color = "red")
etiqueta = np.arange(0,30,1)
plt.xticks(etiqueta)
plt.xticks(rotation = 90)
plt.xlabel("Número de palabras")
plt.ylabel("Número de visitas")
plt.title("Número de palabras/Visitas")


# In[ ]:




