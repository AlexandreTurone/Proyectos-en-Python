#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Librerías

import pandas as pd 
import numpy as np
import random
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
classification_report, confusion_matrix,accuracy_score,
    fbeta_score,
    make_scorer,
    recall_score,)


# In[2]:


#Importamos los datasets

file_1 = "/Users/alexandremartinez/Desktop/Data_Science/Datasets/Registro_Ventas/archive/201904 sales reciepts.xlsx"
dataset_1 = pd.read_excel(file_1)

file_2 = "/Users/alexandremartinez/Desktop/Data_Science/Datasets/Registro_Ventas/archive/customer.xlsx"
dataset_2 = pd.read_excel(file_2)

file_3 = "/Users/alexandremartinez/Desktop/Data_Science/Datasets/Registro_Ventas/archive/pastry inventory.xlsx"
dataset_3 = pd.read_excel(file_3)

file_4 = "/Users/alexandremartinez/Desktop/Data_Science/Datasets/Registro_Ventas/archive/product.xlsx"
dataset_4 = pd.read_excel(file_4)

#Elegimos las columnas pertinentes para el estudio

dataset_1 = dataset_1[["transaction_date", "transaction_time", "customer_id","product_id", "quantity", "unit_price"]]
dataset_2 = dataset_2[["customer_id", "gender", "birth_year"]]
dataset_3 = dataset_3[["product_id", "waste", "quantity_sold"]]
dataset_4 = dataset_4[["product_id","product_category"]]


# In[3]:


#Observamos que los dataset no tienen las mismas longitudes así que tendremos 
#que abordar el problema desde otra perpectiva. 

dataset_1 = dataset_1.sort_values(by=["customer_id"])
dataset_1.head()


# In[4]:


dataset_2.head()


# In[5]:


dataset_3.head()


# In[6]:


dataset_4.head()


# In[7]:


#Unimos las cuatro tables según el Customer id y el Product id

#dataset_1 = dataset_1.merge(dataset_3, on="product_id", how="inner")

#dataset = dataset_1
dataset_2 = dataset_2.to_dict()
dataset_3 = dataset_3.to_dict()

bir = dataset_2["birth_year"]
gen = dataset_2["gender"]
was = dataset_3["waste"]
quan = dataset_3["quantity_sold"]
cat = dataset_4["product_category"]

dataset_1['gender'] = dataset_1['customer_id'].map(gen)
dataset_1['birth_year'] = dataset_1['customer_id'].map(bir)
dataset_1['waste'] = dataset_1['customer_id'].map(was)
dataset_1['quantity_sold'] = dataset_1['customer_id'].map(quan)
dataset_1["category_product"] = dataset_1['customer_id'].map(cat)
dataset_1["category_product"] = dataset_1['customer_id'].map(cat)


#Mejoramos el dataset

com_ven = []
precio = []
precio_venta = [0.3,0.35,0.40,0.45, 0.5]
for i in range(len(dataset_1)):
    precio.append(dataset_1["unit_price"].values[i] + (dataset_1["unit_price"].values[i] * np.random.choice(precio_venta)))

for i in range(len(dataset_1)):
    if dataset_1["quantity_sold"].values[i] > 0:
        com_ven.append("y")
    else:
        com_ven.append("n")


dataset_1["precio_venta"] = precio
dataset_1["Venta"] = com_ven

h = pd.to_datetime(dataset_1['transaction_time'], format='%H:%M:%S')
horas = h.dt.hour
dataset_1['transaction_date'] = dataset_1['transaction_date'].dt.strftime('%A')
dataset_1['transaction_time'] = horas

stock = []
for i in range(len(dataset_1)):
    stock.append(dataset_1["waste"].values[i] + dataset_1["quantity_sold"].values[i])

dataset_1["Stock"] = stock
gender = ["M", "F"]
age = np.arange(18,81)
q_sold = np.arange(0,32)
dataset_1["quantity_sold"] = (np.random.choice(q_sold,len(dataset_1)) +  np.random.uniform(0, 5, len(dataset_1)).astype(int))
dataset_1["gender"] = np.random.choice(gender, len(dataset_1))
dataset_1["birth_year"] = np.random.choice(age, len(dataset_1))
# Lista de productos
productos = ['Bakery', 'Branded','Coffee','Coffee beans','Drinking Chocolate','Flavours','Loose Tea','Packaged Chocolate','Tea']
# Pesos asignados a cada producto (pueden ser cualquier valor numérico)
pesos = [10, 80, 500, 400, 2000, 100, 90, 400, 70]
# Número de selecciones aleatorias que deseas realizar
num_selecciones = len(dataset_1)
# Realizar selecciones aleatorias ponderadas
dataset_1["category_product"] = random.choices(productos, weights=pesos, k=num_selecciones)



#dataset_1["category_product"] = np.random.choice(productos, len(dataset_1))


# In[8]:


#Nos deshacemos de los elementos NaN

dataset_1 = dataset_1.dropna(axis=0)


# In[9]:


#Una vez preparados los datos vemos una descripción del dataset

dataset_1.describe(include= 'all').style.background_gradient(cmap="summer")


# In[10]:


#Vemos si podemos optimizar el Stock de modo a poder tener un
#mayor beneficio sin tener que reponer tanto

categorias = set(dataset_1["category_product"].values)
categorias = list(categorias)
categorias = list(filter(lambda x: x is not np.nan, categorias))

reposicion_producto = dataset_1.groupby([ "category_product", "transaction_date"])["quantity_sold"].sum()
reposicion_stock = dataset_1.groupby([ "category_product", "transaction_date"])["Stock"].sum()

plt.figure(figsize=(15,15))
#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.5, wspace=0.4, hspace=0.4)

num_rows = 3
num_columns = 3
num_plots = num_columns * num_rows - 1

for i in range(num_plots):  # Start the index at 1
    plt.subplot(num_rows, num_columns, i+1)
    plt.scatter(list(set(dataset_1[dataset_1["category_product"] == categorias[i-1]]["transaction_date"])), reposicion_producto[categorias[i-1]], label=f"Venta diaria {categorias[i-1]}")
    plt.scatter(list(set(dataset_1[dataset_1["category_product"] == categorias[i-1]]["transaction_date"])), reposicion_stock[categorias[i-1]], label=f"Stock diaro {categorias[i-1]}", s = 10)
    plt.xticks(rotation=45)
    plt.title(f"{categorias[i-1]}")
    plt.tight_layout()
    plt.legend()
    plt.grid(True)

plt.show()


# In[11]:


### Ahora vamos a aplicar una mejora en la cantidad del stock

dias = sorted(set(dataset_1['transaction_date'].values))

#stock final = venta mensual media + stock actua
dataset_1["Nuevo_stock_total"] = np.zeros(len(dataset_1))


for k in range(len(dias)):
    
    ventas = dataset_1[dataset_1['transaction_date'] == dias[k]]
    ventas = ventas[["transaction_date", "quantity_sold", "Stock"]]
    ventas = ventas.dropna(axis=0)
    dif = abs(ventas["Stock"].values.mean() - ventas["quantity_sold"].values.mean())
    print("")
    print(f"La diferencia entre la media del stock y de las ventas el día {dias[k]} es de {dif}")
    print("Escriba la diferencia de stock óptimo")
    x = input()
    x = float(x)
    
    if dif < 2:
        while  x >= abs(ventas["Stock"].values.mean() - ventas["quantity_sold"].values.mean()):
    
            for i in range(len(ventas)):
                if ventas["Stock"].values[i] - ventas["quantity_sold"].values[i] < (x+0.5):
                    ventas["Stock"].values[i] = ventas["Stock"].values[i] + 1

                else: 
                    ventas["Stock"].values[i] = ventas["Stock"].values[i] - 1
    
            condicion = dataset_1["transaction_date"] == dias[k]
            dataset_1.loc[condicion, "Nuevo_stock_total"] = ventas["Stock"]
            
    else:
        
        while  abs(ventas["Stock"].values.mean() - ventas["quantity_sold"].values.mean()) >= x:
    
            for i in range(len(ventas)):
                if ventas["Stock"].values[i] - ventas["quantity_sold"].values[i] > (x+0.5):
                    ventas["Stock"].values[i] = ventas["Stock"].values[i] - 1

                else: 
                    ventas["Stock"].values[i] = ventas["Stock"].values[i] + 1
    
            condicion = dataset_1["transaction_date"] == dias[k]
            dataset_1.loc[condicion, "Nuevo_stock_total"] = ventas["Stock"]
             
print("")
print("------------------------------")
print("La optimización está lista")


# In[12]:


reposicion_stock_nueva = dataset_1.groupby(["category_product", "transaction_date"])["Nuevo_stock_total"].sum()

plt.figure(figsize=(15,15))

for i in range(num_plots):  # Start the index at 1
    plt.subplot(num_rows, num_columns, i+1)
    plt.scatter(list(set(dataset_1[dataset_1["category_product"] == categorias[i-1]]["transaction_date"])), reposicion_producto[categorias[i-1]], label=f"Venta diaria {categorias[i-1]}")
    plt.scatter(list(set(dataset_1[dataset_1["category_product"] == categorias[i-1]]["transaction_date"])), reposicion_stock[categorias[i-1]], label=f"Stock diario {categorias[i-1]}", s = 10)
    plt.scatter(list(set(dataset_1[dataset_1["category_product"] == categorias[i-1]]["transaction_date"])), reposicion_stock_nueva[categorias[i-1]], label=f"Nuevo Stock diario {categorias[i-1]}", s = 10)

    plt.xticks(rotation=45)
    plt.title(f"{categorias[i-1]}")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

plt.show()



# In[17]:


#Los siguiente que veremos es en qué horas hay un pico de compra

hor = sorted(set(dataset_1["transaction_time"].values))

hora = []
venta = []
for j in range(len(hor)):
    total_venta = []
    sub_dataset_1 = dataset_1[dataset_1["transaction_time"] == hor[j]][["category_product","quantity_sold", "precio_venta"]]
    for i in range(len(sub_dataset_1)):
        total_venta.append(sub_dataset_1["quantity_sold"].values[i]*sub_dataset_1["precio_venta"].values[i])
    venta.append(round(sum(total_venta)))
    hora.append(hor[j])
    
diagram_venta = {}

diagram_venta = {"horas": hora,
                    "venta_total":venta}
    
diagram = pd.DataFrame(diagram_venta)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.hist2d(diagram["horas"], diagram["venta_total"]/10000, bins=(10, 10), cmap='Blues')
plt.xlabel("Horas de venta")
plt.ylabel("Cantidad total vendida")
plt.title("Ventas por hora")
cbar = plt.colorbar(label='Frecuencia')

plt.subplot(1,2,2)
plt.scatter(diagram["horas"], diagram["venta_total"]/10000)
plt.xlabel("Horas de venta")
plt.ylabel("Cantidad total vendida")
plt.title("Ventas por hora")

plt.tight_layout()
plt.show



# In[18]:


#Vamos a crear un modelo que tenga me prediga si va a haber compra
#cierto día a cierta hora

def predict_compra(data, dia_semana_ingles,hora):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestRegressor
    import warnings
    import seaborn as sns
    
    warnings.filterwarnings("ignore")
    
    # Crear una copia del DataFrame original para evitar modificar los datos originales
    data_copy = data.copy()
    
    # Crear LabelEncoders para las variables categóricas "transaction_date" y "gender"
    encoder_dia = LabelEncoder()
    encoder_venta = LabelEncoder()

    
    # Codificar las variables categóricas en el DataFrame copiado
    data_copy["transaction_date_encoded"] = encoder_dia.fit_transform(data_copy["transaction_date"])
    data_copy["venta_encoded"] = encoder_venta.fit_transform(data_copy["Venta"])
    
    data_copy = data_copy.drop(["transaction_date", "gender", "category_product", "Venta" ], axis=1)
    #Correlation Matrix
    
    plt.figure(figsize=(20,10))
    corr_matrix = sns.heatmap(data_copy.corr(), cmap='summer', annot=True, square=True ) 
    plt.show()
    
    
    y = data_copy["venta_encoded"]
    X = data_copy[["transaction_date_encoded", "transaction_time"]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_KN = KNeighborsClassifier()
    model_KN.fit(X_train, y_train)
    model_KN.score(X_test, y_test)
    
    model_Random = RandomForestRegressor()
    model_Random.fit(X_train, y_train)
    
    print("")
    print(f"El modelo de random forest tiene un rendimiento de {model_Random.score(X_test, y_test)} ")
    print(f"El modelo de KNeighbors tiene un rendimiento de {model_KN.score(X_test, y_test)} ")
    print("")
    
    score_Random = model_Random.score(X_test, y_test)
    score_KN = model_KN.score(X_test, y_test)

    if score_Random < score_KN:
        
        #Vamos a tratar de mejorar nuestro modelo
        from sklearn.model_selection import GridSearchCV
        
        model = KNeighborsClassifier()
       
        param_grid = {
            "n_neighbors": np.arange(1,10,1),
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
        }
        
         # Usar GridSearchCV para una búsqueda exhaustiva
        grid_search = GridSearchCV(model, param_grid, cv=5)
        



        #Evaluar el modelo en los datos de prueba

        grid_search.fit(X_train, y_train)
    
        model = KNeighborsClassifier(n_neighbors=grid_search.best_params_["n_neighbors"], 
                                 algorithm=grid_search.best_params_["algorithm"])
        model.fit(X_train, y_train)
    
        print("--------------------")
        print("")
    
   
        print("--------------------")
        print("")
    
        print(f"La precisión del modelo en el grupo test es de {model.score(X_test, y_test)}")
        print("")
        print("--------------------")

        # Codificar las variables de entrada para la predicción
        day = encoder_dia.transform([dia_semana_ingles])
    
        # Realizar la predicción utilizando las variables codificadas
        X = np.array([day, hora]).reshape(-1, 2)
        prediccion = model.predict(X)
        prediccion_decodificada = encoder_venta.inverse_transform(prediccion)
        
        print("")
        print("--------------------")
        print("Evaluación del modelo")
        
        sns.heatmap(confusion_matrix(y_test, model.predict(X_test)),
                   annot=True,
                   fmt="g",
                   cbar=False,
                   cmap="Greens", 
                   annot_kws={"size":15})
        
        plt.title("Confusion matrix (Test set)")
        plt.xlabel("Predict")
        plt.ylabel("Values")
        plt.show()
        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Precisión: {round(accuracy,2)}")
        print(classification_report(y_test, model.predict(X_test)))
    
        

    
        # Invertir la codificación para obtener la predicción en su forma original
    
        print(f"La cantidad que se venderá el {dia_semana_ingles} a las {hora} es {prediccion_decodificada}")
    
    else:
            
        model = RandomForestRegressor()
        
        #Vamos a tratar de mejorar nuestro modelo
        from sklearn.model_selection import GridSearchCV
    
        param_grid = {
            "n_estimators": np.arange(100,500,10)
                
        }

            
        # Usar GridSearchCV para una búsqueda exhaustiva
        grid_search = GridSearchCV(model, param_grid, cv=5)
        
        


        #Evaluar el modelo en los datos de prueba

        grid_search.fit(X_train, y_train)
    
        model = RandomForestRegressor(n_estimators=grid_search.best_params_["n_estimators"])
        model.fit(X_train, y_train)
    
        print("--------------------")
        print("")
        
   
        print("--------------------")
        print("")
    
        print(f"La precisión del modelo en el grupo test es de {model.score(X_test, y_test)}")
        print("")
        print("--------------------")

        # Codificar las variables de entrada para la predicción
        day = encoder_dia.transform([dia_semana_ingles])

        # Realizar la predicción utilizando las variables codificadas
        X = np.array([day, hora]).reshape(-1, 2)
        prediccion = model.predict(X)
        prediccion_decodificada = encoder_venta.inverse_transform(prediccion)

    
    
        # Invertir la codificación para obtener la predicción en su forma original
    
        print(f"¿Se venderá {dia_semana_ingles} a las {hora}? {prediccion_decodificada}")
    
    
    return prediccion_decodificada


# In[20]:


predict_compra(dataset_1, "Monday", 7)


# In[21]:


#A continuación realizamos una funcion análisis y prediccion

def predict_producto(data, dia_semana_ingles,hora, edad, genero):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.neighbors import KNeighborsClassifier
    import warnings
    import seaborn as sns
    
    warnings.filterwarnings("ignore")
    
    # Crear una copia del DataFrame original para evitar modificar los datos originales
    data_copy_1 = data.copy()
    
    # Crear LabelEncoders para las variables categóricas "transaction_date" y "gender"
    encoder_dia = LabelEncoder()
    encoder_genero = LabelEncoder()
    encoder_categoria = LabelEncoder()

    
    # Codificar las variables categóricas en el DataFrame copiado
    data_copy_1["transaction_date_encoded"] = encoder_dia.fit_transform(data_copy_1["transaction_date"])
    data_copy_1["gender_encoded"] = encoder_genero.fit_transform(data_copy_1["gender"])
    data_copy_1["category_encoded"] = encoder_categoria.fit_transform(data_copy_1["category_product"])

    data_copy_1 = data_copy_1.drop(["transaction_date", "gender", "category_product" ], axis=1)
    #Correlation Matrix
    
    
    
    y = data_copy_1["category_encoded"]
    X = data_copy_1[["transaction_date_encoded", "birth_year", "gender_encoded", "transaction_time"]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #Vamos a tratar de mejorar nuestro modelo
    from sklearn.model_selection import GridSearchCV




    param_grid = {
        "n_neighbors": np.arange(1,10,1),
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
    }


    # Usar GridSearchCV para una búsqueda exhaustiva
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)


    #Evaluar el modelo en los datos de prueba

    grid_search.fit(X_train, y_train)
    
    model = KNeighborsClassifier(n_neighbors=grid_search.best_params_["n_neighbors"], 
                                 algorithm=grid_search.best_params_["algorithm"])
    model.fit(X_train, y_train)
    
    print("--------------------")
    print("")
    
   
    print("--------------------")
    print("")
    
    print(f"La precisión del modelo en el grupo test es de {model.score(X_test, y_test)}")
    print("")
    print("--------------------")

    # Codificar las variables de entrada para la predicción
    gen = encoder_genero.transform([genero])
    day = encoder_dia.transform([dia_semana_ingles])
    
    # Realizar la predicción utilizando las variables codificadas
    X = np.array([day, edad, gen, hora]).reshape(-1, 4)
    prediccion = model.predict(X)
    
    
    # Invertir la codificación para obtener la predicción en su forma original
    prediccion_decodificada = encoder_categoria.inverse_transform(prediccion)
    
    print(f"El producto que consumirá una persona de género {genero}, edad {edad}, el {dia_semana_ingles} a las {hora} es {prediccion_decodificada}")
    
    return prediccion_decodificada


# In[22]:


#Convertimos las variables categóricas en numéricas (enteros)

prediccion = predict_producto(dataset_1, "Monday", 7 , 25, "F" )


# In[ ]:




