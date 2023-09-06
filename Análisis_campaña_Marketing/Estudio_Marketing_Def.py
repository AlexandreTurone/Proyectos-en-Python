#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns



url = "/Users/alexandremartinez/Desktop/Data_Science/Datasets/Marketing_Camp_dataset/marketing_campaign.csv"

dataset = pd.read_csv(url, sep=";")

dataset.head()


# In[34]:


#Elegimos qué columnas eliminaremos

columnas_elimar = ["ID", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5",
                   "AcceptedCmp1", "AcceptedCmp2", "Z_CostContact", "Z_Revenue", "Response", "NumWebVisitsMonth",
                  "Kidhome", "Teenhome"]


dataset = dataset.drop(columnas_elimar, axis=1)

#Vamos a unir las columnas de gastos en diferentes productos y reemplazarlas por el total de gastos

total = dataset["MntWines"].values + dataset["MntFruits"].values + dataset["MntMeatProducts"].values + dataset["MntFishProducts"].values + dataset["MntSweetProducts"].values + dataset["MntGoldProds"].values

columnas_eliminar = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]

dataset = dataset.drop(columnas_eliminar, axis=1)

dataset["total_gastado"] = total


dataset.head()



# In[35]:


#Vamos a responder a la pregunta
#¿Qué clientes gastarán más? 
#Lo primero será crear un subdataset con las columnas que nos interesan 

dataset_1 = dataset.copy()

dataset_1 = dataset_1.drop(["Dt_Customer", "Recency", "NumDealsPurchases", "NumWebPurchases",
                            "NumCatalogPurchases","NumStorePurchases", "Complain"], axis=1)

dataset_1.head()


# In[36]:


#Limpiamos los datos, de modo a eliminar las filas nulas 

dataset_1 = dataset_1.dropna(axis=0)
dataset_1 = dataset_1.reset_index(drop=True)

#Cambiamos la columna de Años de nacimiento por edades, ingresos anuales por mensuales y gasto bimensual por mensual

dataset_1["Year_Birth"] = (2023 - dataset_1["Year_Birth"].values)
dataset_1["Income"] = dataset_1["Income"].values/12
dataset_1["total_gastado"] = dataset_1["total_gastado"].values/24




# In[37]:


#Cambiamos la categoría educacion y status marital por equivalencias numéricas

def dummy(data, edu, mari):
    
    import pandas as pd
    
    dummy_edu = pd.get_dummies(data[edu], prefix="Edu")
    dummy_marital = pd.get_dummies(data[mari], prefix="Marital")

    return dummy_edu, dummy_marital


dummy_edu, dummy_marital = dummy(dataset_1, "Education", "Marital_Status")

dataset_1 = dataset_1.drop(["Education", "Marital_Status"], axis=1)
dataset_1 = pd.concat([dataset_1, dummy_edu, dummy_marital], axis=1)

dataset_1.head()


# In[38]:


def minmax( income, gastado, edad):
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    scaler_income = scaler.fit_transform(income.values.reshape(-1,1))
    scaler_gastado = scaler.fit_transform(gastado.values.reshape(-1,1))
    scaler_edad = scaler.fit_transform(edad.values.reshape(-1,1))
    
    return scaler_income, scaler_gastado, scaler_edad


scaler_income, scaler_gastado, scaler_edad = minmax(dataset_1["Income"], dataset_1["total_gastado"], 
                                                    dataset_1["Year_Birth"] )

dataset_1["edad_escalada"] = scaler_edad
dataset_1["income_escalado"] = scaler_income
dataset_1["gasto_escalado"] = scaler_gastado

dataset_1.head()


# In[39]:


#Vamos a ver si el desescalado funciona adecuadamente

def minmax_inv(edad, income, gastado, original_edad, original_income, original_gastado):
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    
    edad_ori = scaler.fit(original_edad.values.reshape(-1,1))
    edad_inv = edad_ori.inverse_transform(edad)
    
    income_ori = scaler.fit(original_income.values.reshape(-1,1))
    income_inv = income_ori.inverse_transform(income)
    
    gastado_ori = scaler.fit(original_gastado.values.reshape(-1,1))
    gasto_inv = gastado_ori.inverse_transform(gastado)
    
    return edad_inv,income_inv, gasto_inv



# Convierte cada lista en un array bidimensional
#edad_inv = np.array(edad_inv).reshape(-1, 1)
#gasto_inv = np.array(gasto_inv).reshape(-1, 1)
#income_inv = np.array(income_inv).reshape(-1, 1)

# Crea un DataFrame a partir de estas tres columnas
#M = pd.DataFrame({'Edad Invertida': edad_inv[:, 0], 'Gasto Invertido': gasto_inv[:, 0], 'Ingreso Invertido': income_inv[:, 0]})

#M.head()

#Verificamos los Outliers

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.suptitle("Dataset con Outliers", fontsize=16)

plt.subplot(1,3,1)
sns.boxplot(dataset_1["edad_escalada"])
plt.title("Edad escalada")


plt.subplot(1,3,2)
sns.boxplot(dataset_1["income_escalado"])
plt.title("Ingreso escaldo")


plt.subplot(1,3,3)
sns.boxplot(dataset_1["gasto_escalado"])
plt.title("Gasto escalado")

#Para eliminar los Outliers, vamos a definir cada IQR 
#Visto que están todos al mismo nivel, elegimos las edades que es el que más se va

print("Longitud inicial del dataset:", len(dataset_1))

def IQR(data, string_data ):
    q1 = data[string_data].quantile(0.25)
    q3 = data[string_data].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    return lower_bound, upper_bound

lower_bound, upper_bound = IQR(dataset_1, "edad_escalada")

dataset_1 = dataset_1[(dataset_1['edad_escalada'] >= lower_bound) & (dataset_1['edad_escalada'] <= upper_bound)]
dataset_1 = dataset_1[(dataset_1['income_escalado'] <= 0.64) & (dataset_1['gasto_escalado'] <= 0.64)]

print("Longitud final del dataset:", len(dataset_1))



plt.figure(figsize=(12,6))
plt.suptitle("Dataset sin Outliers", fontsize=16)

plt.subplot(1,3,1)
sns.boxplot(dataset_1["edad_escalada"])
plt.title("Edad escalada")


plt.subplot(1,3,2)
sns.boxplot(dataset_1["income_escalado"])
plt.title("Ingreso escaldo")


plt.subplot(1,3,3)
sns.boxplot(dataset_1["gasto_escalado"])
plt.title("Gasto escalado")






# In[40]:


#La variable a predecir es "Los gastos"
#Mis columnas de interes serán todas las demás 

y = dataset_1["gasto_escalado"]
X = dataset_1.drop(["gasto_escalado","Year_Birth","Income","total_gastado"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)


def modelos(xtrain, ytrain, xtest, ytest):
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import LinearRegression
    

    
    model_Random = RandomForestRegressor()
    model_Random.fit(X_train, y_train) 
    
    model_SVR = SVR()
    model_SVR.fit(X_train, y_train)
    
    model_DTR = DecisionTreeRegressor()
    model_DTR.fit(X_train, y_train)
    
    model_Linear = LinearRegression()
    model_Linear.fit(X_train, y_train)
    
    model_Ridge = Ridge()
    model_Ridge.fit(X_train, y_train)
    
    
    print("Train score Random:", model_Random.score(X_train, y_train))
    print("Test score Random:", model_Random.score(X_test, y_test))
    
    print("______________________")
    
    print("Train score SVR:", model_SVR.score(X_train, y_train))
    print("Test score SVR:", model_SVR.score(X_test, y_test))
    
        
    print("______________________")

        
    print("Train score Linear:", model_Linear.score(X_train, y_train))
    print("Test score Linear:", model_Linear.score(X_test, y_test))
    
    print("______________________")

    
    print("Train score DTR:", model_DTR.score(X_train, y_train))
    print("Test score DTR:", model_DTR.score(X_test, y_test))
    
    print("______________________")

    
    print("Train score Ridge:", model_Ridge.score(X_train, y_train))
    print("Test score Ridge:", model_Ridge.score(X_test, y_test))
    
    
    return model_DTR, model_Linear, model_Random, model_Ridge, model_SVR


model_DTR, model_Linear, model_Random, model_Ridge, model_SVR = modelos(X_train, y_train, X_test, y_test)



# In[41]:


#Vamos a tratar de mejorar nuestro modelo
from sklearn.model_selection import GridSearchCV
import multiprocessing


# Configurar GridSearchCV o RandomizedSearchCV con paralelización
n_jobs = multiprocessing.cpu_count()  # Utiliza todos los núcleos de CPU disponibles


param_grid = {
    "n_estimators": np.arange(100,500,5)

    }


# Usar GridSearchCV para una búsqueda exhaustiva
grid_search = GridSearchCV(model_Random, param_grid, cv=5, n_jobs=n_jobs)


#Evaluar el modelo en los datos de prueba

#grid_search.fit(X_train, y_train)
#score = grid_search.score(X_test, y_test)

#print("Mejores hiperparámetros encontrados:", grid_search.best_params_)
#print("Puntaje en datos de prueba:", score)


# In[42]:


from sklearn.ensemble import RandomForestRegressor

model_Random_def = RandomForestRegressor(n_estimators=195)
model_Random_def.fit(X_train, y_train)

print("Train score Random:", model_Random_def.score(X_train, y_train))
print("Test score Random:", model_Random_def.score(X_test, y_test))


# In[44]:


import seaborn as sns

sns.scatterplot(x=X_test["income_escalado"], y=y_test, label="Valor teórico")
sns.scatterplot(x=X_test["income_escalado"], y=model_Random_def.predict(X_test), label="Predicción")



# In[12]:


# Calcular las predicciones del modelo
predicciones = model_Random_def.predict(X_test)

# Calcular los errores
errores = y_test - predicciones

plt.scatter(y_test, errores, color='blue', alpha=0.5)
plt.xlabel('Valores Teóricos')
plt.ylabel('Errores')
plt.title('Gráfico de Dispersión de Errores')
plt.grid(True)
plt.show()


# In[13]:


dataset_1.head()


# In[14]:


#Por ejemplo, para seleccionar el 30% de las filas de manera aleatoria:
porcentaje_muestra = 0.1  # 30% de las filas
muestra_aleatoria = dataset_1.sample(frac=porcentaje_muestra, random_state=42)

def predict(model,data):
    
    lista_predict = []
    
    for i in range(len(data)):
        X = data.drop(["Year_Birth","Income","total_gastado", "gasto_escalado"], axis=1)
        lista_predict.append([data["Edu_2n Cycle"].values[i], data["Edu_Basic"].values[i],
                             data["Edu_Graduation"].values[i], data["Edu_Master"].values[i],
                              data["Edu_PhD"].values[i], data["Marital_Absurd"].values[i],
                             data["Marital_Alone"].values[i], data["Marital_Divorced"].values[i],
                             data["Marital_Married"].values[i], data["Marital_Single"].values[i], 
                             data["Marital_Together"].values[i], data["Marital_Widow"].values[i],
                             data["Marital_YOLO"].values[i], data["edad_escalada"].values[i],
                             data["income_escalado"].values[i],model.predict(X)[i]])
        
    return lista_predict


import warnings

# Desactivar las advertencias de sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

lista_pred = predict(model_Random_def,muestra_aleatoria)

dataset_predicho = pd.DataFrame(lista_pred, columns=["Edu_2n Cycle","Edu_Basic","Edu_Graduation",
                                                     "Edu_Master","Edu_PhD","Marital_Absurd","Marital_Alone",
                                                     "Marital_Divorced","Marital_Married",
                                                     "Marital_Single","Marital_Together",
                                                     "Marital_Widow","Marital_YOLO","edad_escalada",
                                                     "income_escalado", "Gasto_predicho"])

dataset_predicho.head(5)


# In[15]:


muestra_aleatoria["Income"].values.reshape(-1,1).shape


# In[16]:


scaler = MinMaxScaler()
edad_ori = scaler.fit(muestra_aleatoria["Year_Birth"].values.reshape(-1,1))
edad_inv = edad_ori.inverse_transform(dataset_predicho["edad_escalada"].values.reshape(-1,1))
income_ori = scaler.fit(muestra_aleatoria["Income"].values.reshape(-1,1))
income_inv = income_ori.inverse_transform(dataset_predicho["income_escalado"].values.reshape(-1,1))
gastado_ori = scaler.fit(muestra_aleatoria["total_gastado"].values.reshape(-1,1))
gastado_inv = gastado_ori.inverse_transform(dataset_predicho["Gasto_predicho"].values.reshape(-1,1))

#Convierte cada lista en un array bidimensional
edad_inv = np.array(edad_inv).reshape(-1, 1)
gastado_inv = np.array(gastado_inv).reshape(-1, 1)
income_inv = np.array(income_inv).reshape(-1, 1)

dataset_predicho["edad_escalada"] = edad_inv[:, 0]
dataset_predicho["income_escalado"] = income_inv[:, 0]
dataset_predicho["Gasto_predicho"] = gastado_inv[:, 0]

print("La media de gastos es:", dataset_predicho["Gasto_predicho"].mean())
print("Lo máximo gastado es:", max(dataset_predicho["Gasto_predicho"].values))

#len(dataset_predicho[dataset_predicho["Gasto_predicho"]>=50])

dataset_resumen = dataset_predicho[dataset_predicho["Gasto_predicho"]>=20]


# In[17]:


dataset_resumen.describe()


# In[18]:


#Sabiendo que las columnas con media 0 no van a tener gastos


columnas = ['Edu_2n Cycle', 'Edu_Graduation', 'Edu_Master', 'Edu_PhD',
        'Marital_Divorced',
       'Marital_Married', 'Marital_Single', 'Marital_Together',
       'Marital_Widow']

gasto_total = []

for i in range(len(columnas)):
    for j in range(len(dataset_resumen["Gasto_predicho"].values)):
        if dataset_resumen[columnas[i]].values[j] == 1:
            gasto_total.append(dataset_resumen["Gasto_predicho"].values[j])
            
        
        
        
    
            
            
    print("")
    print(f"Los gastos generados por la columna {columnas[i]} son {round(sum(gasto_total))}")
    print("")

            


# In[19]:


#Quién es más probable que vuelva a comprar ?

dataset_2 = dataset.drop(["Dt_Customer", "NumDealsPurchases", "NumWebPurchases",
                            "NumCatalogPurchases","NumStorePurchases", "Complain"], axis=1)
dataset_2.head()


# In[20]:


dataset_2["Year_Birth"] = (2023 - dataset_2["Year_Birth"].values)
dataset_2["Income"] = dataset_2["Income"].values/12
dataset_2["total_gastado"] = dataset_2["total_gastado"].values/24

scaler_income, scaler_gastado, scaler_edad = minmax(dataset_2["Income"], dataset_2["total_gastado"], 
                                                    dataset_2["Year_Birth"] )

dataset_2["edad_escalada"] = scaler_edad
dataset_2["income_escalado"] = scaler_income
dataset_2["gasto_escalado"] = scaler_gastado



# In[21]:


#Supongamos que nuestro público objetivo son los menores de 60 años

print("Longitud del dataset antes del filtro:", len(dataset_2))

dataset_2 = dataset_2[dataset_2["Year_Birth"]<60]

print("Longitud del dataset después del filtro:", len(dataset_2))


# In[22]:


dataset_2.head()


# In[23]:


rec = dataset_2["Recency"].value_counts()

#creamos un subdataset con las recency menores o iguales a la media 

data_rec = dataset_2[dataset_2["Recency"]<=rec.mean()]
data_rec = data_rec.dropna(axis=0)



# In[30]:


#Contamos las veces que se repite cada elemento
recency = data_rec["Recency"].value_counts()
education = data_rec["Education"].value_counts()
marital = data_rec["Marital_Status"].value_counts()
edad = data_rec["Year_Birth"].value_counts()

# Agrupamos por categoría y calculamos la media de 'Recency'
education_recency_mean = data_rec.groupby('Education')['Recency'].mean()
marital_recency_mean = data_rec.groupby('Marital_Status')['Recency'].mean()
edad_recency_mean = data_rec.groupby('Year_Birth')['Recency'].mean()


# Encontramos el nivel de estudio, estado civil y edad con la menor media de 'Recency'
nivel_estudio_menor_recency = education_recency_mean.idxmin()
estado_civil_menor_recency = marital_recency_mean.idxmin()
edad_menor_recency = edad_recency_mean.idxmin()


data_resumen = pd.concat([education_recency_mean,marital_recency_mean, edad_recency_mean], axis=1)
data_resumen.fillna(0, inplace=True)

data_resumen.head()

# Imprimir el resultado
print(f"El nivel de estudio con la menor Recency es: {nivel_estudio_menor_recency}")
print(f"El estado civil con la menor Recency es: {estado_civil_menor_recency}")
print(f"La edad con la menor Recency es: {edad_menor_recency}")





# In[32]:


data_resumen.head(200)

