import re
import time
import nltk
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from nltk import pos_tag
from unidecode import unidecode
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
import matplotlib.style as style
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import cosine
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

rt_reviews = "train_new.txt"
df = pd.read_csv(rt_reviews, encoding='ISO-8859-1', sep="|")
df.columns = ["Lenguage", "Text"]

def limpiar_tokenizar(lang, texto):

    lemmatizer = WordNetLemmatizer()
    RefactorText = texto.lower()
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    RefactorText = re.sub(regex, ' ', RefactorText)
    RefactorText = re.sub("\d+", ' ', RefactorText)
    RefactorText = re.sub("\\s+", ' ', RefactorText)
    RefactorText = unidecode(RefactorText)
    RefactorText = re.sub("\d+/\d+", ' ', RefactorText)
    RefactorText = re.sub("\d{1,2}/\d{1,2}/\d{2,4}", ' ', RefactorText)
    RefactorText = RefactorText.split(sep=' ')
    RefactorText = [lemmatizer.lemmatize(token) for token in RefactorText]
    RefactorText = [token for token in RefactorText if len(token) > 1]

    return RefactorText

df['texto_tokenizado'] = df.apply(lambda x: limpiar_tokenizar(x['Lenguage'], x['Text']), axis=1)
df.to_csv(
    'resultadoTokeniz.csv', index=False)

texto_tidy = df.explode(column='texto_tokenizado')
texto_tidy = texto_tidy.drop(columns='Text')
texto_tidy = texto_tidy.rename(columns={'texto_tokenizado': 'token'})
texto_tidy.head(10)

# Especificar la ruta del archivo CSV
ruta_archivo = r'.\resultadoTokeniz.csv'

# Leer el archivo CSV
df = pd.read_csv(ruta_archivo, nrows=100000)

# Acceder a los datos de las columnas
freshness = df['Lenguage']
review = df['Text']
texto_tokenizado = df['texto_tokenizado']
df.head()

# hallando xtrain, xtext, ytrain, ytest

datos_X = df['texto_tokenizado']
datos_y = df['Lenguage']

print(datos_X.unique())
print(datos_y.unique())


X_train, X_test, y_train, y_test = train_test_split(
    datos_X,
    datos_y,
    test_size=0.80,
    random_state=42
)

vectorizer = CountVectorizer()

# Ajustar el vectorizador con los datos de entrenamiento
Xtrain = vectorizer.fit_transform(X_train)

# Transformar los datos de prueba utilizando el vectorizador ajustado
Xtest = vectorizer.transform(X_test)

print(Xtrain.shape)

print("Conjunto de entrenamiento - Características:", Xtrain.shape)
print("Conjunto de entrenamiento - Etiquetas:", y_train.shape)
print("Conjunto de prueba - Características:", Xtest.shape)
print("Conjunto de prueba - Etiquetas:", y_test.shape)

# implementacion de random forest

modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)

print("Dimensiones entreamiento :", Xtrain.shape)
print("Dimensiones prueba ", y_train.shape)

rf_Model = RandomForestClassifier()
cross_val_scores = cross_val_score(rf_Model, Xtrain, y_train, cv=2)
rf_Model.fit(Xtrain, y_train)
train_accuracy = rf_Model.score(Xtrain, y_train) * 100
test_accuracy = rf_Model.score(Xtest, y_test) * 100

print("Precisión del entrenamiento:", "{:.3f}%".format(train_accuracy))
print("Precisión de prueba:", "{:.3f}%".format(test_accuracy))


print("Resultados de la validación cruzada:")
print(cross_val_scores)
print("Precisión media: {:.2f}".format(cross_val_scores.mean()))

# Crear el modelo de Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)

print("Dimensiones entreamiento :", Xtrain.shape)
print("Dimensiones prueba ", y_train.shape)

# Número de árboles en el bosque aleatorio
n_estimators = [int(x) for x in np.linspace(start=150, stop=400, num=4)]
# Número de características a considerar en cada división
max_features = ['auto', 'sqrt']
# Número máximo de niveles en el árbol
max_depth = None,
# Número mínimo de muestras requeridas para dividir un nodo
min_samples_split = [5, 10]
# Número mínimo de muestras requeridas en cada nodo hoja
min_samples_leaf = [1, 2]
# Método de selección de muestras para entrenar cada árbol

param_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              }
print(param_grid)

rf_Grid = GridSearchCV(
    estimator=rf_Model, param_grid=param_grid, cv=2, verbose=2, n_jobs=4)
rf_Grid.fit(Xtrain, y_train)


train_accuracy = rf_Grid.score(Xtrain, y_train) * 100
test_accuracy = rf_Grid.score(Xtest, y_test) * 100
print("Precisión del entrenamiento:", "{:.3f}%".format(train_accuracy))
print("Precisión de prueba:", "{:.3f}%".format(test_accuracy))

# Crear el modelo de Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)

print("Dimensiones :", Xtrain.shape)
print("Dimensiones ", y_train.shape)


# Número de árboles en el bosque aleatorio
n_estimators = [int(x) for x in np.linspace(start=100, stop=150, num=3)]
# Número de características a considerar en cada división
max_features = ['sqrt', 'sqrt']
# Número máximo de niveles en el árbol
max_depth = [2, 4]
# Número mínimo de muestras requeridas para dividir un nodo
min_samples_split = [2, 5]
# Número mínimo de muestras requeridas en cada nodo hoja
min_samples_leaf = [1, 2]
param_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
}

print(param_grid)


rf_Grid = GridSearchCV(
    estimator=rf_Model, param_grid=param_grid, cv=2, verbose=2, n_jobs=4)
rf_Grid.fit(Xtrain, y_train)

rf_Grid.best_params_
{'max_depth': 4,
 'max_features': 'auto',
 'min_samples_leaf': 1,
 'min_samples_split': 5,
 'n_estimators': 150}
# Evaluar el rendimiento del modelo en los datos de entrenamiento y prueba
train_accuracy = rf_Grid.score(Xtrain, y_train) * 100
test_accuracy = rf_Grid.score(Xtest, y_test) * 100
print("Precisión del entrenamiento:", "{:.3f}%".format(train_accuracy))
print("Precisión de prueba:", "{:.3f}%".format(test_accuracy))


param_sets = [
    {'n_estimators': 30, 'min_samples_split': 15,
        'min_samples_leaf': 15, 'max_depth': 28, 'max_features': 'sqrt'},
]

for i, params in enumerate(param_sets):
    start_time = time.time()  # Registrar el tiempo de inicio

    # Crear una instancia del modelo RandomForestClassifier
    modelo_rf = RandomForestClassifier(**params)

    # Entrenar el modelo con los parámetros actuales

    modelo_rf.fit(Xtrain, y_train)

    # Evaluar el rendimiento del modelo (por ejemplo, calcular la precisión)
    train_accuracy = modelo_rf.score(Xtrain, y_train) * 100
    test_accuracy = modelo_rf.score(Xtest, y_test) * 100

    end_time = time.time()  # Registrar el tiempo de finalización
    elapsed_time = end_time - start_time  # Calcular el tiempo transcurrido

    # Redondear el tiempo transcurrido a dos decimales
    elapsed_time = round(elapsed_time, 2)

    # Imprimir los resultados con el número de prueba, tiempo de ejecución y precisión en %
    print("Número de prueba:", i+1)
    print("Parámetros:", params)
    print("Precisión de entrenamiento:", "{:.2f}%".format(train_accuracy))
    print("Precisión de prueba:", "{:.2f}%".format(test_accuracy))
    print("Tiempo de ejecución :", elapsed_time, "segundos")
    print("--------------------------------------")

    # === Arboles de decision ===

    # Entrenar el clasificador de árboles de decisión
    clf = DecisionTreeClassifier()
    clf.fit(Xtrain,y_train)

    # Realizar predicciones en el conjunto de prueba
    predictions = clf.predict(Xtest)

    # Evaluar el rendimiento del clasificador
    accuracy = accuracy_score(y_test, predictions)
    print(f'Precisión en el conjunto de prueba: {accuracy}')

    # Mostrar el informe de clasificación
    print('\nInforme de clasificación:')
    print(classification_report(y_test, predictions))

    # === ******************* ===

    # Calcular la matriz de confusión
    y_pred = modelo_rf.predict(Xtest)
    cm = confusion_matrix(y_test, y_pred)

    labels = np.unique(y_test)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')
    plt.title("Matriz de Confusión")
    plt.xlabel("Etiqueta Predicha")
    plt.ylabel("Etiqueta Verdadera")
    plt.show()
