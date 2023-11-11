import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from unidecode import unidecode
import re

import warnings
import nltk

# Ignorar las advertencias
warnings.filterwarnings('ignore')

# Descargar recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Ahora puedes usar las funciones de nltk en tu código
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

#Red neuronal
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Gráficos
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns

from collections import Counter
from wordcloud import WordCloud


ruta_Archivo = './train_new.txt'

df = pd.read_csv(ruta_Archivo,encoding='ISO-8859-1',sep='|')
df.columns = ["Lenguage", "Text"]

def limpiar_tokenizar(lang, texto):
    lemmatizer = WordNetLemmatizer()
    # Convertir a minúsculas
    RefactorText = texto.lower()
    # Remover caracteres de puntuación y otros caracteres no deseados
    regex = r'[^\w\s]'
    RefactorText = re.sub(regex, ' ', RefactorText)
    # Remover números
    RefactorText = re.sub("\d+", ' ', RefactorText)
    # Remover espacios adicionales
    RefactorText = re.sub("\\s+", ' ', RefactorText)
    # Eliminar acentos y caracteres especiales
    RefactorText = unidecode(RefactorText)
    # Remover fechas en formato "dd/mm/aaaa" y "dd/mm/aa"
    RefactorText = re.sub("\d+/\d+", ' ', RefactorText)
    RefactorText = re.sub("\d{1,2}/\d{1,2}/\d{2,4}", ' ', RefactorText)
    # Tokenizar el texto
    RefactorText = RefactorText.split(sep=' ')
    # Lematizar las palabras
    RefactorText = [lemmatizer.lemmatize(token) for token in RefactorText]
    # Filtrar palabras con longitud mayor a 1
    RefactorText = [token for token in RefactorText if len(token) > 1]
    return RefactorText

####################################################################################################

df['texto_tokenizado'] = df.apply(lambda x: limpiar_tokenizar(x['Lenguage'], x['Text']), axis=1)
df.to_csv('resultadoTokeniz.csv', index=False)

texto_tidy = df.explode(column='texto_tokenizado')
texto_tidy = texto_tidy.drop(columns='Text')
texto_tidy = texto_tidy.rename(columns={'texto_tokenizado': 'token'})
texto_tidy.head(10)

# Especificar la ruta del archivo CSV
ruta_archivo = r'./resultadoTokeniz.csv'

# Leer el archivo CSV
df = pd.read_csv(ruta_archivo, nrows=100000)

# Acceder a los datos de las columnas
freshness = df['Lenguage']
review = df['Text']
texto_tokenizado = df['texto_tokenizado']
df.head()

################################################################################################

#Histograma del atributo clase
ax=plt.subplots(1,1,figsize=(10,8))

sns.countplot(x='Lenguage', data=df)  # Cambia 'Idiomas' por 'Lenguage'
plt.title("Idiomas")
plt.show()

##################################################################################################

# Función para obtener las palabras más frecuentes por idioma
def palabras_mas_frecuentes_idioma(df, idioma_seleccionado):
    textos_por_idioma = df[df['Lenguage'] == idioma_seleccionado]['texto_tokenizado']
    todas_las_palabras = [word for sublist in textos_por_idioma for word in eval(sublist)]  # Convertir la cadena a lista
    contador_palabras = Counter(todas_las_palabras)
    return contador_palabras.most_common(10)

# Función para crear y mostrar la nube de palabras por idioma
def mostrar_nube_palabras(idioma_seleccionado, palabras_frecuentes):
    # Verificar si hay palabras después del preprocesamiento
    if palabras_frecuentes:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(palabras_frecuentes))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Nube de palabras en {idioma_seleccionado}')
        plt.show()
    else:
        print(f"No hay palabras para generar la nube de palabras en {idioma_seleccionado}.")

# Crear y mostrar la nube de palabras para inglés
idioma_seleccionado_en = 'en'
palabras_frecuentes_en = palabras_mas_frecuentes_idioma(df, idioma_seleccionado_en)
mostrar_nube_palabras(idioma_seleccionado_en, palabras_frecuentes_en)

# Crear y mostrar la nube de palabras para neerlandés
idioma_seleccionado_nl = 'nl'
palabras_frecuentes_nl = palabras_mas_frecuentes_idioma(df, idioma_seleccionado_nl)
mostrar_nube_palabras(idioma_seleccionado_nl, palabras_frecuentes_nl)

# Obtener las palabras más frecuentes por idioma
def palabras_mas_frecuentes_por_idioma(df):
    palabras_frecuentes_por_idioma = {}
    for idioma in df['Lenguage'].unique():
        textos_por_idioma = df[df['Lenguage'] == idioma]['texto_tokenizado']
        todas_las_palabras = [word for sublist in textos_por_idioma for word in eval(sublist)]  # Convertir la cadena a lista
        contador_palabras = Counter(todas_las_palabras)
        palabras_frecuentes_por_idioma[idioma] = contador_palabras.most_common(10)
    return palabras_frecuentes_por_idioma

# Obtener las palabras más frecuentes para cada idioma
palabras_frecuentes_por_idioma = palabras_mas_frecuentes_por_idioma(df)

# Imprimir las palabras más frecuentes para cada idioma
for idioma, palabras_frecuentes in palabras_frecuentes_por_idioma.items():
    print(f"Palabras más frecuentes en {idioma}: {[word[0] for word in palabras_frecuentes]}")


###########################################################################################################

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Lenguage'], test_size=0.2, random_state=42)

# Preprocesamiento adicional si es necesario (por ejemplo, vectorización de texto)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Codificación de las etiquetas
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Construir y entrenar el modelo MLP con la configuración específica
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    max_iter=200,
    momentum=0.9,
    random_state=42
)
mlp_classifier.fit(X_train_tfidf, y_train_encoded)

# Predicciones en el conjunto de prueba
y_pred = mlp_classifier.predict(X_test_tfidf)

# Evaluación del modelo
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test_encoded, y_pred))

