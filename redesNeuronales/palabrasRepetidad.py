
# Obtener las palabras más frecuentes por idioma
def palabras_mas_frecuentes_por_idioma(df):
    palabras_frecuentes_por_idioma = {}
    for idioma in df['Lenguage'].unique():
        textos_por_idioma = df[df['Lenguage'] == idioma]['texto_tokenizado']
        todas_las_palabras = [word for sublist in textos_por_idioma for word in sublist]
        contador_palabras = Counter(todas_las_palabras)
        palabras_frecuentes_por_idioma[idioma] = contador_palabras.most_common(10)
    return palabras_frecuentes_por_idioma

palabras_frecuentes = palabras_mas_frecuentes_por_idioma(df)

# Visualizar las palabras más frecuentes por idioma
for idioma, palabras in palabras_frecuentes.items():
    print(f"Palabras más frecuentes en {idioma}: {palabras}")

# Crear una nube de palabras para un idioma específico (puedes personalizar según tus necesidades)
idioma_seleccionado = 'español'  # Cambia al idioma que desees analizar
textos_por_idioma = df[df['Lenguage'] == idioma_seleccionado]['texto_tokenizado']
todas_las_palabras = ' '.join([word for sublist in textos_por_idioma for word in sublist])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(todas_las_palabras)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(f'Nube de palabras en {idioma_seleccionado}')
plt.show()