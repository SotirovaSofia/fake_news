import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


#subo el dataset desde git
df_fake = pd.read_csv("https://raw.githubusercontent.com/SotirovaSofia/fake_news/refs/heads/main/onlyfakes1000.csv")
df_true = pd.read_csv("https://raw.githubusercontent.com/SotirovaSofia/fake_news/refs/heads/main/onlytrue1000.csv")

print(df_fake.head())
print(df_true.head())


# primeros pasos EDA que de poco sirve porque es texto y solo es una columna

print(df_fake.shape)
print(df_fake.info())
print(df_fake.describe())
print(df_fake.duplicated().sum())
df_fake["noticia_false"]=1
df_true["noticia_false"]=0
print(df_fake["noticia_false"].value_counts())

# unimos los dos dataframes
df = pd.concat([df_fake, df_true], ignore_index=True)
df.head()

# distribución de las clases (Noticias falsas vs reales)
sns.countplot(x=df["noticia_false"], palette="coolwarm")
plt.title("Distribución de Noticias Falsas y Reales")
plt.show()


# análisis de la longitud de las noticias --> por lo que se ve en el gráfico, las noticias 
df_fake["text_length"] = df["text"].apply(lambda x: len(str(x).split()))
sns.histplot(data=df_fake, x="text_length",  bins=50, kde=True, palette="coolwarm")
plt.title("Distribución de la longitud de las noticias falsas")
plt.show()


# análisis de la longitud de las noticias --> por lo que se ve en el gráfico, las noticias 
df_true["text_length"] = df_true["text"].apply(lambda x: len(str(x).split()))
sns.histplot(data=df_true, x="text_length",  bins=50, kde=True, palette="coolwarm")
plt.title("Distribución de la longitud de las noticias reales")
plt.show()

# todo el texto en minúsculas
df_true = df_true[df_true["text"].str.lower()]
df_fake = df_fake[df_fake["text"].str.lower()]


# primer nálisis de palabras más frecuentes
fake_text = " ".join(df_fake[df_fake["noticia_false"] == 1]["text"])
real_text = " ".join(df_true[df_true["noticia_false"] == 0]["text"])

print(len(real_text))  
print(real_text[:500])  
plt.subplot(1, 2, 1)
wordcloud_fake = WordCloud(width=500, height=300, background_color="black").generate(fake_text)
plt.imshow(wordcloud_fake, interpolation="bilinear")
plt.title("Palabras más comunes en noticias falsas")
plt.axis("off")

plt.subplot(1, 2, 2)
wordcloud_real = WordCloud(width=500, height=300, background_color="black").generate(real_text)  
plt.imshow(wordcloud_real, interpolation="bilinear")  
plt.title("Palabras más comunes en noticias reales")
plt.axis("off")

plt.show()



# análisis de Stopwords, que básicamente son palabras que no aportan información
nltk.download("stopwords")
stop_words = set(stopwords.words("spanish"))

def filtrar_stopwords(text):
    words = word_tokenize(str(text).lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(filtrar_stopwords)

df.to_csv("fake_news_cleaned.csv", index=False)


fake_text2 = " ".join(df[df["noticia_false"] == 1]["clean_text"])

plt.subplot(1, 2, 2)
wordcloud_real = WordCloud(width=500, height=300, background_color="black").generate(df["clean_text"])  
plt.imshow(wordcloud_real, interpolation="bilinear")  
plt.title("Palabras más comunes en noticias reales")
plt.axis("off")

plt.show()