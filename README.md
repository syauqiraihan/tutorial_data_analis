# Analisis Tingkat Kebahagian Negara-Negara Di Dunia dengan beberapa aspek pertimbangan

untuk mendownload data bisa akses [link berikut](https://drive.google.com/drive/folders/1KTniZixWMkbD5fRbVmG-YnzyCtA1flaJ)!



## Analisis Tingkat Kebahagiaan Negara di Dunia Berdasarkan Beberapa Faktor
<img src="https://github.com/syauqiraihan/tutorial_data_analis/blob/main/Distribusi%20Happines%20Box).png">

### insight
- GDP per capita adalah faktor penjelas terbesar yang mempengaruhi tingkat kebahagiaan seseorang secara general di seluruh dunia
- Sedangkan generosity dan perceptions of corruption berada di posisi terendah yang mempengaruhi tingkat kebahagiaan seseorang di seluruh dunia

## Analisis Tingkat Kebahagiaan Negara di Dunia Berdasarkan Beberapa Faktor Dilihat dari Garis Regresi
<img src="https://github.com/syauqiraihan/tutorial_data_analis/blob/main/Distribusi%20Kebahagiaan%20(red).png">

### insight
- Yang paling stabil dalam mempengaruhi tingkat kebahagiaan adalah "Generosity"
- Yang paling pesat mempengaruhi tren positif tingkat kebahagiaan di dunia secara general adalah "Explained by Sosial Support"
- GDP per capita adalah faktor penjelas terbesar yang mempengaruhi tingkat kebahagiaan seseorang secara general di seluruh dunia
- Sedangkan generosity dan perceptions of corruption berada di posisi terendah yang mempengaruhi tingkat kebahagiaan seseorang di seluruh dunia

## Analisis Tingkat Kebahagiaan Negara di Indonesia Dibandingkan Uzbekistan
<img src="https://github.com/syauqiraihan/tutorial_data_analis/blob/main/Rank%20Indo%20Vs%20Uzbek.png">

### insight
- Global Median ada di angka 5,57
- Indonesia berada di posisi 87 di dunia untuk skor tingkat kebahagaiaannya
- Sedangkan uzbekistan berada jauh diatas Indonesia di posisi 53

## Analisis Faktor Tingkat Kebahagiaan Negara di Indonesia Dibandingkan Uzbekistan
<img src="https://github.com/syauqiraihan/tutorial_data_analis/blob/main/Faktor%20Indo%20Uzbek.png">

### insight
- Faktor terbesar mengapa uzbekistan jauh diatas Indonesia skor tingkat kebahagiaannya adalah dipengaruhi dari faktor "Dystopia"
- Ada beberapa faktor yang uzbekistan juga unggul dibandingkan Indoneisa yaitu "social support" "health life expectancy" , " Perceptions of corruption" dan "Freedom to make life choices" atau kebebasan bernegara
- Indonesia ada 2 faktor yang sedikit unggul dari Uzbekistan yaitu "GDP per capita" dan "generosity"

## Analisis Tingkat Kebahagiaan Negara di Beberapa Region Dunia 
<img src="https://github.com/syauqiraihan/tutorial_data_analis/blob/main/full%20country%20afc%20semifinal.png">

### insight

##### Asia Bagian Timur
- Japan memuncaki tingkat kebahagian di asia kawasan timur dengan skor kebahagiaan 6,04
- Disusul korea selatan, mongolia, dan china

#### Asia tenggara
- Singapura menempati posisi tertinggi tingkat kebahagiaan di asia tenggara dengan skor kebahagiaan 6,48
- Myanmar berada di posisi terbawah dengan skor kebahagiaan 4,39

#### Asia Barat
- Israel berada di posisi pertama dengan skor tingkat kebahagiaan sebesar 7,36
- Dan lebanon di posisi terbawah dengan skor tingkat kebahagiaan 2,96

#### Asia Tengah
- Di Asia tengah posisi pertama diduduki oleh Kazakhstan dengan skor tingkat kebahagiaan 6,23 dan Tajikistan berada di posisi terakhir dengan skor tingkat kebahagiaan 5,38

#### Perbandingan Antara Indonesia, Jepang dan Iraq sebagai kandidat 3 besar AFC u 23
- Uzbekistan ada di posisi pertama dengan skor tingkat kebahagiaan adalah 6,06
- Disusul oleh Indonesia dengan skor tingkat kebahagiaan 5,24
- Di posisi ketiga ada Iraq dengan skor tingkat kebahagiaan 4,94

'''python
# 1. Business Understanding



# 2. Data Understanding

## ⚔️IMPORTING LIBRARIES

!pip install xgboost

import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn. linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics

## 🛠️LOADING DATA

df = pd.read_csv(r"C:\Users\Aditya P J\Documents\Python Scripts\Data\diamonds.csv")
df

df.head()

df.info()

df.sample(10)

### DESKRIPSI DATA

Dataset berikut berisi informasi harga dan atribut lainnya.

~ carat (0.2-5.01): Carat adalah berat fisik berlian yang diukur dalam carat metrik. Satu carat sama dengan 0.20 gram dan dibagi menjadi 100 poin.

~ cut (Fair, Good, Very Good, Premium, Ideal): Kualitas potongan. Semakin presisi potongan berlian, semakin memikat berlian tersebut di mata sehingga dinilai dengan nilai tinggi.

~ color (dari J (worst) hingga D (best)): Warna berlian berkualitas permata muncul dalam berbagai nuansa. Dalam rentang dari tidak berwarna hingga kuning muda atau coklat muda. Berlian yang tidak berwarna adalah yang paling langka. Warna alami lainnya (seperti biru, merah, pink) dikenal sebagai "fancy," dan penilaian warnanya berbeda dari berlian putih yang tidak berwarna.

~ clarity (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)): Berlian dapat memiliki karakteristik internal yang dikenal sebagai inklusi atau karakteristik eksternal yang dikenal sebagai cacat. Berlian tanpa inklusi atau cacat sangat langka; namun, sebagian besar karakteristik hanya dapat dilihat dengan pembesaran.

~ depth (43-79): Ini adalah persentase kedalaman total yang setara dengan z / mean(x, y) = 2 * z / (x + y). Kedalaman berlian adalah tingginya (dalam milimeter) yang diukur dari culet (ujung bawah) hingga meja (permukaan atas datar) seperti yang disebutkan dalam diagram berlabel di atas.

~ table (43-95): Ini adalah lebar bagian atas berlian relatif terhadap titik terlebar. Ini memberikan berlian kilauan dan kecemerlangan yang menakjubkan dengan memantulkan cahaya ke segala arah yang ketika dilihat oleh pengamat, tampak berkilau.

~ price ($326 - $18826): Ini adalah harga berlian dalam dolar AS. Ini adalah kolom target kita dalam dataset ini.

~ x (0 - 10.74): Panjang berlian (dalam mm).

~ y (0 - 58.9): Lebar berlian (dalam mm).

~ z (0 - 31.8): Kedalaman berlian (dalam mm).

## ⛓️DATA ANALYSIS

### Memeriksa nilai dan variabel kategori yang hilang

#Checking missing value in dataset
df.info()

Catatan :
Terdapat total data adalah 53940, berdasarkan informasi mengenai jumlah isian perkolom, terlihat bahwa jumlah baris adalah 53840. Jadi data tersebut tidak memiliki missing value. 

Tipe data (cut, color, dan clarity) adalah object, sehingga perlu di convert menjadi variabel numerik Sebelum kita memasukkan data ke dalam algoritma. 

### Mengevaluasi fitur kategorikal

plt.figure(figsize=(10,9))
cols = sns.color_palette("coolwarm", 5)
ax = sns.violinplot(x="cut", y="price", data=df, palette=cols, scale="count")
ax.set_title("Diamond Cut for Price", color="#774571", fontsize=20)
ax.set_ylabel("Price", color="#4e4c39", fontsize=15)
ax.set_xlabel("Cut", color="#4e4c39", fontsize=15)
plt.show()


plt.figure(figsize=(12,9))
# Menggunakan palet warna 'coolwarm' dari seaborn
cols = sns.color_palette("coolwarm", 7)
ax = sns.violinplot(x="color", y="price", data=df, palette=cols, scale="count")
ax.set_title("Diamond Colors for Price", color="#774571", fontsize=20)
ax.set_ylabel("Price", color="#4e4c39", fontsize=15)
ax.set_xlabel("Color", color="#4e4c39", fontsize=15)
plt.show()


plt.figure(figsize=(13,8))
# Menggunakan palet warna 'viridis' dari seaborn
cols = sns.color_palette("viridis", 8)
ax = sns.violinplot(x="clarity", y="price", data=df, palette=cols, scale="count")
ax.set_title("Diamond Clarity for Price", color="#774571", fontsize=20)
ax.set_ylabel("Price", color="#4e4c39", fontsize=15)
ax.set_xlabel("Clarity", color="#4e4c39", fontsize=15)
plt.show()


Catatan :
Potongan "Ideal" diamonds adalah yang paling banyak jumlahnya, sedangkan "Fair" diamonds adalah yang paling sedikit jumlahnya. Lebih banyak diamonds dari semua jenis potongan untuk kategori harga yang lebih rendah.

Dengan warna "J" diamonds, yang merupakan yang terburuk, sangat langka, namun "H" dan "G" diamonds lebih banyak jumlahnya meskipun kualitasnya juga rendah.

Dengan kejelasan "IF" diamonds, yang merupakan yang terbaik, serta "I1" diamonds, yang merupakan yang terburuk, sangat langka, sementara yang lainnya sebagian besar memiliki kejelasan di antara keduanya.

### Statistik Deskriptif

# Melakukan Analisis Univariat untuk deskripsi statistik dan pemahaman tentang sebaran data
df.describe().T


Catatan :
    
    "Price" seperti yang diharapkan cenderung condong ke kanan, dengan jumlah titik data yang lebih banyak di sebelah kiri.
Di bawah fitur dimensional 'x', 'y', & 'z' - nilai minimum adalah 0 sehingga membuat titik data tersebut menjadi objek berlian 1D atau 2D yang tidak masuk akal - oleh karena itu perlu untuk diimputasi dengan nilai yang sesuai atau dihapus sama sekali.

# Melakukan Analisis Bivariat dengan memeriksa pairplot
ax = sns.pairplot(df, hue="cut", palette=cols)


Catatan:
1. Terdapat fitur "unnamed" yang tidak berguna, yang merupakan indeks dan perlu dihilangkan.
2. Terdapat outlier yang perlu ditangani karena dapat mempengaruhi kinerja model
3. Kolom "y" dan "z" memiliki beberapa outlier dimensional dalam dataset dan perlu dihilangkan.
4. Fitur "depth" & "table" seharusnya dibatasi setelah diperiksa Plot Garis.

### Memeriksa Potensi Outlier

lm = sns.lmplot(x = 'price', y= 'y', data = df, scatter_kws = {'color': "#FED8B1"}, line_kws = {'color' : '#4e4c39'})
plt.title('Plot Garis Price vs y', color = '#774571', fontsize = 15)
plt.show()

lm = sns.lmplot(x = 'price', y= 'z', data = df, scatter_kws = {'color': "#FED8B1"}, line_kws = {'color' : '#4e4c39'})
plt.title('Plot Garis Price vs z', color = '#774571', fontsize = 15)
plt.show()

lm = sns.lmplot(x = 'price', y= 'depth', data = df, scatter_kws = {'color': "#FED8B1"}, line_kws = {'color' : '#4e4c39'})
plt.title('Plot Garis Price vs depth', color = '#774571', fontsize = 15)
plt.show()

lm = sns.lmplot(x = 'price', y= 'table', data = df, scatter_kws = {'color': "#FED8B1"}, line_kws = {'color' : '#4e4c39'})
plt.title('Plot Garis Price vs table', color = '#774571', fontsize = 15)
plt.show()

Catatan:
Dengan melakukan plot di atas, outlier dapat dilihat dengan mudah.

## 🪛DATA PREPROCESSING

### Pembersihan Data

#Menghapus fitur "Unnamed"
df = df.drop(["Unnamed: 0"], axis=1)
df.shape

#Menghapus titik data yang memiliki nilai minimum 0 pada salah satu fitur x, y, atau z. 
df = df.drop(df[df["x"]==0].index)
df = df.drop(df[df["y"]==0].index)
df = df.drop(df[df["z"]==0].index)
df.shape

### Menghapus Ouliers

#Menghapus outlier (karena memiliki dataset yang besar) dengan menentukan langkah-langkah yang sesuai di seluruh fiturdf = data_df[(data_df["depth"]<75)&(data_df["depth"]>45)]
df = df[(df["depth"]<75)&(df["depth"]>45)]
df = df[(df["table"]<80)&(df["table"]>40)]
df = df[(df["x"]<40)]
df = df[(df["y"]<40)]
df = df[(df["z"]<40)&(df["z"]>2)]
df.shape

### Encoding Variabel Kategorik

#Membuat salinan untuk menjaga data asli dalam bentuknya yang utuh
df1 = df.copy()

#Menerapkan label encoder pada kolom-kolom dengan data kategorikal
columns = ['cut','color','clarity']
label_encoder = LabelEncoder()
for col in columns:
    df1[col] = label_encoder.fit_transform(df1[col])
df1.describe()

Catatan:

Setelah fitur-fitur kategorikal dikonversi menjadi kolom-kolom numerik, kita juga mendapatkan ringkasan 5 poin bersama dengan jumlah, rata-rata, dan standar deviasi untuk mereka. 

Sekarang, kita dapat menganalisis matriks korelasi setelah selesai dengan pre-processing untuk pemilihan fitur yang mungkin guna membuat dataset lebih bersih dan optimal sebelum kita masukkan ke dalam algoritma.

### Matriks Kolerasi

#Mengeksaminasi matriks korelasi menggunakan heatmap
cmap = sns.diverging_palette(205, 133, 63, as_cmap=True)
cols = (["#FFCDEA", "#E59BE9", "#FB9AD1", "#BC7FCD", "#D862BC", "#86469C"])
corrmat= df1.corr()
f, ax = plt.subplots(figsize=(15,12))
sns.heatmap(corrmat,cmap=cols,annot=True)

Catatan:

Fitur "carat", "x", "y", "z" memiliki korelasi yang tinggi dengan variabel target kita, yaitu harga.

Fitur "cut", "clarity", "depth" memiliki korelasi yang sangat rendah (<|0.1|) sehingga mungkin dapat dihapus, meskipun karena hanya ada beberapa fitur yang dipilih, kita tidak akan melakukannya.

## 🪅MODEL BUILDING

# Mendefinisikan variabel independen dan dependen
x = df1.drop(["price"],axis =1)
y = df1["price"]
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.20, random_state=25)

# Membangun Pipeline Standar Scaler dan Model untuk Berbagai Regressor

pipeline_lr=Pipeline([("scalar1",StandardScaler()),
                     ("lr",LinearRegression())])

pipeline_lasso=Pipeline([("scalar2", StandardScaler()),
                      ("lasso",Lasso())])

pipeline_dt=Pipeline([("scalar3",StandardScaler()),
                     ("dt",DecisionTreeRegressor())])

pipeline_rf=Pipeline([("scalar4",StandardScaler()),
                     ("rf",RandomForestRegressor())])


pipeline_kn=Pipeline([("scalar5",StandardScaler()),
                     ("kn",KNeighborsRegressor())])


pipeline_xgb=Pipeline([("scalar6",StandardScaler()),
                     ("xgb",XGBRegressor())])

# Daftar semua saluran pipa
pipelines = [pipeline_lr, pipeline_lasso, pipeline_dt, pipeline_rf, pipeline_kn, pipeline_xgb]

# Dictionary of pipelines and model types for ease of reference
pipeline_dict = {0: "LinearRegression", 1: "Lasso", 2: "DecisionTree", 3: "RandomForest",4: "KNeighbors", 5: "XGBRegressor"}

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(x_train, y_train)

cv_results_rms = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, x_train,y_train,scoring="neg_root_mean_squared_error", cv=12)
    cv_results_rms.append(cv_score)
    print("%s: %f " % (pipeline_dict[i], -1 * cv_score.mean()))

# Prediksi model pada data pengujian dengan XGBClassifier yang memberikan kita RMSE paling sedikit
pred = pipeline_xgb.predict(x_test)
print("R^2:",metrics.r2_score(y_test, pred))
print("Adjusted R^2:",1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1))

## 🪡END
'''


