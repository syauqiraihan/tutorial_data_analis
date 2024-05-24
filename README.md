# Analisis Tingkat Kebahagian Negara-Negara Di Dunia dengan beberapa aspek pertimbangan

untuk mendownload data bisa akses [link berikut](https://drive.google.com/drive/folders/1IRG069z2OwI5KDBwN0M1G3AUNNsQ2iU4)!




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
'''

