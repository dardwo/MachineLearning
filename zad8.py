import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv('flats_for_clustering.tsv', header=0, sep='\t')
data["Piętro"] = data["Piętro"].apply(lambda x: 0 if x in ["parter", "niski parter"] else x)
data["Piętro"] = data["Piętro"].apply(lambda x: 5 if x in ["poddasze"] else x)
data = data[data["cena"] <= 1000000]
data = data.dropna()

new_data = StandardScaler().fit_transform(data)

#algorytm k srednich 
kmeans = KMeans(n_clusters=5, n_init='auto').fit(new_data)

#redukcja liczby wymiarow
pca = PCA(n_components=2)
pca.fit(new_data)
new_data = pca.transform(new_data)

data[['X', 'y']] = new_data

plt.scatter(data['X'], data['y'], c=kmeans.labels_, cmap='turbo')
plt.title('Wykres punktowy')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()