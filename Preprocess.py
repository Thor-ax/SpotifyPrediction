import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA

class Preprocess:


    def __init__(self, ds):
        self.ds = ds
        self.le = LabelEncoder()

    def infos(self):
        print('-----')
        print('dataset infos')
        print('-----')
        print(self.ds.describe())
        print('-----')
        print(self.ds.info())

    def nbNanValues(self):
        print('-----')
        print('Nan values')
        print('-----')
        print(self.ds.isna().sum())

    def preprocessDsForPopularityPrediction(self):
        self.ds['release_date'] = pd.to_datetime(self.ds['release_date'], format='%Y-%m-%d').dt.year
        le = LabelEncoder()
        self.ds['artist_name'] = le.fit_transform(self.ds['artist_name'])
        self.ds['track_name'] = le.fit_transform(self.ds['track_name'])
        self.ds['genres'] = le.fit_transform(self.ds['genres'])

        

        self.ds.drop(['id'], axis=1, inplace=True)

        features_columns = [col for col in self.ds.columns if col not in ['popularity']]

        #scale datas
        sts = StandardScaler()

        sts = sts.fit(self.ds[features_columns])

        data_scaler = sts.transform(self.ds[features_columns])

        data_scaler = pd.DataFrame(data_scaler)
        data_scaler.columns = features_columns

        # get X_data and y_data
        data_scaler['popularity'] = self.ds['popularity']
        y_data = data_scaler['popularity']

        X_data = data_scaler
        X_data.drop(['popularity'], axis=1, inplace=True)
        X_data = X_data.to_numpy()

        self.X_data = X_data
        self.y_data = y_data
        return (X_data, y_data)
   

    def pca(self, X_data):
        pca = PCA(n_components=7)
        print(X_data)
        print(X_data.shape)
        pca.fit(X_data)
        X_data = pca.transform(X_data)
        print(X_data.shape)
        print(X_data)
        self.X_data = X_data
        return X_data

    def PreprocessDs(self, isTrainingDs):
        self.ds['release_date'] = pd.to_datetime(self.ds['release_date'], format='%Y-%m-%d').dt.year
        self.ds['explicit'] = self.le.fit_transform(self.ds['explicit'])
        if(isTrainingDs):
            self.ds['genre'] = self.le.fit_transform(self.ds['genre'])

        features_columns = [col for col in self.ds.columns if col not in ['time_signature', 'genre']]

        sts = StandardScaler()

        sts = sts.fit(self.ds[features_columns])

        data_scaler = sts.transform(self.ds[features_columns])

        data_scaler = pd.DataFrame(data_scaler)
        data_scaler.columns = features_columns

        if(isTrainingDs):
            data_scaler['genre'] = self.ds['genre']
            y_data = data_scaler['genre']

            X_data = data_scaler
            X_data.drop(['genre'], axis=1, inplace=True)
            X_data = X_data.to_numpy()

            self.X_data = X_data
            self.y_data = y_data
            return (X_data, y_data)

        X_data = data_scaler.to_numpy()
        self.X_data = X_data
        return X_data
    def inverse(self,pred):
        return self.le.inverse_transform(pred)
    def getDs(self):
        return self.ds

    def frequency(self, test_data):
        dist_cols = 6
        dist_rows = len(test_data.columns)
        plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))
        print(test_data)
        for i, col in enumerate(test_data.columns):
            ax = plt.subplot(dist_rows, dist_cols, i + 1)
            ax = sns.kdeplot(self.ds[col], color="Red", shade=True)
            ax = sns.kdeplot(test_data[col], color="Blue", shade=True)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax = ax.legend(["train", "test"])
        plt.show()

    def corr(self):
        corr = self.ds.corr()
        f, ax = plt.subplots(figsize=(10, 6))
        hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm")
        plt.show()