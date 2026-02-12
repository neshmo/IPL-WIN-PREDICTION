from sklearn.preprocessing import LabelEncoder

class SafeLabelEncoder:
    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, series):
        self.encoder.fit(series)

    def transform(self, series):
        return self.encoder.transform(series)

    def fit_transform(self, series):
        return self.encoder.fit_transform(series)
