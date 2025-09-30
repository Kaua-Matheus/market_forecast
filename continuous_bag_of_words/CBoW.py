# ---- Importações ----
import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Classifier:

    def document_vector(self, word_list, model):
        word_vectors = [model.wv[word] for word in word_list if word in model.wv]
        if not word_vectors:
            return np.zeros(model.vector_size)
        return np.mean(word_vectors, axis=0)

    def build(self, dataframe):
        corpus = [value for value in dataframe['Sentence']]
        tokenized_corpus = [sentence.split() for sentence in corpus]
        model = Word2Vec(
            tokenized_corpus, 
            sg=0, 
            vector_size=100, 
            window=5,
            min_count=1,
            workers=4)


        # ---- Treinamento ----
        # X label and y label
        X = np.array([self.document_vector(doc, model) for doc in tokenized_corpus])
        y = np.array(dataframe['Sentiment'].to_list())

        # Separação entre teste e treino
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    def predict(self):
        # Treinando classificador
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(self.X_train, self.y_train)

        y_pred = classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        print(f"Accuracy: {accuracy*100:.2f}%")

        return classifier