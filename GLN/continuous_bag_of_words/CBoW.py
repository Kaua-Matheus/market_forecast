# ---- Importações ----
import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Classifier:

    def __init__(self):
        self.model = None
        self.classifier = None


    # Privates
    def __document_vector(self, word_list: list):

        if self.model != None:
            word_vectors = [self.model.wv[word] for word in word_list if word in self.model.wv]
            if not word_vectors:
                return np.zeros(self.model.vector_size)
            return np.mean(word_vectors, axis=0)
        

    def __train(self) -> LogisticRegression:

        # Treinando classificador
        self.classifier = LogisticRegression(max_iter=1000)
        self.classifier.fit(self.X_train, self.y_train)

        y_pred = self.classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        print("Model trained successfully!")
        print(f"Accuracy: {accuracy*100:.2f}%")

        return self.classifier
    

    # Publics
    def build(self, dataframe: pd.DataFrame):

        corpus = [value for value in dataframe['Sentence']]
        tokenized_corpus = [sentence.split() for sentence in corpus]
        self.model = Word2Vec(
            tokenized_corpus, 
            sg=0, 
            vector_size=100, 
            window=5,
            min_count=1,
            workers=4)

        X = np.array([self.__document_vector(doc) for doc in tokenized_corpus])
        y = np.array(dataframe['Sentiment'].to_list())

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return self.__train()
    

    def predict(self, texts: list) -> list:

        if self.classifier is None:
            raise ValueError("Modelo ainda não trainado. Use a função train() para treinar o modelo.")
        

        # Texto tokenizado
        tokenized_text = [sentence.split() for sentence in texts]
        text_vector = [self.__document_vector(doc) for doc in tokenized_text]
        probabilities = self.classifier.predict(text_vector)

        return probabilities

        # return {
        #     "negative_prob": probabilities[0],
        #     "positive_prob": probabilities[1],
        #     "sentiment_score": probabilities[1] - probabilities[0]
        # }