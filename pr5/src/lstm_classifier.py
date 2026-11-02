from typing import List

import numpy as np
from keras.layers import LSTM, Dense, Dropout, Flatten, Input
from keras.models import Sequential

from networks import EMBEDDING_DIMS, MAXLEN, NeuralClassifier
from sequences import pad_trunc, tokenize_and_vectorize


class LSTMClassifier(NeuralClassifier):
    """
    Классификатор тональности на основе сети с долгой краткосрочной памятью (LSTM).

    Обучение, сохранение и предсказание унаследованы от общего класса сетей из ПР4,
    поэтому LSTM получает на вход те же последовательности векторов Word2Vec, что CNN и SimpleRNN.
    """

    def __init__(self, classes, num_neurons: int = 50, dropout: float = 0.2,
                 return_sequences: bool = True, layers: int = 1):
        """
        Задает гиперпараметры LSTM-сети.

        :param classes: список названий классов
        :param num_neurons: число нейронов (размер вектора памяти) в слое LSTM
        :param dropout: доля выключаемых нейронов
        :param return_sequences: передавать дальше выходы всех шагов (True) или только последнего
        :param layers: число слоев LSTM, поставленных друг за другом
        """
        super().__init__(classes)
        self.num_neurons = num_neurons
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.layers = layers

    def build(self) -> Sequential:
        """
        LSTM (один или несколько слоев) -> Dropout -> Flatten -> Dense -> softmax.

        Промежуточные слои всегда возвращают выходы всех шагов, так как следующему
        слою LSTM на вход нужна последовательность.

        :return: скомпилированная модель Keras
        """
        layers = [Input(shape=(MAXLEN, EMBEDDING_DIMS))]
        for i in range(self.layers):
            last = i == self.layers - 1
            layers.append(LSTM(self.num_neurons, return_sequences=self.return_sequences or not last))
        layers.append(Dropout(self.dropout))
        if self.return_sequences:
            layers.append(Flatten())
        layers.append(Dense(len(self.classes), activation="softmax"))
        model = Sequential(layers)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        return model

    def key_words(self, tokens: List[str], word_vectors, top: int = 3) -> List[str]:
        """
        Находит слова, сильнее всего повлиявшие на ответ сети. Из документа по очереди
        убирается каждое слово, и смотрится, насколько падает вероятность выбранного класса.

        :param tokens: леммы документа
        :param word_vectors: модель WordVectors из ПР3
        :param top: сколько слов вернуть
        :return: список лемм в порядке убывания влияния
        """
        known = [t for t in tokens if t in word_vectors.model.wv]
        unique = list(dict.fromkeys(known[:MAXLEN]))
        if len(unique) < 2:
            return []
        variants = [known] + [[t for t in known if t != word] for word in unique]
        probs = self.predict(pad_trunc(tokenize_and_vectorize(variants, word_vectors), MAXLEN, EMBEDDING_DIMS))
        label = int(probs[0].argmax())
        drops = probs[0, label] - probs[1:, label]
        return [unique[i] for i in np.argsort(-drops)[:top] if drops[i] > 0.01]
