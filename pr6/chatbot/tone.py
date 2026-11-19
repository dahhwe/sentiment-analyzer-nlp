from typing import List

import numpy as np

CLASSES = ["negative", "neutral", "positive"]


class ToneAnalyzer:
    """
    Тональность реплик по сетям из прошлых работ: среднее вероятностей CNN (ПР4) и LSTM (ПР5).

    Сети обучены на отзывах длиной в десятки слов, а реплики в диалоге короткие, поэтому
    уверенными считаются только ответы с вероятностью класса не меньше порога.
    """

    def __init__(self, normalize, word_vectors, networks, pad_trunc, tokenize_and_vectorize,
                 maxlen: int, embedding_dims: int, threshold: float = 0.6):
        """
        Создает анализатор.

        :param normalize: функция нормализации ПР1 (текст -> леммы без стоп-слов)
        :param word_vectors: модель Word2Vec из ПР3, на которой обучены сети
        :param networks: обученные классификаторы с методом predict (CNN и LSTM)
        :param pad_trunc: функция pad_trunc из ПР4
        :param tokenize_and_vectorize: функция tokenize_and_vectorize из ПР4
        :param maxlen: длина последовательности
        :param embedding_dims: размерность векторов слов
        :param threshold: минимальная вероятность, при которой тональность считается уверенной
        """
        self.normalize = normalize
        self.word_vectors = word_vectors
        self.networks = networks
        self.pad_trunc = pad_trunc
        self.tokenize_and_vectorize = tokenize_and_vectorize
        self.maxlen = maxlen
        self.embedding_dims = embedding_dims
        self.threshold = threshold

    def probabilities(self, texts: List[str]) -> np.ndarray:
        """
        Вероятности классов тональности.

        :param texts: исходные тексты
        :return: массив (число текстов, 3) в порядке negative, neutral, positive
        """
        sequences = self.tokenize_and_vectorize([self.normalize(t) for t in texts], self.word_vectors)
        x = self.pad_trunc(sequences, self.maxlen, self.embedding_dims)
        return np.mean([network.predict(x) for network in self.networks], axis=0)

    def labels(self, texts: List[str]) -> List[str]:
        """
        Уверенная тональность текстов: класс с наибольшей вероятностью, если она не меньше
        порога, иначе neutral.

        :param texts: исходные тексты
        :return: список меток negative, neutral или positive
        """
        probs = self.probabilities(texts)
        return [CLASSES[int(p.argmax())] if p.max() >= self.threshold else "neutral" for p in probs]
