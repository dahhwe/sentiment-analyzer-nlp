from typing import List

import numpy as np


def tokenize_and_vectorize(documents: List[List[str]], word_vectors, unit_length: bool = True) -> List[List[np.ndarray]]:
    """
    Переводит документы в последовательности векторов слов Word2Vec из ПР3.

    Слова, которых нет в словаре Word2Vec, пропускаются. По умолчанию векторы нормируются
    по длине, чтобы частые слова с длинными векторами не перевешивали остальные.

    :param documents: документы в виде списков лемм
    :param word_vectors: обученная модель WordVectors из ПР3
    :param unit_length: нормировать векторы слов по длине
    :return: для каждого документа список векторов его слов
    """
    wv = word_vectors.model.wv
    vectorized_data = []
    for tokens in documents:
        sample_vecs = []
        for token in tokens:
            if token in wv:
                vector = wv[token]
                sample_vecs.append(vector / np.linalg.norm(vector) if unit_length else vector)
        vectorized_data.append(sample_vecs)
    return vectorized_data


def pad_trunc(data: List[List[np.ndarray]], maxlen: int, embedding_dims: int) -> np.ndarray:
    """
    Приводит все последовательности к длине maxlen: длинные усекаются,
    короткие дополняются в конце нулевыми векторами.

    :param data: последовательности векторов слов
    :param maxlen: длина последовательности
    :param embedding_dims: размерность векторов слов
    :return: массив размера (число документов, maxlen, embedding_dims)
    """
    result = np.zeros((len(data), maxlen, embedding_dims), dtype="float32")
    for i, sample in enumerate(data):
        sample = sample[:maxlen]
        if sample:
            result[i, :len(sample)] = sample
    return result
