import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.layers import LSTM, Dense, Embedding, Input
from keras.models import Model
from sklearn.preprocessing import normalize

SPECIAL_TOKENS = ["<pad>", "<начало>", "<конец>"]
PAD, START, END = range(3)


class Seq2SeqAutoencoder:
    """
    Модель sequence-to-sequence «кодировщик – декодировщик» на двух слоях LSTM.

    Модель обучается как автокодировщик: кодировщик читает леммы документа и сжимает их
    в вектор идеи (выход скрытого слоя и состояние памяти LSTM), а декодировщик по этому
    вектору восстанавливает исходную последовательность слово за словом. Так как у сети
    есть узкое место, вектор идеи вынужден хранить главное содержание документа и
    используется как семантическое представление документа (аналог вектора Doc2Vec).
    """

    def __init__(self, word_vectors, vocab_size: int = 3000, maxlen: int = 30,
                 num_neurons: int = 128, reverse_input: bool = True):
        """
        Задает гиперпараметры модели.

        :param word_vectors: обученная модель WordVectors из ПР3 (веса слоя Embedding)
        :param vocab_size: размер словаря с учетом служебных токенов
        :param maxlen: сколько первых лемм документа кодируется и восстанавливается
        :param num_neurons: число нейронов в LSTM кодировщика и декодировщика
        :param reverse_input: подавать документ кодировщику в обратном порядке
        """
        self.word_vectors = word_vectors
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.num_neurons = num_neurons
        self.reverse_input = reverse_input
        self.mean = None
        self.vocabulary = []
        self.index = {}
        self.model = None
        self.encoder_model = None
        self.decoder_model = None

    def build_vocabulary(self, documents: List[List[str]]):
        """
        Составляет словарь из самых частых лемм корпуса, у которых есть вектор Word2Vec.

        :param documents: документы корпуса в виде списков лемм
        """
        wv = self.word_vectors.model.wv
        counts = Counter(token for doc in documents for token in doc if token in wv)
        words = [w for w, _ in counts.most_common(self.vocab_size - len(SPECIAL_TOKENS))]
        self.vocabulary = SPECIAL_TOKENS + words
        self.index = {w: i for i, w in enumerate(self.vocabulary)}

    def to_ids(self, tokens: List[str]) -> List[int]:
        """
        Переводит леммы в номера словаря. Слова не из словаря отбрасываются, как и слова
        без вектора Word2Vec в ПР4: с токеном <unk> декодировщик учился повторять только его.

        :param tokens: список лемм
        :return: список номеров не длиннее maxlen
        """
        return [self.index[t] for t in tokens if t in self.index][:self.maxlen]

    def encoder_data(self, documents: List[List[str]]) -> np.ndarray:
        """
        Готовит вход кодировщика: номера лемм, дополненные нулями (<pad>) до maxlen.

        :param documents: документы в виде списков лемм
        :return: массив (число документов, maxlen)
        """
        data = np.zeros((len(documents), self.maxlen), dtype="int32")
        for i, doc in enumerate(documents):
            ids = self.to_ids(doc)
            if self.reverse_input:
                ids = ids[::-1]
            data[i, :len(ids)] = ids
        return data

    def decoder_data(self, documents: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Готовит данные декодировщика для обучения с «усилением учителем»: вход начинается
        с токена <начало>, а целевая последовательность сдвинута на шаг и заканчивается <конец>.

        :param documents: документы в виде списков лемм
        :return: (вход декодировщика, целевые номера), оба размера (число документов, maxlen + 1)
        """
        decoder_input = np.zeros((len(documents), self.maxlen + 1), dtype="int32")
        decoder_target = np.zeros((len(documents), self.maxlen + 1), dtype="int32")
        for i, doc in enumerate(documents):
            ids = self.to_ids(doc)
            decoder_input[i, :len(ids) + 1] = [START] + ids
            decoder_target[i, :len(ids) + 1] = ids + [END]
        return decoder_input, decoder_target

    def embedding_matrix(self) -> np.ndarray:
        """
        Матрица весов слоя Embedding: нормированные векторы Word2Vec из ПР3 для слов словаря,
        нулевой вектор для <pad> и случайные векторы для токенов <начало> и <конец>.

        :return: массив (vocab_size, размерность векторов)
        """
        wv = self.word_vectors.model.wv
        rng = np.random.default_rng(1337)
        matrix = np.zeros((len(self.vocabulary), wv.vector_size), dtype="float32")
        for i, word in enumerate(self.vocabulary):
            if i == PAD:
                continue
            vector = wv[word] if word in wv else rng.normal(size=wv.vector_size)
            matrix[i] = vector / np.linalg.norm(vector)
        return matrix

    def build(self):
        """
        Строит модель для обучения, а также отдельные модели кодировщика и декодировщика
        для получения векторов идей и пошагового восстановления текста (как в лекции).
        """
        embedding = Embedding(len(self.vocabulary), self.word_vectors.model.wv.vector_size,
                              embeddings_initializer=Constant(self.embedding_matrix()),
                              trainable=False, mask_zero=True)

        encoder_inputs = Input(shape=(None,), dtype="int32")
        encoder = LSTM(self.num_neurons, return_state=True)
        _, state_h, state_c = encoder(embedding(encoder_inputs))
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None,), dtype="int32")
        decoder_lstm = LSTM(self.num_neurons, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(embedding(decoder_inputs), initial_state=encoder_states)
        decoder_dense = Dense(len(self.vocabulary), activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.encoder_model = Model(encoder_inputs, encoder_states)
        thought_input = [Input(shape=(self.num_neurons,)), Input(shape=(self.num_neurons,))]
        outputs, state_h, state_c = decoder_lstm(embedding(decoder_inputs), initial_state=thought_input)
        self.decoder_model = Model([decoder_inputs] + thought_input, [decoder_dense(outputs), state_h, state_c])

    def train(self, documents: List[List[str]], epochs: int = 40, batch_size: int = 64, verbose: int = 2):
        """
        Составляет словарь, строит модель и обучает ее восстанавливать документы корпуса.
        10 % документов отводится на проверку, обучение останавливается, если ошибка
        на проверке не уменьшается 3 эпохи подряд. После обучения запоминается средний
        выход кодировщика по корпусу (нужен для поиска похожих документов).

        :param documents: документы корпуса в виде списков лемм
        :param epochs: максимальное число эпох
        :param batch_size: размер пакета
        :param verbose: подробность вывода Keras
        :return: история обучения
        """
        keras.utils.set_random_seed(1337)
        documents = [doc for doc in documents if doc]
        self.build_vocabulary(documents)
        self.build()
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        encoder_input = self.encoder_data(documents)
        decoder_input, decoder_target = self.decoder_data(documents)
        stopper = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        history = self.model.fit([encoder_input, decoder_input], decoder_target, batch_size=batch_size,
                                 epochs=epochs, validation_split=0.1, callbacks=[stopper], verbose=verbose)
        self.mean = self.encoder_model.predict(encoder_input, batch_size=256, verbose=0)[0].mean(axis=0)
        return history

    def thought_vectors(self, documents: List[List[str]]) -> np.ndarray:
        """
        Вычисляет векторы идей документов: объединение выхода скрытого слоя h
        и состояния памяти c кодировщика, нормированное по длине.

        :param documents: документы в виде списков лемм
        :return: массив (число документов, 2 * num_neurons)
        """
        state_h, state_c = self.encoder_model.predict(self.encoder_data(documents), batch_size=256, verbose=0)
        return normalize(np.hstack([state_h, state_c]))

    def search_vectors(self, documents: List[List[str]]) -> np.ndarray:
        """
        Вычисляет векторы для поиска похожих документов. Берется только выход скрытого слоя h:
        состояние памяти c не ограничено функцией th, растет вместе с длиной документа, и по
        полному вектору идеи все короткие тексты оказываются похожи друг на друга. Из h
        вычитается средний выход кодировщика по корпусу, затем вектор нормируется по длине.

        :param documents: документы в виде списков лемм
        :return: массив (число документов, num_neurons)
        """
        state_h, _ = self.encoder_model.predict(self.encoder_data(documents), batch_size=256, verbose=0)
        return normalize(state_h - self.mean)

    def decode(self, tokens: List[str]) -> List[str]:
        """
        Восстанавливает документ по его вектору идеи: декодировщик получает токен <начало>,
        затем на каждом шаге свое предыдущее слово, пока не выдаст <конец> или maxlen слов.

        :param tokens: леммы документа
        :return: восстановленный список лемм
        """
        thought = self.encoder_model.predict(self.encoder_data([tokens]), verbose=0)
        target = np.array([[START]])
        result = []
        while len(result) < self.maxlen:
            output, state_h, state_c = self.decoder_model.predict([target] + list(thought), verbose=0)
            token = int(output[0, -1].argmax())
            if token == END:
                break
            result.append(self.vocabulary[token])
            target = np.array([[token]])
            thought = [state_h, state_c]
        return result

    def save(self, path: Path):
        """
        Сохраняет веса модели, словарь и средний выход кодировщика.

        :param path: путь без расширения, например models/seq2seq
        """
        self.model.save_weights(str(path.with_suffix(".weights.h5")))
        path.with_suffix(".json").write_text(json.dumps({
            "vocabulary": self.vocabulary, "maxlen": self.maxlen, "num_neurons": self.num_neurons,
            "reverse_input": self.reverse_input, "mean": self.mean.tolist(),
        }, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path):
        """
        Загружает словарь и настройки, заново строит модель и загружает в нее веса.

        :param path: путь без расширения, например models/seq2seq
        """
        config = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        self.vocabulary = config["vocabulary"]
        self.index = {w: i for i, w in enumerate(self.vocabulary)}
        self.vocab_size = len(self.vocabulary)
        self.maxlen = config["maxlen"]
        self.num_neurons = config["num_neurons"]
        self.reverse_input = config["reverse_input"]
        self.mean = np.array(config["mean"], dtype="float32")
        self.build()
        self.model.load_weights(str(path.with_suffix(".weights.h5")))

    @staticmethod
    def exists(path: Path) -> bool:
        """
        Проверяет, сохранена ли модель на диске.

        :param path: путь без расширения
        :return: True, если есть веса и словарь
        """
        return path.with_suffix(".json").exists() and path.with_suffix(".weights.h5").exists()
