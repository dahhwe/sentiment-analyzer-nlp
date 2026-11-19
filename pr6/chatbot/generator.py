import json
from pathlib import Path
from typing import List, Tuple

import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.layers import LSTM, Dense, Embedding, Input
from keras.models import Model

PAD, START, END = 0, 1, 2
SPECIAL_TOKENS = ["<pad>", "<начало>", "<конец>"]


class ReplyGenerator:
    """
    Порождающая модель ответов: кодировщик и декодировщик на LSTM, как в чат-боте из лекции 10
    и в модели seq2seq из ПР5.

    Кодировщик читает леммы реплики пользователя, декодировщик по вектору идеи пишет ответ
    слово за словом. Токенами ответа служат слова вместе со знаками препинания («кадров,»),
    поэтому детокенизация сводится к соединению токенов через пробел.
    """

    def __init__(self, word_vectors, num_neurons: int = 128, decoder_dims: int = 64):
        """
        Задает гиперпараметры модели.

        :param word_vectors: модель WordVectors, векторами которой инициализируется вход кодировщика
        :param num_neurons: число нейронов в LSTM кодировщика и декодировщика
        :param decoder_dims: размерность векторов слов декодировщика
        """
        self.word_vectors = word_vectors
        self.num_neurons = num_neurons
        self.decoder_dims = decoder_dims
        self.source_vocab = []
        self.target_vocab = []
        self.source_index = {}
        self.target_index = {}
        self.max_source = 0
        self.max_target = 0
        self.model = None
        self.encoder_model = None
        self.decoder_model = None

    def build_vocabularies(self, sources: List[List[str]], replies: List[str]):
        """
        Составляет словари лемм реплик и токенов ответов.

        :param sources: реплики пользователя в виде списков лемм
        :param replies: тексты ответов
        """
        self.source_vocab = ["<pad>"] + sorted({w for s in sources for w in s})
        self.target_vocab = SPECIAL_TOKENS + sorted({t for r in replies for t in r.split()})
        self.source_index = {w: i for i, w in enumerate(self.source_vocab)}
        self.target_index = {t: i for i, t in enumerate(self.target_vocab)}
        self.max_source = max(len(s) for s in sources)
        self.max_target = max(len(r.split()) for r in replies) + 1

    def source_data(self, sources: List[List[str]]) -> np.ndarray:
        """
        Вход кодировщика: номера лемм в обратном порядке (как в ПР5), дополненные нулями.
        Незнакомые слова отбрасываются.

        :param sources: реплики в виде списков лемм
        :return: массив (число реплик, max_source)
        """
        data = np.zeros((len(sources), self.max_source), dtype="int32")
        for i, lemmas in enumerate(sources):
            ids = [self.source_index[w] for w in lemmas if w in self.source_index][:self.max_source][::-1]
            data[i, :len(ids)] = ids
        return data

    def target_data(self, replies: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Данные декодировщика для обучения «с усилением учителем».

        :param replies: тексты ответов
        :return: (вход декодировщика с <начало>, целевые номера с <конец>)
        """
        decoder_input = np.zeros((len(replies), self.max_target), dtype="int32")
        decoder_target = np.zeros((len(replies), self.max_target), dtype="int32")
        for i, reply in enumerate(replies):
            ids = [self.target_index[t] for t in reply.split()]
            decoder_input[i, :len(ids) + 1] = [START] + ids
            decoder_target[i, :len(ids) + 1] = ids + [END]
        return decoder_input, decoder_target

    def embedding_matrix(self) -> np.ndarray:
        """
        Начальные веса слоя Embedding кодировщика: векторы Word2Vec для известных лемм,
        небольшие случайные векторы для остальных.

        :return: массив (размер словаря, размерность векторов)
        """
        wv = self.word_vectors.model.wv
        rng = np.random.default_rng(1337)
        matrix = rng.normal(scale=0.1, size=(len(self.source_vocab), wv.vector_size)).astype("float32")
        matrix[PAD] = 0
        for i, word in enumerate(self.source_vocab[1:], start=1):
            if word in wv:
                matrix[i] = wv[word] / np.linalg.norm(wv[word])
        return matrix

    def build(self):
        """
        Строит модель для обучения и отдельные модели кодировщика и декодировщика для генерации.
        """
        encoder_inputs = Input(shape=(None,), dtype="int32")
        encoder_embedding = Embedding(len(self.source_vocab), self.word_vectors.model.wv.vector_size,
                                      embeddings_initializer=Constant(self.embedding_matrix()), mask_zero=True)
        _, state_h, state_c = LSTM(self.num_neurons, return_state=True)(encoder_embedding(encoder_inputs))

        decoder_inputs = Input(shape=(None,), dtype="int32")
        decoder_embedding = Embedding(len(self.target_vocab), self.decoder_dims, mask_zero=True)
        decoder_lstm = LSTM(self.num_neurons, return_sequences=True, return_state=True)
        decoder_dense = Dense(len(self.target_vocab), activation="softmax")
        outputs, _, _ = decoder_lstm(decoder_embedding(decoder_inputs), initial_state=[state_h, state_c])
        self.model = Model([encoder_inputs, decoder_inputs], decoder_dense(outputs))

        self.encoder_model = Model(encoder_inputs, [state_h, state_c])
        thought = [Input(shape=(self.num_neurons,)), Input(shape=(self.num_neurons,))]
        outputs, h, c = decoder_lstm(decoder_embedding(decoder_inputs), initial_state=thought)
        self.decoder_model = Model([decoder_inputs] + thought, [decoder_dense(outputs), h, c])

    def train(self, sources: List[List[str]], replies: List[str], epochs: int = 300,
              batch_size: int = 16, verbose: int = 0):
        """
        Обучает модель на парах «реплика – ответ». Отложенной выборки нет: каждая пара
        в корпусе одна, поэтому обучение останавливается, когда ошибка на обучающих данных
        перестает уменьшаться 10 эпох подряд.

        :param sources: реплики в виде списков лемм
        :param replies: тексты ответов
        :param epochs: максимальное число эпох
        :param batch_size: размер пакета
        :param verbose: подробность вывода Keras
        :return: история обучения
        """
        keras.utils.set_random_seed(1337)
        self.build_vocabularies(sources, replies)
        self.build()
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        decoder_input, decoder_target = self.target_data(replies)
        stopper = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
        return self.model.fit([self.source_data(sources), decoder_input], decoder_target, batch_size=batch_size,
                              epochs=epochs, callbacks=[stopper], verbose=verbose)

    def generate(self, lemmas: List[str]) -> Tuple[str, float]:
        """
        Пишет ответ на реплику: декодировщик получает <начало> и выбирает самое вероятное слово,
        пока не выдаст <конец>. Модели вызываются напрямую, а не через predict: для одного примера
        это в несколько раз быстрее. Уверенность равна средней вероятности выбранных слов.

        :param lemmas: леммы реплики пользователя
        :return: (текст ответа, уверенность от 0 до 1)
        """
        state = [np.array(s) for s in self.encoder_model(self.source_data([lemmas]), training=False)]
        target = np.array([[START]])
        words, probs = [], []
        while len(words) < self.max_target:
            output, h, c = [np.array(t) for t in self.decoder_model([target] + state, training=False)]
            token = int(output[0, -1].argmax())
            probs.append(float(output[0, -1, token]))
            if token == END:
                break
            words.append(self.target_vocab[token])
            target = np.array([[token]])
            state = [h, c]
        return " ".join(words), float(np.mean(probs))

    def save(self, path: Path):
        """
        Сохраняет веса модели и словари.

        :param path: путь без расширения, например models/generator
        """
        self.model.save_weights(str(path.with_suffix(".weights.h5")))
        path.with_suffix(".json").write_text(json.dumps({
            "source_vocab": self.source_vocab, "target_vocab": self.target_vocab,
            "max_source": self.max_source, "max_target": self.max_target,
        }, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path):
        """
        Загружает словари, заново строит модель и загружает веса.

        :param path: путь без расширения
        """
        config = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        self.source_vocab, self.target_vocab = config["source_vocab"], config["target_vocab"]
        self.source_index = {w: i for i, w in enumerate(self.source_vocab)}
        self.target_index = {t: i for i, t in enumerate(self.target_vocab)}
        self.max_source, self.max_target = config["max_source"], config["max_target"]
        self.build()
        self.model.load_weights(str(path.with_suffix(".weights.h5")))

    @staticmethod
    def exists(path: Path) -> bool:
        """
        Проверяет, сохранена ли модель.

        :param path: путь без расширения
        :return: True, если есть веса и словари
        """
        return path.with_suffix(".json").exists() and path.with_suffix(".weights.h5").exists()
