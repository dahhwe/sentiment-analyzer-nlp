from pathlib import Path

import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import (Activation, Bidirectional, Conv1D, Dense, Dropout, Flatten,
                          GlobalMaxPooling1D, Input, SimpleRNN)
from keras.models import Sequential, model_from_json

MAXLEN = 100
EMBEDDING_DIMS = 100
BATCH_SIZE = 32
EPOCHS = 15
SEED = 1337


class NeuralClassifier:
    """
    Общая часть сверточной и рекуррентной сетей: обучение, сохранение, загрузка и предсказание.

    На вход сеть получает документ как последовательность из MAXLEN векторов слов Word2Vec,
    на выходе выдает вероятности классов тональности (softmax).
    """

    def __init__(self, classes):
        """
        Создает классификатор без обученной сети.

        :param classes: список названий классов в порядке выходов сети
        """
        self.classes = list(classes)
        self.model = None

    def build(self) -> Sequential:
        """
        Строит и компилирует сеть. Реализуется в наследниках.

        :return: модель Keras
        """
        raise NotImplementedError

    def train(self, x: np.ndarray, y: np.ndarray, verbose: int = 2):
        """
        Обучает сеть. 10 % обучающей выборки отводится на проверку, обучение
        останавливается (EarlyStopping), если точность на проверке не растет 3 эпохи подряд.

        :param x: массив последовательностей (число документов, MAXLEN, EMBEDDING_DIMS)
        :param y: номера классов
        :param verbose: подробность вывода Keras
        :return: история обучения
        """
        keras.utils.set_random_seed(SEED)
        self.model = self.build()
        stopper = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
        return self.model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS,
                              validation_split=0.1, callbacks=[stopper], verbose=verbose)

    def save(self, path: Path):
        """
        Сохраняет структуру сети в JSON и веса в отдельный файл.

        :param path: путь без расширения, например models/cnn
        """
        path.with_suffix(".json").write_text(self.model.to_json(), encoding="utf-8")
        self.model.save_weights(str(path.with_suffix(".weights.h5")))

    def load(self, path: Path):
        """
        Загружает структуру сети из JSON и веса из файла.

        :param path: путь без расширения, например models/cnn
        """
        self.model = model_from_json(path.with_suffix(".json").read_text(encoding="utf-8"))
        self.model.load_weights(str(path.with_suffix(".weights.h5")))

    @staticmethod
    def exists(path: Path) -> bool:
        """
        Проверяет, сохранена ли сеть на диске.

        :param path: путь без расширения
        :return: True, если есть оба файла
        """
        return path.with_suffix(".json").exists() and path.with_suffix(".weights.h5").exists()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Вычисляет вероятности классов.

        :param x: массив последовательностей
        :return: массив вероятностей (число документов, число классов)
        """
        return self.model.predict(x, batch_size=BATCH_SIZE, verbose=0)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Доля правильно классифицированных документов.

        :param x: массив последовательностей
        :param y: номера классов
        :return: accuracy
        """
        return float((self.predict(x).argmax(axis=1) == y).mean())


class CNNClassifier(NeuralClassifier):
    """
    Сверточная сеть: одномерные фильтры скользят по последовательности векторов слов,
    GlobalMaxPooling1D оставляет по одному максимальному значению от каждого фильтра.
    """

    def __init__(self, classes, filters: int = 100, kernel_size: int = 3,
                 hidden_dims: int = 100, dropout: float = 0.5):
        """
        Задает гиперпараметры сверточной сети.

        :param classes: список названий классов
        :param filters: число фильтров Conv1D
        :param kernel_size: ширина фильтра (сколько соседних слов он видит)
        :param hidden_dims: число нейронов полносвязного слоя
        :param dropout: доля выключаемых нейронов
        """
        super().__init__(classes)
        self.filters = filters
        self.kernel_size = kernel_size
        self.hidden_dims = hidden_dims
        self.dropout = dropout

    def build(self) -> Sequential:
        """
        Conv1D -> GlobalMaxPooling1D -> Dense -> Dropout -> ReLU -> Dense -> softmax.

        :return: скомпилированная модель Keras
        """
        model = Sequential([
            Input(shape=(MAXLEN, EMBEDDING_DIMS)),
            Conv1D(self.filters, self.kernel_size, padding="valid", activation="relu", strides=1),
            GlobalMaxPooling1D(),
            Dense(self.hidden_dims),
            Dropout(self.dropout),
            Activation("relu"),
            Dense(len(self.classes)),
            Activation("softmax"),
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model


class RNNClassifier(NeuralClassifier):
    """
    Рекуррентная сеть SimpleRNN: слова подаются по одному, и на каждом шаге
    вместе с очередным словом на вход скрытого слоя поступает его выход с прошлого шага.
    """

    def __init__(self, classes, num_neurons: int = 50, dropout: float = 0.5,
                 return_sequences: bool = True, go_backwards: bool = False, bidirectional: bool = False):
        """
        Задает гиперпараметры рекуррентной сети.

        :param classes: список названий классов
        :param num_neurons: число нейронов рекуррентного слоя
        :param dropout: доля выключаемых нейронов
        :param return_sequences: передавать дальше выходы всех шагов (True) или только последнего
        :param go_backwards: читать последовательность с конца
        :param bidirectional: двунаправленный слой (Bidirectional)
        """
        super().__init__(classes)
        self.num_neurons = num_neurons
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.bidirectional = bidirectional

    def build(self) -> Sequential:
        """
        SimpleRNN -> Dropout -> Flatten -> Dense -> softmax.

        :return: скомпилированная модель Keras
        """
        rnn = SimpleRNN(self.num_neurons, return_sequences=self.return_sequences, go_backwards=self.go_backwards)
        layers = [Input(shape=(MAXLEN, EMBEDDING_DIMS)), Bidirectional(rnn) if self.bidirectional else rnn,
                  Dropout(self.dropout)]
        if self.return_sequences:
            layers.append(Flatten())
        layers.append(Dense(len(self.classes), activation="softmax"))
        model = Sequential(layers)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        return model
