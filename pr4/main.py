import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PR4_DIR = Path(__file__).resolve().parent
PR1_DIR = PR4_DIR.parent / "pr1"
PR2_DIR = PR4_DIR.parent / "pr2"
PR3_DIR = PR4_DIR.parent / "pr3"
sys.path.append(str(PR1_DIR / "src"))
sys.path.append(str(PR2_DIR / "src"))
sys.path.append(str(PR3_DIR / "src"))

from normalizer import TextNormalizer
from tokenizer import RegexTokenizer
from lsa import LSAModel
from steering import SentimentSteering
from analyzer import DocumentAnalyzer
from doc_vectors import DocVectors
from word_vectors import WordVectors

from src.networks import CNNClassifier, EMBEDDING_DIMS, MAXLEN, RNNClassifier
from src.sequences import pad_trunc, tokenize_and_vectorize

PR1_DATA = PR1_DIR / "data" / "dataset.csv"
PR1_CACHE = PR1_DIR / "data" / "cached_dataset.csv"
PR3_MODELS = PR3_DIR / "models"
MODELS_DIR = PR4_DIR / "models"

ANALYZERS = {
    "lsa": "LSA (ПР2)",
    "word2vec": "Word2Vec (ПР3)",
    "doc2vec": "Doc2Vec (ПР3)",
    "cnn": "CNN (ПР4)",
    "rnn": "SimpleRNN (ПР4)",
}
SHORT_LABELS = {"positive": "pos", "neutral": "neu", "negative": "neg"}
LABEL_MAP = {"positive": "Позитивный 😊", "negative": "Негативный 😞", "neutral": "Нейтральный 😐"}


def load_corpus(tokenizer, normalizer):
    """
    Загружает корпус из ПР1 вместе с кэшем нормализации. Если кэша нет,
    корпус нормализуется конвейером ПР1 и кэш сохраняется.

    :param tokenizer: токенизатор RegexTokenizer
    :param normalizer: нормализатор TextNormalizer
    :return: (нормализованные документы в виде списков лемм, метки тональности)
    """
    df = pd.read_csv(PR1_DATA).dropna(subset=["text", "label"])
    labels = df["label"].astype(str).tolist()
    processed = None
    if PR1_CACHE.exists():
        cached = pd.read_csv(PR1_CACHE, keep_default_na=False)
        if len(cached) == len(df):
            processed = cached["processed_text"].astype(str).tolist()
    if processed is None:
        print("=== Предобработка корпуса (токенизация и нормализация из ПР1) ===")
        processed = [" ".join(normalizer.normalize(tokenizer.tokenize(t))) for t in df["text"].astype(str)]
        pd.DataFrame({"processed_text": processed, "label": labels}).to_csv(PR1_CACHE, index=False)
    keep = [i for i, text in enumerate(processed) if text.strip()]
    return [processed[i].split() for i in keep], [labels[i] for i in keep]


def load_pr3_models():
    """
    Загружает модели Word2Vec и Doc2Vec, обученные в ПР3.

    :return: (WordVectors, DocVectors) или None, если модели ПР3 еще не обучены
    """
    word2vec_path = PR3_MODELS / "word2vec.model"
    doc2vec_path = PR3_MODELS / "doc2vec.model"
    if not (word2vec_path.exists() and doc2vec_path.exists()):
        return None
    word_vectors, doc_vectors = WordVectors(), DocVectors()
    word_vectors.load(str(word2vec_path))
    doc_vectors.load(str(doc2vec_path))
    return word_vectors, doc_vectors


def read_documents(path):
    """
    Читает корпус документов пользователя.

    :param path: TXT-файл (один документ на строку) или папка с TXT-файлами
    :return: список текстов
    """
    path = Path(path)
    if path.is_dir():
        return [f.read_text(encoding="utf-8").strip() for f in sorted(path.glob("*.txt"))]
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def short(text, length=70):
    """
    Сокращает текст для вывода в консоль.

    :param text: исходный текст
    :param length: максимальная длина
    :return: текст в одну строку не длиннее length символов
    """
    text = " ".join(text.split())
    return text if len(text) <= length else text[:length] + "..."


class SentimentAnalyzers:
    """
    Три семантических анализатора тональности, обученные на одной обучающей выборке корпуса ПР1:
    LSA со стирингом (ПР2), Word2Vec и Doc2Vec со стирингом (ПР3), CNN и SimpleRNN (ПР4).
    """

    def __init__(self, document_analyzer, word_vectors, classes):
        """
        Создает набор анализаторов.

        :param document_analyzer: DocumentAnalyzer из ПР3 (векторы LSA, Word2Vec и Doc2Vec)
        :param word_vectors: модель WordVectors из ПР3
        :param classes: список классов тональности
        """
        self.document_analyzer = document_analyzer
        self.word_vectors = word_vectors
        self.classes = classes
        self.steering = {}
        self.networks = {"cnn": CNNClassifier(classes), "rnn": RNNClassifier(classes)}

    def sequences(self, documents):
        """
        Готовит вход для нейросетей: последовательности векторов Word2Vec длины MAXLEN.

        :param documents: документы в виде списков лемм
        :return: массив (число документов, MAXLEN, EMBEDDING_DIMS)
        """
        return pad_trunc(tokenize_and_vectorize(documents, self.word_vectors), MAXLEN, EMBEDDING_DIMS)

    def train_steering(self, idx_train, y):
        """
        Обучает стиринг (LDA из ПР2) на векторах LSA, Word2Vec и Doc2Vec обучающей выборки.

        :param idx_train: номера документов обучающей выборки
        :param y: метки тональности всех документов корпуса
        """
        for method in ("lsa", "word2vec", "doc2vec"):
            steering = SentimentSteering()
            steering.train(self.document_analyzer.corpus_vectors(method)[idx_train], y[idx_train])
            self.steering[method] = steering

    def prepare_networks(self, x_train, y_train, retrain=False):
        """
        Загружает CNN и SimpleRNN из папки models или обучает их на обучающей выборке.

        :param x_train: последовательности обучающей выборки
        :param y_train: номера классов обучающей выборки
        :param retrain: обучить сети заново, даже если они сохранены
        """
        MODELS_DIR.mkdir(exist_ok=True)
        for name, network in self.networks.items():
            path = MODELS_DIR / name
            if not retrain and network.exists(path):
                network.load(path)
                continue
            print(f"\n=== Обучение {ANALYZERS[name]} ===")
            start = time.time()
            network.train(x_train, y_train)
            network.model.summary()
            network.save(path)
            relative = path.relative_to(PR4_DIR).as_posix()
            print(f"Время обучения: {time.time() - start:.0f} с, модель сохранена в {relative}.json и {relative}.weights.h5")

    def predict(self, documents):
        """
        Определяет тональность документов всеми анализаторами.

        :param documents: документы в виде списков лемм
        :return: словарь {анализатор: массив вероятностей (число документов, число классов)}
        """
        result = {}
        for method, steering in self.steering.items():
            vectors = self.document_analyzer.vectors(documents, method)
            result[method] = steering.model.predict_proba(vectors)
        x = self.sequences(documents)
        for name, network in self.networks.items():
            result[name] = network.predict(x)
        return result


def print_text(analyzers, normalize, text):
    """
    Выводит тональность одного текста по каждому анализатору.

    :param analyzers: SentimentAnalyzers
    :param normalize: функция, переводящая текст в список лемм
    :param text: исходный текст
    """
    lemmas = normalize(text)
    known = sum(1 for t in lemmas if t in analyzers.word_vectors.model.wv)
    print("\n--- Анализ текста ---")
    print(f"Исходный текст:      {text}")
    print(f"После нормализации:  {lemmas}")
    print(f"Слов с векторами Word2Vec: {known} из {len(lemmas)}")
    print(f"{'Анализатор':<20}{'Тональность':<18}" + "".join(f"{c:<10}" for c in analyzers.classes))
    for name, probs in analyzers.predict([lemmas]).items():
        label = LABEL_MAP[analyzers.classes[int(probs[0].argmax())]]
        print(f"{ANALYZERS[name]:<20}{label:<18}" + "".join(f"{p:<10.2f}" for p in probs[0]))


def print_corpus(analyzers, normalize, path):
    """
    Обрабатывает корпус документов пользователя: тональность каждого документа
    по всем анализаторам и распределение тональности по корпусу.

    :param analyzers: SentimentAnalyzers
    :param normalize: функция, переводящая текст в список лемм
    :param path: путь к корпусу
    """
    texts = read_documents(path)
    predictions = analyzers.predict([normalize(t) for t in texts])
    labels = {name: [analyzers.classes[i] for i in probs.argmax(axis=1)] for name, probs in predictions.items()}

    print(f"\n--- Обработка корпуса: {path} ({len(texts)} документов) ---")
    print("  №   " + "".join(f"{name.upper():<9}" for name in labels) + "Текст")
    for i, text in enumerate(texts):
        row = "".join(f"{SHORT_LABELS[labels[name][i]]:<9}" for name in labels)
        print(f"{i + 1:>3}   {row}{short(text)}")

    print("\nРаспределение тональности по корпусу:")
    print(f"{'Анализатор':<20}" + "".join(f"{c:<10}" for c in analyzers.classes))
    for name, values in labels.items():
        print(f"{ANALYZERS[name]:<20}" + "".join(f"{values.count(c):<10}" for c in analyzers.classes))


def print_evaluation(analyzers, x_test, idx_test, y):
    """
    Сравнивает все анализаторы по точности на тестовой выборке корпуса ПР1.

    :param analyzers: SentimentAnalyzers
    :param x_test: последовательности тестовой выборки
    :param idx_test: номера документов тестовой выборки
    :param y: метки тональности всех документов корпуса
    """
    print("\n--- Сравнение анализаторов на тестовой выборке корпуса ПР1 (30%) ---")
    print(f"{'Анализатор':<20}{'Метод':<46}Accuracy")
    for method, steering in analyzers.steering.items():
        vectors = analyzers.document_analyzer.corpus_vectors(method)[idx_test]
        accuracy = (steering.model.predict(vectors) == y[idx_test]).mean()
        print(f"{ANALYZERS[method]:<20}{'вектор документа + LDA (стиринг)':<46}{accuracy:.2f}")
    y_test = np.array([analyzers.classes.index(label) for label in y[idx_test]])
    for name, network in analyzers.networks.items():
        method = f"последовательность Word2Vec, {network.model.count_params()} параметров"
        print(f"{ANALYZERS[name]:<20}{method:<46}{network.accuracy(x_test, y_test):.2f}")


def main():
    """
    Точка входа: разбор аргументов командной строки и запуск нужного режима.
    """
    parser = argparse.ArgumentParser(description="Семантический анализатор корпуса документов (CNN, SimpleRNN)")
    parser.add_argument("--text", type=str, help="Определить тональность текста всеми анализаторами")
    parser.add_argument("--corpus", type=str, help="Корпус пользователя: TXT (документ на строку) или папка с TXT")
    parser.add_argument("--evaluate", action="store_true", help="Сравнить анализаторы на тестовой выборке")
    parser.add_argument("--train", action="store_true", help="Обучить CNN и SimpleRNN заново")
    args = parser.parse_args()

    if not any([args.text, args.corpus, args.evaluate, args.train]):
        parser.print_help()
        return
    if args.corpus and not Path(args.corpus).exists():
        print(f"Файл {args.corpus} не найден!")
        return

    pr3_models = load_pr3_models()
    if pr3_models is None:
        print("Модели Word2Vec и Doc2Vec из ПР3 не найдены. Сначала выполните: python ../pr3/main.py --train")
        return
    word_vectors, doc_vectors = pr3_models

    tokenizer = RegexTokenizer()
    normalizer = TextNormalizer(use_lemmatization=True, use_stemming=False)
    documents, labels = load_corpus(tokenizer, normalizer)
    if doc_vectors.corpus_size != len(documents):
        print("Модель Doc2Vec из ПР3 обучена на другом корпусе. Выполните: python ../pr3/main.py --train")
        return

    lsa = LSAModel(n_topics=100)
    lsa.fit([" ".join(doc) for doc in documents])
    document_analyzer = DocumentAnalyzer(lsa, word_vectors, doc_vectors, documents, tokenizer, normalizer)

    y = np.array(labels)
    classes = sorted(set(labels))
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.3, random_state=42)

    analyzers = SentimentAnalyzers(document_analyzer, word_vectors, classes)
    analyzers.train_steering(idx_train, y)
    x = analyzers.sequences(documents)
    y_index = np.array([classes.index(label) for label in labels])
    analyzers.prepare_networks(x[idx_train], y_index[idx_train], retrain=args.train)

    if args.text:
        print_text(analyzers, document_analyzer.normalize_text, args.text)
    elif args.corpus:
        print_corpus(analyzers, document_analyzer.normalize_text, args.corpus)
    elif args.evaluate:
        print_evaluation(analyzers, x[idx_test], idx_test, y)


if __name__ == "__main__":
    main()
