import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PR5_DIR = Path(__file__).resolve().parent
PR1_DIR = PR5_DIR.parent / "pr1"
PR2_DIR = PR5_DIR.parent / "pr2"
PR3_DIR = PR5_DIR.parent / "pr3"
PR4_DIR = PR5_DIR.parent / "pr4"
sys.path.append(str(PR1_DIR / "src"))
sys.path.append(str(PR2_DIR / "src"))
sys.path.append(str(PR3_DIR / "src"))
sys.path.append(str(PR4_DIR / "src"))

from normalizer import TextNormalizer
from tokenizer import RegexTokenizer
from lsa import LSAModel
from steering import SentimentSteering
from analyzer import DocumentAnalyzer
from doc_vectors import DocVectors
from word_vectors import WordVectors
from networks import CNNClassifier, EMBEDDING_DIMS, MAXLEN, RNNClassifier
from sequences import pad_trunc, tokenize_and_vectorize

from src.generator import TemplateGenerator
from src.lstm_classifier import LSTMClassifier
from src.patterns import extract_facts
from src.seq2seq import Seq2SeqAutoencoder

PR1_DATA = PR1_DIR / "data" / "dataset.csv"
PR1_CACHE = PR1_DIR / "data" / "cached_dataset.csv"
PR3_MODELS = PR3_DIR / "models"
PR4_MODELS = PR4_DIR / "models"
MODELS_DIR = PR5_DIR / "models"

ANALYZERS = {
    "lsa": "LSA (ПР2)",
    "word2vec": "Word2Vec (ПР3)",
    "doc2vec": "Doc2Vec (ПР3)",
    "cnn": "CNN (ПР4)",
    "rnn": "SimpleRNN (ПР4)",
    "lstm": "LSTM (ПР5)",
    "seq2seq": "Seq2seq (ПР5)",
}
STEERING_METHODS = ("lsa", "word2vec", "doc2vec", "seq2seq")
SHORT_LABELS = {"positive": "pos", "neutral": "neu", "negative": "neg"}
LABEL_MAP = {"positive": "Позитивный 😊", "negative": "Негативный 😞", "neutral": "Нейтральный 😐"}


def load_corpus(tokenizer, normalizer):
    """
    Загружает корпус из ПР1 вместе с кэшем нормализации. Если кэша нет,
    корпус нормализуется конвейером ПР1 и кэш сохраняется.

    :param tokenizer: токенизатор RegexTokenizer
    :param normalizer: нормализатор TextNormalizer
    :return: (исходные тексты, документы в виде списков лемм, метки тональности)
    """
    df = pd.read_csv(PR1_DATA).dropna(subset=["text", "label"])
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    processed = None
    if PR1_CACHE.exists():
        cached = pd.read_csv(PR1_CACHE, keep_default_na=False)
        if len(cached) == len(df):
            processed = cached["processed_text"].astype(str).tolist()
    if processed is None:
        print("=== Предобработка корпуса (токенизация и нормализация из ПР1) ===")
        processed = [" ".join(normalizer.normalize(tokenizer.tokenize(t))) for t in texts]
        pd.DataFrame({"processed_text": processed, "label": labels}).to_csv(PR1_CACHE, index=False)
    keep = [i for i, text in enumerate(processed) if text.strip()]
    return [texts[i] for i in keep], [processed[i].split() for i in keep], [labels[i] for i in keep]


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


def print_wrapped(title, text, width=110):
    """
    Выводит сгенерированный текст с переносом строк по словам.

    :param title: заголовок блока
    :param text: текст
    :param width: ширина строки
    """
    print(f"\n{title}:")
    line = ""
    for word in text.split():
        if line and len(line) + len(word) + 1 > width:
            print(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    print(line)


class SemanticAnalyzers:
    """
    Четыре семантических анализатора, обученных на одном корпусе ПР1:
    LSA (ПР2), Word2Vec и Doc2Vec (ПР3), CNN и SimpleRNN (ПР4), LSTM и seq2seq (ПР5).
    """

    def __init__(self, document_analyzer, word_vectors, corpus_texts, corpus_labels, classes):
        """
        Создает набор анализаторов.

        :param document_analyzer: DocumentAnalyzer из ПР3 (векторы LSA, Word2Vec и Doc2Vec)
        :param word_vectors: модель WordVectors из ПР3
        :param corpus_texts: исходные тексты корпуса (для поиска похожих документов)
        :param corpus_labels: метки тональности корпуса
        :param classes: список классов тональности
        """
        self.document_analyzer = document_analyzer
        self.word_vectors = word_vectors
        self.corpus_texts = corpus_texts
        self.corpus_labels = corpus_labels
        self.classes = classes
        self.steering = {}
        self.networks = {"cnn": CNNClassifier(classes), "rnn": RNNClassifier(classes)}
        self.lstm = LSTMClassifier(classes)
        self.seq2seq = Seq2SeqAutoencoder(word_vectors)
        self.corpus_thoughts = None
        self.corpus_search = None

    def sequences(self, documents):
        """
        Готовит вход для нейросетей: последовательности векторов Word2Vec длины MAXLEN.

        :param documents: документы в виде списков лемм
        :return: массив (число документов, MAXLEN, EMBEDDING_DIMS)
        """
        return pad_trunc(tokenize_and_vectorize(documents, self.word_vectors), MAXLEN, EMBEDDING_DIMS)

    def load_pr4_networks(self):
        """
        Загружает CNN и SimpleRNN, обученные в ПР4.

        :return: True, если обе сети найдены
        """
        for name, network in self.networks.items():
            if not network.exists(PR4_MODELS / name):
                return False
            network.load(PR4_MODELS / name)
        return True

    def prepare_models(self, documents, x_train, y_train, retrain=False):
        """
        Загружает LSTM и seq2seq из папки models или обучает их.
        LSTM обучается на обучающей выборке, автокодировщик seq2seq обучается без учителя
        на всем корпусе, как LSA и Doc2Vec в предыдущих работах.

        :param documents: все документы корпуса в виде списков лемм
        :param x_train: последовательности обучающей выборки
        :param y_train: номера классов обучающей выборки
        :param retrain: обучить модели заново, даже если они сохранены
        """
        MODELS_DIR.mkdir(exist_ok=True)
        path = MODELS_DIR / "lstm"
        if not retrain and self.lstm.exists(path):
            self.lstm.load(path)
        else:
            print("\n=== Обучение LSTM (ПР5) ===")
            start = time.time()
            self.lstm.train(x_train, y_train)
            self.lstm.model.summary()
            self.lstm.save(path)
            print(f"Время обучения: {time.time() - start:.0f} с, модель сохранена в models/lstm.json и models/lstm.weights.h5")

        path = MODELS_DIR / "seq2seq"
        if not retrain and self.seq2seq.exists(path):
            self.seq2seq.load(path)
        else:
            print("\n=== Обучение seq2seq (ПР5) ===")
            start = time.time()
            self.seq2seq.train(documents)
            self.seq2seq.model.summary()
            self.seq2seq.save(path)
            print(f"Время обучения: {time.time() - start:.0f} с, модель сохранена в models/seq2seq.json и models/seq2seq.weights.h5")
        self.corpus_thoughts = self.seq2seq.thought_vectors(documents)
        self.corpus_search = self.seq2seq.search_vectors(documents)

    def corpus_vectors(self, method):
        """
        Векторы документов корпуса для стиринга.

        :param method: "lsa", "word2vec", "doc2vec" или "seq2seq"
        :return: матрица векторов
        """
        if method == "seq2seq":
            return self.corpus_thoughts
        return self.document_analyzer.corpus_vectors(method)

    def vectors(self, documents, method):
        """
        Векторы новых документов для стиринга.

        :param documents: документы в виде списков лемм
        :param method: "lsa", "word2vec", "doc2vec" или "seq2seq"
        :return: матрица векторов
        """
        if method == "seq2seq":
            return self.seq2seq.thought_vectors(documents)
        return self.document_analyzer.vectors(documents, method)

    def train_steering(self, idx_train, y):
        """
        Обучает стиринг (LDA из ПР2) на векторах документов обучающей выборки.

        :param idx_train: номера документов обучающей выборки
        :param y: метки тональности всех документов корпуса
        """
        for method in STEERING_METHODS:
            steering = SentimentSteering()
            steering.train(self.corpus_vectors(method)[idx_train], y[idx_train])
            self.steering[method] = steering

    def predict(self, documents):
        """
        Определяет тональность документов всеми анализаторами.

        :param documents: документы в виде списков лемм
        :return: словарь {анализатор: массив вероятностей (число документов, число классов)}
        """
        x = self.sequences(documents)
        result = {}
        for method in ANALYZERS:
            if method in self.steering:
                result[method] = self.steering[method].model.predict_proba(self.vectors(documents, method))
            elif method == "lstm":
                result[method] = self.lstm.predict(x)
            else:
                result[method] = self.networks[method].predict(x)
        return result

    def most_similar(self, tokens, vector):
        """
        Находит самый похожий документ корпуса по выходу кодировщика seq2seq.
        Документ, совпадающий с запросом (сходство больше 0.999), пропускается. Для текстов
        короче 5 слов из словаря seq2seq поиск не выполняется: по паре слов вектор идеи
        получается неточным, и найденный документ обычно не связан с запросом.

        :param tokens: леммы документа
        :param vector: вектор документа для поиска (search_vectors)
        :return: (текст, тональность, косинусное сходство) или None
        """
        if len(self.seq2seq.to_ids(tokens)) < 5:
            return None
        scores = self.corpus_search @ vector
        for i in np.argsort(-scores):
            if scores[i] < 0.999:
                return self.corpus_texts[i], self.corpus_labels[i], float(scores[i])
        return None

    def analyze(self, texts, normalize):
        """
        Полный анализ документов, результаты которого заполняют шаблоны генератора.

        :param texts: исходные тексты
        :param normalize: функция, переводящая текст в список лемм
        :return: список словарей с результатами анализа каждого документа
        """
        documents = [normalize(t) for t in texts]
        predictions = self.predict(documents)
        search = self.seq2seq.search_vectors(documents)
        analyses = []
        for i, text in enumerate(texts):
            probs = predictions["lstm"][i]
            analyses.append({
                "text": text,
                "lemmas": documents[i],
                "label": self.classes[int(probs.argmax())],
                "probability": float(probs.max()),
                "probabilities": dict(zip(self.classes, probs)),
                "votes": {name: self.classes[int(p[i].argmax())] for name, p in predictions.items() if name != "lstm"},
                "all_probabilities": {name: p[i] for name, p in predictions.items()},
                "keywords": self.lstm.key_words(documents[i], self.word_vectors),
                "facts": extract_facts(text),
                "similar": self.most_similar(documents[i], search[i]),
                "search": search[i],
            })
        return analyses


def print_text(analyzers, generator, normalize, text):
    """
    Анализирует один текст всеми анализаторами и выводит сгенерированные заключение и ответ.

    :param analyzers: SemanticAnalyzers
    :param generator: TemplateGenerator
    :param normalize: функция, переводящая текст в список лемм
    :param text: исходный текст
    """
    analysis = analyzers.analyze([text], normalize)[0]
    print("\n--- Анализ текста ---")
    print(f"Исходный текст:      {text}")
    print(f"После нормализации:  {analysis['lemmas']}")
    print(f"Восстановлено seq2seq по вектору идеи: {analyzers.seq2seq.decode(analysis['lemmas'])}")
    print(f"{'Анализатор':<20}{'Тональность':<18}" + "".join(f"{c:<10}" for c in analyzers.classes))
    for name, probs in analysis["all_probabilities"].items():
        label = LABEL_MAP[analyzers.classes[int(probs.argmax())]]
        print(f"{ANALYZERS[name]:<20}{label:<18}" + "".join(f"{p:<10.2f}" for p in probs))
    print_wrapped("Заключение (генератор на шаблонах)", generator.conclusion(analysis))
    print_wrapped("Ответ автору (генератор на шаблонах)", generator.reply(analysis))


def print_corpus(analyzers, generator, normalize, path):
    """
    Обрабатывает корпус документов пользователя: тональность каждого документа по всем
    анализаторам и сводка по корпусу, сгенерированная по шаблонам.

    :param analyzers: SemanticAnalyzers
    :param generator: TemplateGenerator
    :param normalize: функция, переводящая текст в список лемм
    :param path: путь к корпусу
    """
    texts = read_documents(path)
    analyses = analyzers.analyze(texts, normalize)
    names = list(ANALYZERS)

    print(f"\n--- Обработка корпуса: {path} ({len(texts)} документов) ---")
    print("  №   " + "".join(f"{name.upper():<9}" for name in names) + "Текст")
    for i, analysis in enumerate(analyses):
        votes = dict(analysis["votes"], lstm=analysis["label"])
        row = "".join(f"{SHORT_LABELS[votes[name]]:<9}" for name in names)
        print(f"{i + 1:>3}   {row}{short(analysis['text'], 50)}")

    vectors = np.array([a["search"] for a in analyses])
    similarity = vectors @ vectors.T
    pairs = sorted(((similarity[i, j], i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))), reverse=True)
    print_wrapped("Сводка по корпусу (генератор на шаблонах)",
                  generator.corpus_report(analyses, [(i, j, float(s)) for s, i, j in pairs[:3]]))


def print_evaluation(analyzers, x_test, idx_test, y):
    """
    Сравнивает все анализаторы по точности на тестовой выборке корпуса ПР1.

    :param analyzers: SemanticAnalyzers
    :param x_test: последовательности тестовой выборки
    :param idx_test: номера документов тестовой выборки
    :param y: метки тональности всех документов корпуса
    """
    print("\n--- Сравнение анализаторов на тестовой выборке корпуса ПР1 (30%) ---")
    print(f"{'Анализатор':<20}{'Метод':<46}Accuracy")
    y_test = np.array([analyzers.classes.index(label) for label in y[idx_test]])
    for name in ANALYZERS:
        if name in analyzers.steering:
            vectors = analyzers.corpus_vectors(name)[idx_test]
            accuracy = (analyzers.steering[name].model.predict(vectors) == y[idx_test]).mean()
            method = "вектор идеи + LDA (стиринг)" if name == "seq2seq" else "вектор документа + LDA (стиринг)"
        else:
            network = analyzers.lstm if name == "lstm" else analyzers.networks[name]
            accuracy = network.accuracy(x_test, y_test)
            method = f"последовательность Word2Vec, {network.model.count_params()} параметров"
        print(f"{ANALYZERS[name]:<20}{method:<46}{accuracy:.2f}")


def main():
    """
    Точка входа: разбор аргументов командной строки и запуск нужного режима.
    """
    parser = argparse.ArgumentParser(description="Семантический анализатор с генератором текста на шаблонах (LSTM, seq2seq)")
    parser.add_argument("--text", type=str, help="Проанализировать текст и сгенерировать заключение и ответ")
    parser.add_argument("--corpus", type=str, help="Корпус пользователя: TXT (документ на строку) или папка с TXT")
    parser.add_argument("--evaluate", action="store_true", help="Сравнить анализаторы на тестовой выборке")
    parser.add_argument("--train", action="store_true", help="Обучить LSTM и seq2seq заново")
    parser.add_argument("--seed", type=int, default=0, help="Зерно для выбора вариантов шаблонов (по умолчанию 0)")
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
    texts, documents, labels = load_corpus(tokenizer, normalizer)
    if doc_vectors.corpus_size != len(documents):
        print("Модель Doc2Vec из ПР3 обучена на другом корпусе. Выполните: python ../pr3/main.py --train")
        return

    lsa = LSAModel(n_topics=100)
    lsa.fit([" ".join(doc) for doc in documents])
    document_analyzer = DocumentAnalyzer(lsa, word_vectors, doc_vectors, documents, tokenizer, normalizer)

    y = np.array(labels)
    classes = sorted(set(labels))
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.3, random_state=42)

    analyzers = SemanticAnalyzers(document_analyzer, word_vectors, texts, labels, classes)
    if not analyzers.load_pr4_networks():
        print("Сети CNN и SimpleRNN из ПР4 не найдены. Сначала выполните: python ../pr4/main.py --train")
        return
    x = analyzers.sequences(documents)
    y_index = np.array([classes.index(label) for label in labels])
    analyzers.prepare_models(documents, x[idx_train], y_index[idx_train], retrain=args.train)
    analyzers.train_steering(idx_train, y)

    generator = TemplateGenerator({name: title.split(" (")[0] for name, title in ANALYZERS.items()}, seed=args.seed)
    if args.text:
        print_text(analyzers, generator, document_analyzer.normalize_text, args.text)
    elif args.corpus:
        print_corpus(analyzers, generator, document_analyzer.normalize_text, args.corpus)
    elif args.evaluate:
        print_evaluation(analyzers, x[idx_test], idx_test, y)


if __name__ == "__main__":
    main()
