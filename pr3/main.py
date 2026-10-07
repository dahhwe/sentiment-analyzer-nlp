import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PR3_DIR = Path(__file__).resolve().parent
PR1_DIR = PR3_DIR.parent / "pr1"
PR2_DIR = PR3_DIR.parent / "pr2"
sys.path.append(str(PR1_DIR / "src"))
sys.path.append(str(PR2_DIR / "src"))

from normalizer import TextNormalizer
from tokenizer import RegexTokenizer
from lsa import LSAModel
from steering import SentimentSteering

from src.analyzer import DocumentAnalyzer, METHOD_NAMES
from src.doc_vectors import DocVectors
from src.word_vectors import WordVectors

PR1_DATA = PR1_DIR / "data" / "dataset.csv"
PR1_CACHE = PR1_DIR / "data" / "cached_dataset.csv"
MODELS_DIR = PR3_DIR / "models"
WORD2VEC_PATH = MODELS_DIR / "word2vec.model"
DOC2VEC_PATH = MODELS_DIR / "doc2vec.model"


def preprocess_corpus(texts, tokenizer, normalizer):
    """
    Нормализует корпус конвейером из ПР1.

    :param texts: исходные тексты
    :param tokenizer: токенизатор RegexTokenizer
    :param normalizer: нормализатор TextNormalizer
    :return: список нормализованных текстов (леммы через пробел)
    """
    processed = []
    total = len(texts)
    for i, text in enumerate(texts):
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"   Обработано {i + 1}/{total} документов...", end="\r")
        processed.append(" ".join(normalizer.normalize(tokenizer.tokenize(text))))
    print()
    return processed


def load_corpus(tokenizer, normalizer):
    """
    Загружает корпус из ПР1. Нормализованные тексты берутся из кэша ПР1,
    а если его нет, корпус нормализуется заново и кэш сохраняется.

    :param tokenizer: токенизатор RegexTokenizer
    :param normalizer: нормализатор TextNormalizer
    :return: (исходные тексты, нормализованные тексты, метки тональности)
    """
    df = pd.read_csv(PR1_DATA).dropna(subset=["text", "label"])
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()

    processed = None
    if PR1_CACHE.exists():
        cached = pd.read_csv(PR1_CACHE, keep_default_na=False)
        if len(cached) == len(texts):
            processed = cached["processed_text"].astype(str).tolist()
    if processed is None:
        print("=== Предобработка корпуса (токенизация и нормализация из ПР1) ===")
        processed = preprocess_corpus(texts, tokenizer, normalizer)
        pd.DataFrame({"processed_text": processed, "label": labels}).to_csv(PR1_CACHE, index=False)

    keep = [i for i, text in enumerate(processed) if text.strip()]
    return [texts[i] for i in keep], [processed[i] for i in keep], [labels[i] for i in keep]


def split_sentences(text, tokenizer, normalizer):
    """
    Разбивает документ на предложения и нормализует каждое из них.

    :param text: исходный текст документа
    :param tokenizer: токенизатор RegexTokenizer
    :param normalizer: нормализатор TextNormalizer
    :return: список предложений, каждое предложение задано списком лемм
    """
    sentences = []
    for sentence in re.split(r"[.!?…]+", text):
        lemmas = normalizer.normalize(tokenizer.tokenize(sentence))
        if len(lemmas) > 1:
            sentences.append(lemmas)
    return sentences


def load_or_train_models(texts, processed, tokenizer, normalizer, retrain=False):
    """
    Загружает модели Word2Vec и Doc2Vec из папки models или обучает их заново.

    :param texts: исходные тексты корпуса
    :param processed: нормализованные тексты корпуса
    :param tokenizer: токенизатор RegexTokenizer
    :param normalizer: нормализатор TextNormalizer
    :param retrain: обучить модели заново, даже если они уже сохранены
    :return: (WordVectors, DocVectors)
    """
    word_vectors = WordVectors()
    doc_vectors = DocVectors()
    if not retrain and WORD2VEC_PATH.exists() and DOC2VEC_PATH.exists():
        word_vectors.load(str(WORD2VEC_PATH))
        doc_vectors.load(str(DOC2VEC_PATH))
        if doc_vectors.corpus_size == len(processed):
            return word_vectors, doc_vectors

    print("=== Обучение моделей (выполняется один раз, модели сохраняются в папку models) ===")
    MODELS_DIR.mkdir(exist_ok=True)

    start = time.time()
    sentences = []
    for text in texts:
        sentences.extend(split_sentences(text, tokenizer, normalizer))
    word_vectors.train(sentences)
    word_vectors.save(str(WORD2VEC_PATH))
    print(f"Word2Vec: {len(sentences)} предложений, словарь {word_vectors.vocab_size} слов "
          f"({time.time() - start:.1f} с)")

    start = time.time()
    doc_vectors.train([doc.split() for doc in processed])
    doc_vectors.save(str(DOC2VEC_PATH))
    print(f"Doc2Vec: {doc_vectors.corpus_size} документов ({time.time() - start:.1f} с)")
    return word_vectors, doc_vectors


def read_documents(path):
    """
    Читает набор документов пользователя.

    :param path: TXT-файл (один документ на строку) или папка с TXT-файлами
    :return: список текстов
    """
    path = Path(path)
    if path.is_dir():
        return [f.read_text(encoding="utf-8").strip() for f in sorted(path.glob("*.txt"))]
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def short(text, length=100):
    """
    Сокращает текст для вывода в консоль.

    :param text: исходный текст
    :param length: максимальная длина
    :return: текст в одну строку не длиннее length символов
    """
    text = " ".join(text.split())
    return text if len(text) <= length else text[:length] + "..."


def to_lemma(analyzer, word):
    """
    Приводит слово к лемме конвейером ПР1.

    :param analyzer: DocumentAnalyzer
    :param word: слово в любой форме
    :return: лемма или None, если слово отброшено как стоп-слово
    """
    lemmas = analyzer.normalize_text(word)
    return lemmas[0] if lemmas else None


def print_similar_words(analyzer, word, topn=10):
    """
    Выводит близкие слова по Word2Vec и по LSA рядом для сравнения.

    :param analyzer: DocumentAnalyzer
    :param word: слово
    :param topn: число слов
    """
    lemma = to_lemma(analyzer, word)
    if lemma is None or not analyzer.word_vectors.has_word(lemma):
        print(f"Слово «{word}» отсутствует в словаре Word2Vec")
        return
    w2v_words = analyzer.word_vectors.similar_words(lemma, topn)
    lsa_words = analyzer.lsa_similar_words(lemma, topn)
    print(f"\n--- Слова, близкие к «{lemma}» ---")
    print(f"   {'Word2Vec':<28}{'LSA (ПР2)'}")
    for i in range(topn):
        left = f"{w2v_words[i][0]} ({w2v_words[i][1]:.2f})" if i < len(w2v_words) else ""
        right = f"{lsa_words[i][0]} ({lsa_words[i][1]:.2f})" if i < len(lsa_words) else ""
        print(f"   {left:<28}{right}")


def print_analogy(analyzer, a, b, c):
    """
    Выводит решение задачи на аналогию «a относится к b, как ? к c».

    :param analyzer: DocumentAnalyzer
    :param a: первое слово пары-образца
    :param b: второе слово пары-образца
    :param c: слово, для которого ищется пара
    """
    lemmas = [to_lemma(analyzer, w) for w in (a, b, c)]
    missing = [w for w, lemma in zip((a, b, c), lemmas) if lemma is None or not analyzer.word_vectors.has_word(lemma)]
    if missing:
        print(f"Слова отсутствуют в словаре Word2Vec: {missing}")
        return
    print(f"\n--- Аналогия: {lemmas[0]} : {lemmas[1]} :: ? : {lemmas[2]} ---")
    print(f"Вектор: {lemmas[0]} - {lemmas[1]} + {lemmas[2]}")
    for word, score in analyzer.word_vectors.analogy(*lemmas):
        print(f"   {word} ({score:.2f})")


def print_odd_one_out(analyzer, words):
    """
    Выводит лишнее по смыслу слово из списка.

    :param analyzer: DocumentAnalyzer
    :param words: список слов
    """
    lemmas = [to_lemma(analyzer, w) for w in words]
    lemmas = [w for w in lemmas if w is not None and analyzer.word_vectors.has_word(w)]
    if len(lemmas) < 3:
        print("Нужно хотя бы три слова из словаря Word2Vec")
        return
    print(f"\n--- Лишнее слово в списке {lemmas} ---")
    print(f"   {analyzer.word_vectors.odd_one_out(lemmas)}")


def print_search(analyzer, texts, labels, query):
    """
    Выводит результаты поиска по корпусу каждым анализатором.

    :param analyzer: DocumentAnalyzer
    :param texts: исходные тексты корпуса
    :param labels: метки тональности корпуса
    :param query: текст запроса
    """
    print("\n--- Поиск похожих документов ---")
    print(f"Запрос:              {query}")
    print(f"После нормализации:  {analyzer.normalize_text(query)}")
    for method, name in METHOD_NAMES.items():
        print(f"{name}:")
        for rank, (idx, score) in enumerate(analyzer.search(query, method), 1):
            print(f"   {rank}. ({score:.2f}) [{labels[idx]}] {short(texts[idx])}")


def print_compare(analyzer, text_a, text_b):
    """
    Выводит сходство двух текстов по каждому анализатору.

    :param analyzer: DocumentAnalyzer
    :param text_a: первый текст
    :param text_b: второй текст
    """
    print("\n--- Сравнение текстов ---")
    print(f"Текст 1:  {text_a}  ->  {analyzer.normalize_text(text_a)}")
    print(f"Текст 2:  {text_b}  ->  {analyzer.normalize_text(text_b)}")
    print("Косинусное сходство:")
    for method, score in analyzer.compare(text_a, text_b).items():
        print(f"   {METHOD_NAMES[method]:<36}{score:.3f}")


def print_documents(analyzer, path, n_clusters):
    """
    Обрабатывает набор документов пользователя анализаторами LSA и Doc2Vec:
    выводит матрицу попарного сходства, самые похожие пары и группы документов.

    :param analyzer: DocumentAnalyzer
    :param path: путь к набору документов
    :param n_clusters: число групп для метода k-средних
    """
    documents = read_documents(path)
    if len(documents) < 2:
        print("В наборе должно быть хотя бы два документа")
        return
    print(f"\n--- Набор документов: {path} ({len(documents)} шт.) ---")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {short(doc, 90)}")

    n = len(documents)
    for method in ("lsa", "doc2vec"):
        matrix, clusters = analyzer.process_documents(documents, method, n_clusters)
        print(f"\n=== {METHOD_NAMES[method]} ===")
        print("Матрица косинусного сходства:")
        print("     " + "".join(f"{j:>6}" for j in range(1, n + 1)))
        for i in range(n):
            print(f"{i + 1:>5}" + "".join(f"{matrix[i, j]:>6.2f}" for j in range(n)))

        pairs = sorted(((matrix[i, j], i, j) for i in range(n) for j in range(i + 1, n)), reverse=True)
        print("Самые похожие пары:")
        for score, i, j in pairs[:n // 2]:
            print(f"   {i + 1} и {j + 1}: {score:.2f}")

        print(f"Группы документов (метод k-средних, k={len(set(clusters))}):")
        for group, cluster in enumerate(dict.fromkeys(clusters), 1):
            members = [str(i + 1) for i in range(n) if clusters[i] == cluster]
            print(f"   группа {group}: документы {', '.join(members)}")


def evaluate(analyzer, labels):
    """
    Сравнивает анализаторы на корпусе ПР1: стиринг (LDA из ПР2) обучается
    на векторах документов каждого анализатора и проверяется на тестовой выборке.

    :param analyzer: DocumentAnalyzer
    :param labels: метки тональности корпуса
    """
    y = np.array(labels)
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.3, random_state=42)
    print("\n--- Сравнение анализаторов: стиринг (LDA из ПР2) на тестовой выборке 30% ---")
    print(f"{'Анализатор':<36}{'Размерность':<14}Accuracy")
    for method, name in METHOD_NAMES.items():
        vectors = analyzer.corpus_vectors(method)
        steering = SentimentSteering()
        steering.train(vectors[idx_train], y[idx_train])
        accuracy = (steering.model.predict(vectors[idx_test]) == y[idx_test]).mean()
        print(f"{name:<36}{vectors.shape[1]:<14}{accuracy:.2f}")


def main():
    """
    Точка входа: разбор аргументов командной строки и запуск нужного режима.
    """
    parser = argparse.ArgumentParser(description="Семантический анализатор набора документов (Word2Vec, Doc2Vec, LSA)")
    parser.add_argument("--words", type=str, help="Близкие по смыслу слова (Word2Vec и LSA)")
    parser.add_argument("--analogy", nargs=3, metavar=("A", "B", "C"), help="Аналогия: A относится к B, как ? к C")
    parser.add_argument("--odd", nargs="+", metavar="WORD", help="Найти лишнее слово в списке")
    parser.add_argument("--search", type=str, help="Найти в корпусе документы, похожие на текст")
    parser.add_argument("--compare", nargs=2, metavar=("TEXT1", "TEXT2"), help="Сравнить два текста")
    parser.add_argument("--docs", type=str, help="Набор документов: TXT (документ на строку) или папка с TXT")
    parser.add_argument("--clusters", type=int, default=3, help="Число групп для набора документов")
    parser.add_argument("--evaluate", action="store_true", help="Сравнить анализаторы на корпусе ПР1")
    parser.add_argument("--train", action="store_true", help="Обучить модели Word2Vec и Doc2Vec заново")
    args = parser.parse_args()

    actions = [args.words, args.analogy, args.odd, args.search, args.compare, args.docs, args.evaluate, args.train]
    if not any(actions):
        parser.print_help()
        return
    if args.docs and not Path(args.docs).exists():
        print(f"Файл {args.docs} не найден!")
        return

    tokenizer = RegexTokenizer()
    normalizer = TextNormalizer(use_lemmatization=True, use_stemming=False)

    texts, processed, labels = load_corpus(tokenizer, normalizer)
    word_vectors, doc_vectors = load_or_train_models(texts, processed, tokenizer, normalizer, args.train)

    lsa = LSAModel(n_topics=100)
    lsa.fit(processed)
    print(f"Корпус ПР1: {len(texts)} документов | Word2Vec: {word_vectors.vocab_size} слов | "
          f"Doc2Vec: {doc_vectors.corpus_size} документов | LSA: {lsa.svd.n_components} тем")

    corpus_docs = [doc.split() for doc in processed]
    analyzer = DocumentAnalyzer(lsa, word_vectors, doc_vectors, corpus_docs, tokenizer, normalizer)

    if args.words:
        print_similar_words(analyzer, args.words)
    elif args.analogy:
        print_analogy(analyzer, *args.analogy)
    elif args.odd:
        print_odd_one_out(analyzer, args.odd)
    elif args.search:
        print_search(analyzer, texts, labels, args.search)
    elif args.compare:
        print_compare(analyzer, *args.compare)
    elif args.docs:
        print_documents(analyzer, args.docs, args.clusters)
    elif args.evaluate:
        evaluate(analyzer, labels)


if __name__ == "__main__":
    main()
