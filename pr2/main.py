import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Токенизатор, нормализатор и классификатор берутся из Практической работы №1
PR1_DIR = Path(__file__).resolve().parent.parent / "pr1"
sys.path.append(str(PR1_DIR / "src"))

from classifier import SentimentClassifierNB
from normalizer import TextNormalizer
from tokenizer import RegexTokenizer

from src.lsa import LSAModel
from src.steering import SentimentSteering

PR1_DATA = PR1_DIR / "data" / "dataset.csv"
PR1_CACHE = PR1_DIR / "data" / "cached_dataset.csv"

LABEL_MAP = {
    "positive": "Позитивный 😊",
    "negative": "Негативный 😞",
    "neutral": "Нейтральный 😐"
}


def preprocess_corpus(texts, tokenizer, normalizer):
    processed = []
    total = len(texts)
    for i, text in enumerate(texts):
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"   Обработано {i + 1}/{total} документов...", end="\r")
        tokens = tokenizer.tokenize(text)
        processed.append(" ".join(normalizer.normalize(tokens)))
    print()
    return processed


def load_corpus(data_path: str, tokenizer, normalizer):
    """
    Загружает корпус: CSV (столбец text и необязательный label) или TXT (один документ на строку).
    Для корпуса из ПР1 используется уже готовый кэш нормализации из ПР1.
    Возвращает: (исходные тексты, нормализованные тексты, метки или None)
    """
    if data_path.endswith(".txt"):
        with open(data_path, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        labels = None
    else:
        df = pd.read_csv(data_path).dropna(subset=["text"])
        texts = df["text"].astype(str).tolist()
        labels = df["label"].astype(str).tolist() if "label" in df.columns else None

    is_pr1_corpus = Path(data_path).resolve() == PR1_DATA
    processed = None
    if is_pr1_corpus and PR1_CACHE.exists():
        cached = pd.read_csv(PR1_CACHE, keep_default_na=False)
        if len(cached) == len(texts):
            processed = cached["processed_text"].astype(str).tolist()
            print(f"Нормализованный корпус загружен из кэша ПР1: {PR1_CACHE}")

    if processed is None:
        print("=== Предобработка корпуса (токенизация + нормализация из ПР1) ===")
        processed = preprocess_corpus(texts, tokenizer, normalizer)
        if is_pr1_corpus:
            pd.DataFrame({"processed_text": processed, "label": labels}).to_csv(PR1_CACHE, index=False)

    keep = [i for i, text in enumerate(processed) if text.strip()]
    texts = [texts[i] for i in keep]
    processed = [processed[i] for i in keep]
    if labels is not None:
        labels = [labels[i] for i in keep]
    return texts, processed, labels


def short(text: str, length: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[:length] + "..."


def evaluate(lsa, processed, labels):
    """Сравнение стиринга (LDA на векторах тем) с наивным Байесом из ПР1 на одной тестовой выборке."""
    idx_train, idx_test = train_test_split(np.arange(len(labels)), test_size=0.3, random_state=42)
    y = np.array(labels)

    steering = SentimentSteering()
    steering.train(lsa.doc_vectors[idx_train], y[idx_train])
    print(f"--- Стиринг: LDA на векторах тем LSA ({lsa.svd.n_components} тем) ---")
    print(steering.evaluate(lsa.doc_vectors[idx_test], y[idx_test]))

    nb = SentimentClassifierNB()
    nb.train([processed[i] for i in idx_train], list(y[idx_train]))
    nb_result = nb.evaluate([processed[i] for i in idx_test], list(y[idx_test]))
    print(f"--- Для сравнения: наивный Байес из ПР1 ---\n{nb_result.splitlines()[0]}")


def main():
    parser = argparse.ArgumentParser(description="Семантический анализатор (LSA) на корпусе из ПР1")
    parser.add_argument("--text", type=str, help="Текст для семантического анализа")
    parser.add_argument("--search", type=str, help="Поисковый запрос (поиск документов по смыслу)")
    parser.add_argument("--compare", nargs=2, metavar=("TEXT1", "TEXT2"), help="Сравнить два текста")
    parser.add_argument("--topics", action="store_true", help="Показать темы LSA")
    parser.add_argument("--evaluate", action="store_true", help="Оценить точность стиринга")
    parser.add_argument("--data", type=str, default=str(PR1_DATA),
                        help="Корпус: CSV (text[,label]) или TXT (один документ на строку)")
    parser.add_argument("--n-topics", type=int, default=100, help="Число тем LSA (по умолчанию 100)")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Файл {args.data} не найден!")
        return

    tokenizer = RegexTokenizer()
    normalizer = TextNormalizer(use_lemmatization=True, use_stemming=False)

    texts, processed, labels = load_corpus(args.data, tokenizer, normalizer)

    lsa = LSAModel(n_topics=args.n_topics)
    lsa.fit(processed)
    print(f"Корпус: {len(texts)} документов, словарь TF-IDF: {len(lsa.terms)} слов, "
          f"тем LSA: {lsa.svd.n_components}")

    steering = None
    if labels is not None and len(set(labels)) > 1:
        steering = SentimentSteering()
        steering.train(lsa.doc_vectors, labels)

    def normalize_text(raw_text: str):
        return normalizer.normalize(tokenizer.tokenize(raw_text))

    def print_documents(found):
        for rank, (idx, score) in enumerate(found, 1):
            label = f" [{labels[idx]}]" if labels is not None else ""
            print(f"   {rank}. ({score:.2f}){label} {short(texts[idx])}")

    def show_topics(n: int = 10):
        print(f"\n--- Темы LSA (первые {n} из {lsa.svd.n_components}) ---")
        for topic in range(min(n, lsa.svd.n_components)):
            print(f"Тема {topic}: {', '.join(lsa.topic_words(topic))}")

    def search(query: str):
        normalized = normalize_text(query)
        print("\n--- Семантический поиск ---")
        print(f"Запрос:              {query}")
        print(f"После нормализации:  {normalized}")
        found = lsa.search(" ".join(normalized))
        if not found:
            print("Слова запроса не найдены в словаре корпуса")
        print_documents(found)

    def compare(text_a: str, text_b: str):
        norm_a, norm_b = normalize_text(text_a), normalize_text(text_b)
        cos_tfidf, cos_lsa = lsa.similarity(" ".join(norm_a), " ".join(norm_b))
        print("\n--- Сравнение текстов ---")
        print(f"Текст 1:  {text_a}  ->  {norm_a}")
        print(f"Текст 2:  {text_b}  ->  {norm_b}")
        print(f"Косинусное сходство TF-IDF:  {cos_tfidf:.3f}")
        print(f"Косинусное сходство LSA:     {cos_lsa:.3f}")

    def analyze_single_text(raw_text: str):
        normalized = normalize_text(raw_text)
        processed_input = " ".join(normalized)

        print("\n--- Результат анализа ---")
        print(f"Исходный текст:      {raw_text}")
        print(f"После нормализации:  {normalized}")
        if not lsa.known_words(processed_input):
            print("Слова текста не найдены в словаре корпуса")
            return

        print("Основные темы LSA:")
        for topic, weight in lsa.main_topics(processed_input):
            print(f"   тема {topic} ({weight:.2f}): {', '.join(lsa.topic_words(topic, 6))}")

        if steering is not None:
            label, conf = steering.predict(lsa.transform([processed_input])[0])
            print(f"Тональность (LDA):   {LABEL_MAP.get(label, label)}")
            print(f"Уверенность:         {conf * 100:.1f}%")

        print("Похожие документы корпуса:")
        print_documents(lsa.search(processed_input, top_n=3))

    if args.topics:
        show_topics()
    elif args.evaluate:
        if labels is None:
            print("Для оценки нужен корпус с метками (столбец label)")
        else:
            evaluate(lsa, processed, labels)
    elif args.search:
        search(args.search)
    elif args.compare:
        compare(*args.compare)
    elif args.text:
        analyze_single_text(args.text)
    else:
        print("\n=== Интерактивный режим (введите 'exit' для выхода) ===")
        print("Текст — семантический анализ, '? запрос' — поиск по корпусу")
        while True:
            try:
                user_input = input("\nВаш текст > ").strip()
                if user_input.lower() in ('exit', 'quit', 'выход', 'q'):
                    break
                if not user_input:
                    continue
                if user_input.startswith("?"):
                    search(user_input[1:].strip())
                else:
                    analyze_single_text(user_input)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
