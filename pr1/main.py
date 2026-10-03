import argparse
import os

import pandas as pd

from src.classifier import SentimentClassifierNB
from src.normalizer import TextNormalizer
from src.tokenizer import RegexTokenizer


def preprocess_corpus(texts, tokenizer, normalizer):
    processed = []
    total = len(texts)
    for i, text in enumerate(texts):
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"   Обработано {i + 1}/{total} документов...", end="\r")
        tokens = tokenizer.tokenize(text)
        norm_tokens = normalizer.normalize(tokens)
        processed.append(" ".join(norm_tokens))
    print()
    return processed


def load_or_build_cache(data_path: str, cache_path: str, tokenizer, normalizer):
    """Загружает предобработанные данные из кэша, либо строит кэш с нуля."""
    if os.path.exists(cache_path):
        cached_df = pd.read_csv(cache_path).dropna()
        return cached_df['processed_text'].tolist(), cached_df['label'].tolist()

    print("=== 1. Первичная предобработка датасета (токенизация + нормализация) ===")
    df = pd.read_csv(data_path).dropna(subset=['text', 'label'])
    processed_texts = preprocess_corpus(df['text'].tolist(), tokenizer, normalizer)

    cached_df = pd.DataFrame({
        'processed_text': processed_texts,
        'label': df['label'].tolist()
    })
    cached_df.to_csv(cache_path, index=False)
    print(f"Кэш нормализованного корпуса успешно сохранен в {cache_path}")
    return processed_texts, df['label'].tolist()


def main():
    parser = argparse.ArgumentParser(description="Анализатор тональности на реальном корпусе RuSentiment")
    parser.add_argument("--text", type=str, help="Пользовательский текст для анализа")
    parser.add_argument("--data", type=str, default="data/dataset.csv", help="Путь к исходному датасету")
    parser.add_argument("--cache", type=str, default="data/cached_dataset.csv", help="Путь к кэшу предобработки")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Файл {args.data} не найден! Запустите: python download_dataset.py")
        return

    tokenizer = RegexTokenizer()
    normalizer = TextNormalizer(use_lemmatization=True, use_stemming=False)

    processed_texts, labels = load_or_build_cache(args.data, args.cache, tokenizer, normalizer)

    classifier = SentimentClassifierNB()
    classifier.train(processed_texts, labels)

    def analyze_single_text(raw_text: str):
        tokens = tokenizer.tokenize(raw_text)
        normalized = normalizer.normalize(tokens)
        processed_input = " ".join(normalized)
        label, conf, reason = classifier.predict(processed_input)

        label_map = {
            "positive": "Позитивный 😊",
            "negative": "Негативный 😞",
            "neutral": "Нейтральный 😐"
        }

        print("\n--- Результат анализа ---")
        print(f"Исходный текст:      {raw_text}")
        print(f"Токены:              {tokens}")
        print(f"После нормализации:  {normalized}")
        print(f"Тональность:         {label_map.get(label, label)}")
        print(f"Уверенность:         {conf * 100:.1f}%")
        print(f"Пояснение:           {reason}")

    if args.text:
        analyze_single_text(args.text)
    else:
        print("\n=== Оценка модели на корпусе ===")
        print(classifier.evaluate(processed_texts[:500], labels[:500]))
        print("\n=== Интерактивный режим (введите 'exit' для выхода) ===")
        while True:
            try:
                user_input = input("\nВаш текст > ").strip()
                if user_input.lower() in ('exit', 'quit', 'выход', 'q'):
                    break
                if not user_input:
                    continue
                analyze_single_text(user_input)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
