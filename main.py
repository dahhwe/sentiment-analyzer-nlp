import argparse
import os

import pandas as pd

from src.classifier import SentimentClassifierNB
from src.normalizer import TextNormalizer
from src.tokenizer import RegexTokenizer


def preprocess_corpus(texts, tokenizer, normalizer):
    """Предобработка корпуса: токенизация + нормализация."""
    processed = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        norm_tokens = normalizer.normalize(tokens)
        processed.append(" ".join(norm_tokens))
    return processed


def main():
    parser = argparse.ArgumentParser(description="Анализатор тональности текста на русском языке (Naive Bayes)")
    parser.add_argument("--text", type=str, help="Пользовательский текст для анализа тональности")
    parser.add_argument("--data", type=str, default="data/dataset.csv", help="Путь к обучающему датасету")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Ошибка: Файл данных {args.data} не найден!")
        return

    df = pd.read_csv(args.data)

    tokenizer = RegexTokenizer()
    normalizer = TextNormalizer(use_lemmatization=True, use_stemming=False)

    print("=== 1. Предобработка корпуса документов ===")
    processed_texts = preprocess_corpus(df['text'].tolist(), tokenizer, normalizer)

    classifier = SentimentClassifierNB()
    classifier.train(processed_texts, df['label'].tolist())
    print("Модель Наивного Байеса успешно обучена на нормализованном корпусе.")
    print(classifier.evaluate(processed_texts, df['label'].tolist()))

    def analyze_single_text(raw_text: str):
        tokens = tokenizer.tokenize(raw_text)
        normalized = normalizer.normalize(tokens)
        processed_input = " ".join(normalized)
        label, conf = classifier.predict(processed_input)

        label_ru = "Позитивный 😊" if label == "positive" else "Негативный 😞"
        print("\n--- Результат анализа ---")
        print(f"Исходный текст:      {raw_text}")
        print(f"Токены:              {tokens}")
        print(f"После нормализации:  {normalized}")
        print(f"Тональность:         {label_ru}")
        print(f"Уверенность:         {conf * 100:.1f}%")

    if args.text:
        analyze_single_text(args.text)
    else:
        print("\n=== Интерактивный режим ===")
        print("Введите предложение на русском языке для анализа (или 'exit' для выхода):")
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
        print("\nЗавершение работы.")


if __name__ == "__main__":
    main()
