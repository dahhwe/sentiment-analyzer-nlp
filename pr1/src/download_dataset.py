import argparse
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "src" else CURRENT_DIR
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "dataset.csv"
CACHE_FILE = DATA_DIR / "cached_dataset.csv"

LABEL_MAPPING = {
    0: "neutral",
    1: "positive",
    2: "negative"
}


def download_and_prepare(samples_per_class: int = 2000):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1. Загрузка датасета MonoHime/ru_sentiment_dataset через Hugging Face ===")
    ds = load_dataset("MonoHime/ru_sentiment_dataset")

    train_data = ds["train"]
    print(f"Всего сырых документов в train: {len(train_data)}")

    print("=== 2. Конвертация меток и балансировка классов ===")
    df = pd.DataFrame({
        "text": train_data["text"],
        "sentiment": train_data["sentiment"]
    })

    df["label"] = df["sentiment"].map(LABEL_MAPPING)
    df = df.dropna(subset=["text", "label"])

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10]

    balanced_df = (
        df.groupby("label", as_index=False)
        .apply(lambda g: g.sample(n=min(len(g), samples_per_class), random_state=42))
        .reset_index(drop=True)
    )

    balanced_df = balanced_df[["text", "label"]].sample(frac=1.0, random_state=42).reset_index(drop=True)

    if CACHE_FILE.exists():
        os.remove(CACHE_FILE)

    balanced_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n[УСПЕХ] Датасет подготовлен и сохранен в: {OUTPUT_FILE}")
    print(f"Итоговый размер корпуса: {len(balanced_df)} документов")
    print(f"Распределение классов:\n{balanced_df['label'].value_counts().to_string()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Загрузчик датасета MonoHime/ru_sentiment_dataset")
    parser.add_argument("--samples", type=int, default=2000,
                        help="Количество примеров на каждый класс (по умолчанию 2000)")
    args = parser.parse_args()

    download_and_prepare(samples_per_class=args.samples)
