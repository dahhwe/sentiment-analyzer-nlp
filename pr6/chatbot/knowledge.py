import re
from typing import Dict, List

import pandas as pd


class DialogNormalizer:
    """
    Нормализация реплик для поиска ответа на основе конвейера ПР1.

    В отличие от ПР1 стоп-слова не удаляются: в коротких репликах вроде «Кто ты?»
    или «Как дела?» они и есть весь смысл, а частые слова и так получают малый вес
    в TF-IDF. Кроме того, из словаря синонимов ПР1 убрана замена сленгового «огонь»
    на «хороший»: для отзывов о товарах это было оправдано, а в разговоре о пожарах
    «огонь» значит пламя.
    """

    def __init__(self, tokenizer, normalizer):
        """
        Создает нормализатор.

        :param tokenizer: RegexTokenizer из ПР1
        :param normalizer: TextNormalizer из ПР1 (используются лемматизация, отрицания и синонимы)
        """
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.normalizer.synonym_map.pop("огонь", None)

    def __call__(self, text: str) -> List[str]:
        """
        Переводит реплику в список лемм.

        :param text: исходный текст
        :return: леммы с привязанными отрицаниями и канонизированными синонимами
        """
        lemmas = self.normalizer.lemmatize(self.tokenizer.tokenize(text))
        return self.normalizer.replace_synonyms(self.normalizer.bind_negations(lemmas))

    def content(self, text: str) -> List[str]:
        """
        Полный конвейер ПР1 со стоп-словами: используется для поиска по длинным фрагментам README,
        где служебные слова только создают ложные совпадения.

        :param text: исходный текст
        :return: значимые леммы
        """
        return self.normalizer.normalize(self.tokenizer.tokenize(text))


def load_dialogs(path) -> List[Dict]:
    """
    Загружает корпус пар «высказывание – ответ». В одной строке файла записана тема,
    несколько вариантов высказывания пользователя через « | » и ответ.

    :param path: путь к dialogs.csv
    :return: список словарей с ключами topic, statements, reply
    """
    df = pd.read_csv(path)
    return [{"topic": row.topic, "statements": row.statements.split(" | "), "reply": row.reply}
            for row in df.itertuples()]


def clean_markdown(text: str) -> str:
    """
    Убирает из строки разметку Markdown: выделение, обратные кавычки и ячейки таблиц.

    :param text: строка Markdown
    :return: обычный текст
    """
    text = text.replace("**", "").replace("`", "").replace("\\|", "/")
    cells = [c.strip() for c in text.strip().strip("|").split("|")] if text.strip().startswith("|") else None
    if cells:
        text = ": ".join(c for c in cells if c)
    return " ".join(text.split())


def load_passages(path) -> List[Dict[str, str]]:
    """
    Делит README генератора на фрагменты для поиска: абзацы, пункты списков и строки таблиц.
    Строки блоков кода присоединяются к предыдущему фрагменту того же раздела (обычно это
    команда, о которой говорит абзац), к каждому фрагменту запоминается заголовок раздела.
    Короткие фрагменты и вводные строки перед списками («Основные параметры:») пропускаются.

    :param path: путь к README в формате Markdown
    :return: список словарей с ключами section и text
    """
    passages, section, current, in_code = [], "", [], False

    def flush():
        """
        Сохраняет накопленный фрагмент.
        """
        if current:
            passages.append({"section": section, "text": clean_markdown(" ".join(current))})
            current.clear()

    for line in open(path, encoding="utf-8").read().splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            if stripped and not current and passages and passages[-1]["section"] == section:
                passages[-1]["text"] += " " + " ".join(stripped.split())
            elif stripped:
                current.append(stripped)
            continue
        if stripped.startswith("#"):
            flush()
            section = stripped.lstrip("#").strip()
        elif not stripped or re.match(r"^\|[-\s|]+\|$", stripped):
            flush()
        elif stripped.startswith("|"):
            flush()
            current.append(stripped)
            flush()
        elif re.match(r"^(-|\d+\.)\s", stripped):
            flush()
            current.append(re.sub(r"^(-|\d+\.)\s", "", stripped))
        else:
            current.append(stripped)
    flush()
    return [p for p in passages if len(p["text"].split()) >= 4 and not p["text"].endswith(":")]
