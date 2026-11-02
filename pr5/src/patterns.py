import re
from datetime import date
from typing import Dict, List

MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня",
          "июля", "августа", "сентября", "октября", "ноября", "декабря"]

FACT_NAMES = {
    "money": ("сумма", "суммы"),
    "date": ("дата", "даты"),
    "time": ("время", "время"),
    "duration": ("срок", "сроки"),
    "percent": ("процент", "проценты"),
    "phone": ("телефон", "телефоны"),
}

PATTERNS = {
    "money": re.compile(r"\b\d+(?:[ .,]\d+)*\s?(?:(?:тыс|млн)\.?\s?)?(?:руб(?:лей|ля|ль)?\.?|р\.|₽|евро|долларов|доллара|\$)",
                        re.IGNORECASE),
    "date": re.compile(r"\b(?:[0123]?\d[./][01]?\d[./](?:\d\d)?\d\d|[0123]?\d\s(?:" + "|".join(MONTHS) +
                       r")(?:\s(?:19|20)\d\d)?)\b", re.IGNORECASE),
    "time": re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b"),
    "duration": re.compile(r"\b\d+(?:[.,]\d+)?\s(?:минут[уы]?|час(?:а|ов)?|дн(?:я|ей)|день|недел[июь]|"
                           r"месяц(?:а|ев)?|год(?:а)?|лет)\b", re.IGNORECASE),
    "percent": re.compile(r"\b\d+(?:[.,]\d+)?\s?%"),
    "phone": re.compile(r"(?:\+7|\b8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}\b"),
}


def is_valid_date(text: str) -> bool:
    """
    Проверяет, что найденная числовая дата существует (как в лекции: шаблон находит
    кандидатов, а модуль datetime отбрасывает невозможные даты вроде 35.13.2020).

    :param text: дата вида ДД.ММ.ГГГГ, ДД.ММ.ГГ или с месяцем словом
    :return: True, если дата допустимая
    """
    parts = re.split(r"[./]", text)
    if len(parts) != 3:
        return True
    day, month, year = (int(p) for p in parts)
    if year < 100:
        year += 2000 if year < 50 else 1900
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


def extract_facts(text: str) -> Dict[str, List[str]]:
    """
    Выделяет из исходного текста количественную информацию регулярными выражениями:
    суммы, даты, время, сроки, проценты и телефоны.

    Токенизатор ПР1 отбрасывает числа, поэтому шаблоны применяются к исходному тексту,
    а не к леммам.

    :param text: исходный текст
    :return: словарь {вид факта: список найденных фрагментов}, пустые виды не включаются
    """
    facts = {}
    for kind, pattern in PATTERNS.items():
        found = []
        for match in pattern.finditer(text):
            value = " ".join(match.group().split())
            if kind == "date" and not is_valid_date(value):
                continue
            if value not in found:
                found.append(value)
        if found:
            facts[kind] = found
    return facts

