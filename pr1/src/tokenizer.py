import re
from typing import List


class RegexTokenizer:
    """
    Пользовательский токенизатор текстов на естественном языке.
    Выделяет слова (включая дефисные конструкции) и удаляет пунктуацию/мусор.
    """

    def __init__(self, lower: bool = True):
        self.lower = lower
        self.pattern = re.compile(r'[а-яА-ЯёЁa-zA-Z]+(?:-[а-яА-ЯёЁa-zA-Z]+)?')

    def tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        if self.lower:
            text = text.lower()
        return self.pattern.findall(text)
