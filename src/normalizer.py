from typing import List, Dict

import nltk
import pymorphy3
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


class TextNormalizer:
    """
    Модуль нормализации словаря:
    1. Удаление стоп-слов
    2. Лемматизация (приведение к словарной форме)
    3. Стемминг (отсечение окончаний)
    4. Канонизация синонимов (схлопывание синонимичных рядов для снижения размерности словаря)
    """

    def __init__(self, use_lemmatization: bool = True, use_stemming: bool = False):
        self.use_lemmatization = use_lemmatization
        self.use_stemming = use_stemming

        try:
            self.stop_words = set(stopwords.words('russian'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('russian'))

        self.morph = pymorphy3.MorphAnalyzer()

        self.stemmer = SnowballStemmer('russian')

        self.synonym_map: Dict[str, str] = {
            'великолепный': 'хороший',
            'отличный': 'хороший',
            'превосходный': 'хороший',
            'прекрасный': 'хороший',
            'замечательный': 'хороший',
            'шикарный': 'хороший',
            'вкусный': 'хороший',
            'отвратительный': 'плохой',
            'ужасный': 'плохой',
            'скверный': 'плохой',
            'паршивый': 'плохой',
            'невкусный': 'плохой'
        }

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stop_words and len(token) > 2]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.morph.parse(token)[0].normal_form for token in tokens]

    def stem(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(token) for token in tokens]

    def replace_synonyms(self, tokens: List[str]) -> List[str]:
        return [self.synonym_map.get(token, token) for token in tokens]

    def normalize(self, tokens: List[str]) -> List[str]:
        """Полный конвейер нормализации."""
        filtered = self.remove_stopwords(tokens)

        if self.use_lemmatization:
            normalized = self.lemmatize(filtered)
        else:
            normalized = filtered

        canonicalized = self.replace_synonyms(normalized)

        if self.use_stemming:
            canonicalized = self.stem(canonicalized)

        return canonicalized
