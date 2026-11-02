import random
from collections import Counter
from typing import Dict, List

import pymorphy3

from src.patterns import FACT_NAMES

TONE = {"positive": "позитивный", "negative": "негативный", "neutral": "нейтральный"}
TONE_SHORT = {"positive": "поз.", "negative": "нег.", "neutral": "нейтр."}

OPENINGS = {
    "positive": ["Автор доволен тем, о чем пишет.", "Текст выражает положительное отношение автора.",
                 "В тексте преобладают положительные оценки."],
    "negative": ["Автор недоволен тем, о чем пишет.", "Текст выражает отрицательное отношение автора.",
                 "В тексте преобладают отрицательные оценки."],
    "neutral": ["Текст носит информационный характер.", "Явной оценки в тексте нет.",
                "Автор скорее сообщает факты, чем выражает отношение."],
}
CONFIDENCE = [(0.8, "уверенно "), (0.6, ""), (0.0, "с небольшим перевесом ")]
AGREE_ALL = ["Все {n} анализаторов дают тот же ответ.", "Остальные анализаторы с этой оценкой согласны."]
AGREE_MOST = ["С этой оценкой согласны {k} из {n}, иначе считают {others}.",
              "Эту оценку подтверждают {k} из {n}, не согласны {others}."]
AGREE_FEW = ["Большинство других анализаторов с LSTM не согласны: {others}."]
KEYWORDS = ["Сильнее всего на оценку повлияли слова {words}.", "Ключевые для оценки слова: {words}."]
FACTS = ["Найденные в тексте факты: {facts}.", "Из текста выделены факты: {facts}."]
SIMILAR = ["Ближайший по вектору идеи seq2seq документ корпуса ({tone} тональность, сходство {score:.2f}): «{text}».",
           "По вектору идеи seq2seq ближе всего документ корпуса с {tone} тональностью (сходство {score:.2f}): «{text}»."]

GREETINGS = ["Здравствуйте!", "Добрый день!"]
REPLY_MAIN = {
    "positive": ["Спасибо за отзыв и высокую оценку!", "Благодарим вас за теплые слова!"],
    "negative": ["Приносим извинения за доставленные неудобства.", "Нам очень жаль, что у вас остались плохие впечатления."],
    "neutral": ["Спасибо за подробную информацию.", "Благодарим за отзыв, он будет полезен другим читателям."],
}
REPLY_FACTS = {
    "duration": "Ожидание {value} действительно слишком долгое, мы разберемся, почему так получилось.",
    "money": "Если сумма {value} кажется вам завышенной, напишите нам, и мы проверим расчет.",
    "date": "Мы уточним, что произошло {value}.",
    "phone": "Мы свяжемся с вами по номеру {value}.",
}
REPLY_END = {
    "positive": ["Будем рады видеть вас снова!", "Ждем вас снова!"],
    "negative": ["Мы передадим ваш отзыв руководству и сделаем все, чтобы это не повторилось.",
                 "Ваш отзыв поможет нам стать лучше."],
    "neutral": ["Если появятся вопросы, мы всегда на связи.", "Будем рады новым отзывам."],
}


class TemplateGenerator:
    """
    Генератор текста на основе шаблонов.

    Каждое предложение берется из заранее написанных шаблонов, а их пропуски заполняются
    результатами семантического анализа: тональностью, ключевыми словами, фактами,
    похожими документами. Вариант шаблона выбирается случайно (с фиксированным зерном),
    а согласование слов с числами и падежами выполняет pymorphy3 из ПР1.
    """

    def __init__(self, analyzer_names: Dict[str, str], seed: int = 0):
        """
        Создает генератор.

        :param analyzer_names: словарь {код анализатора: название для текста}
        :param seed: зерно генератора случайных чисел для выбора вариантов шаблонов
        """
        self.analyzer_names = analyzer_names
        self.rng = random.Random(seed)
        self.morph = pymorphy3.MorphAnalyzer()

    def choose(self, variants: List[str]) -> str:
        """
        Выбирает один из вариантов шаблона.

        :param variants: список шаблонов
        :return: выбранный шаблон
        """
        return self.rng.choice(variants)

    def agree(self, number: int, word: str) -> str:
        """
        Согласует слово с числом: 1 документ, 3 документа, 5 документов.

        :param number: число
        :param word: слово в начальной форме
        :return: строка «число слово»
        """
        return f"{number} {self.morph.parse(word)[0].make_agree_with_number(number).word}"

    def agree_case(self, number: int, word: str, case: str) -> str:
        """
        Согласует существительное с числом в заданном падеже: «для 1 документа», «для 5 документов».

        :param number: число
        :param word: существительное в начальной форме
        :param case: падеж в обозначениях pymorphy3, например "gent"
        :return: строка «число слово»
        """
        singular = number % 10 == 1 and number % 100 != 11
        return f"{number} {self.inflect(word, 'sing' if singular else 'plur', case)}"

    def inflect(self, word: str, *grammemes: str) -> str:
        """
        Ставит слово в нужную форму, например прилагательное в творительный падеж женского рода.

        :param word: слово в начальной форме
        :param grammemes: граммемы pymorphy3, например "femn", "ablt"
        :return: слово в нужной форме (или исходное, если форму построить нельзя)
        """
        form = self.morph.parse(word)[0].inflect(set(grammemes))
        return form.word if form else word

    @staticmethod
    def join(items: List[str]) -> str:
        """
        Соединяет элементы списка через запятую и «и» перед последним.

        :param items: список строк
        :return: строка вида «a, b и c»
        """
        if len(items) <= 1:
            return "".join(items)
        return ", ".join(items[:-1]) + " и " + items[-1]

    @staticmethod
    def quote(text: str, length: int = 90) -> str:
        """
        Сокращает цитату до заданной длины и убирает точку в конце.

        :param text: исходный текст
        :param length: максимальная длина
        :return: текст для вставки в кавычки
        """
        text = " ".join(text.split()).rstrip(".")
        return text if len(text) <= length else text[:length].rsplit(" ", 1)[0] + "..."

    def facts_phrase(self, facts: Dict[str, List[str]]) -> str:
        """
        Описывает найденные факты: «сумма 2 500 рублей и дата 15 мая».

        :param facts: словарь {вид факта: значения}
        :return: фраза с перечислением фактов
        """
        parts = []
        for kind, values in facts.items():
            singular, plural = FACT_NAMES[kind]
            parts.append(f"{plural if len(values) > 1 else singular} {', '.join(values)}")
        return self.join(parts)

    def conclusion(self, analysis: Dict) -> str:
        """
        Генерирует заключение о документе по результатам анализа.

        :param analysis: результат анализа документа (словарь из SemanticAnalyzers.analyze в main.py)
        :return: текст заключения
        """
        label, probability = analysis["label"], analysis["probability"]
        confidence = next(word for bound, word in CONFIDENCE if probability >= bound)
        tone_plural = self.inflect(TONE[label], "plur", "datv")
        sentences = [self.choose(OPENINGS[label]),
                     f"Сеть LSTM {confidence}относит его к {tone_plural} текстам (вероятность {probability:.0%})."]

        votes = analysis["votes"]
        others = {name: vote for name, vote in votes.items() if vote != label}
        k, n = len(votes) - len(others), len(votes)
        others_text = self.join([f"{self.analyzer_names[name]} ({TONE_SHORT[vote]})" for name, vote in others.items()])
        if not others:
            sentences.append(self.choose(AGREE_ALL).format(n=n))
        elif k * 2 > n:
            sentences.append(self.choose(AGREE_MOST).format(k=k, n=self.agree(n, "анализатор"), others=others_text))
        else:
            sentences.append(self.choose(AGREE_FEW).format(others=others_text))

        if analysis["keywords"]:
            words = self.join([f"«{w.replace('_', ' ')}»" for w in analysis["keywords"]])
            sentences.append(self.choose(KEYWORDS).format(words=words))

        if analysis["facts"]:
            sentences.append(self.choose(FACTS).format(facts=self.facts_phrase(analysis["facts"])))

        similar = analysis.get("similar")
        if similar:
            text, tone, score = similar
            template = self.choose(SIMILAR)
            tone_word = self.inflect(TONE[tone], "femn", "nomn" if template == SIMILAR[0] else "ablt")
            sentences.append(template.format(tone=tone_word, score=score, text=self.quote(text)))
        return " ".join(sentences)

    def reply(self, analysis: Dict) -> str:
        """
        Генерирует ответ автору отзыва: благодарность или извинение в зависимости от тональности,
        реакцию на найденные в тексте факты и завершающую фразу.

        :param analysis: результат анализа документа
        :return: текст ответа
        """
        label, facts = analysis["label"], analysis["facts"]
        sentences = [self.choose(GREETINGS), self.choose(REPLY_MAIN[label])]
        if label == "negative":
            for kind, template in REPLY_FACTS.items():
                if kind in facts:
                    sentences.append(template.format(value=facts[kind][0]))
        elif "phone" in facts:
            sentences.append(REPLY_FACTS["phone"].format(value=facts["phone"][0]))
        sentences.append(self.choose(REPLY_END[label]))
        return " ".join(sentences)

    def corpus_report(self, analyses: List[Dict], similar_pairs: List) -> str:
        """
        Генерирует сводку по корпусу документов пользователя.

        :param analyses: результаты анализа всех документов корпуса
        :param similar_pairs: самые похожие пары документов по векторам seq2seq [(i, j, сходство)]
        :return: текст сводки
        """
        n = len(analyses)
        counts = Counter(a["label"] for a in analyses)
        distribution = self.join([
            f"{counts[label]} {self.morph.parse(TONE[label])[0].make_agree_with_number(counts[label]).word}"
            for label in ("negative", "neutral", "positive") if counts[label]
        ])
        sentences = [f"Корпус содержит {self.agree(n, 'документ')}. По оценке сети LSTM среди них {distribution}."]

        unanimous = [i for i, a in enumerate(analyses) if all(v == a["label"] for v in a["votes"].values())]
        disputed = [str(i + 1) for i in range(n) if i not in unanimous]
        text = f"Все анализаторы единодушны для {self.agree_case(len(unanimous), 'документ', 'gent')} из {n}"
        sentences.append(text + (f", расходятся оценки документов № {', '.join(disputed)}." if disputed else "."))

        for label, word in (("negative", "негативный"), ("positive", "позитивный")):
            scored = [(a["probabilities"][label], i) for i, a in enumerate(analyses) if a["label"] == label]
            if scored:
                _, i = max(scored)
                sentences.append(f"Самый {word} документ: № {i + 1} «{self.quote(analyses[i]['text'], 60)}».")

        for label in ("negative", "positive"):
            words = Counter(w for a in analyses if a["label"] == label for w in a["keywords"])
            if words:
                top = self.join([f"«{w.replace('_', ' ')}»" for w, _ in words.most_common(3)])
                tone = self.inflect(TONE[label], "plur", "loct")
                sentences.append(f"В {tone} документах на оценку сильнее всего влияют слова {top}.")

        facts = Counter(kind for a in analyses for kind, values in a["facts"].items() for _ in values)
        if facts:
            found = ", ".join(f"{FACT_NAMES[kind][count > 1]} ({count})" for kind, count in facts.items())
            sentences.append(f"Из документов выделены факты: {found}.")
        if similar_pairs:
            pairs = self.join([f"№ {i + 1} и № {j + 1} ({score:.2f})" for i, j, score in similar_pairs])
            sentences.append(f"Самые похожие по смыслу документы (вектор идеи seq2seq): {pairs}.")
        return " ".join(sentences)
