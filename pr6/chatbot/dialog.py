from operator import itemgetter
from typing import Dict, List, Tuple

from chatbot.search import HybridSearch

CLARIFY = [
    "Не уверен, что правильно понял вопрос. Вас интересует запуск генерации, параметры сцены или разметка?",
    "Об этом я, к сожалению, ничего не знаю. Могу рассказать о запуске генератора, погоде и времени суток "
    "в сценах или о формате разметки.",
    "Не нашел ответа на этот вопрос. Попробуйте спросить иначе, например, как запустить генерацию "
    "или как устроена разметка дыма.",
]
CLARIFY_NEGATIVE = [
    "Жаль, что результат не устроил. Опишите, пожалуйста, что именно не так: рамки разметки, освещение "
    "кадров или вид дыма и пламени?",
    "Понимаю, что это неприятно. Уточните, пожалуйста, в чем проблема: генерация не запускается, "
    "работает медленно или не устраивают сами кадры?",
]
OPPOSITE = {("negative", "positive"), ("positive", "negative")}


class DialogEngine:
    """
    Гибридный диалоговый движок по лекции 12: ответ выбирается из кандидатов, которые
    дают несколько подходов, кроме шаблонного.

    1. Информационный поиск по корпусу пар «высказывание – ответ»: находятся темы,
       высказывания которых ближе всего к реплике пользователя.
    2. Порождающая модель seq2seq пишет свой ответ. Если он совпадает с ответом темы, оценка
       которой отстает от лучшей не больше чем на vote_margin, выбирается эта тема: когда поиск
       не уверен между несколькими темами, решает второй, независимый подход.
    3. Если уверенного ответа в корпусе нет, бот ищет подходящий фрагмент README генератора
       (роль поисковой системы из лекции). Здесь стоп-слова удаляются, как в ПР1: фрагменты длинные,
       и служебные слова давали ложные совпадения даже с вопросами не по теме.
    4. Если не помогло и это, бот честно говорит, что не понял, и задает уточняющий вопрос.

    Если реплика после нормализации дословно совпадает с высказыванием корпуса («Привет!»),
    отвечает соответствующая тема: короткое высказывание вместе с длинным ответом иначе может
    не набрать порога.

    Тональность реплики (сети ПР4 и ПР5) используется как фильтр: темы, высказывания которых
    уверенно противоположны реплике по тональности, получают штраф. Кроме того, на уверенно
    негативную реплику, которую бот не понял, он отвечает не общим уточнением, а просьбой
    описать проблему. Контекст: реплика,
    начинающаяся с «а» или «и» («А в высоком качестве?»), дополняется предыдущей репликой.
    """

    def __init__(self, normalize, dialogs: List[Dict], passages: List[Dict], word_vectors, generator, tone,
                 answer_threshold: float = 0.4, passage_threshold: float = 0.4, tone_penalty: float = 0.1,
                 vote_margin: float = 0.1):
        """
        Строит индексы поиска по корпусу диалогов и по README.

        :param normalize: DialogNormalizer (вызов дает все леммы, метод content только значимые)
        :param dialogs: корпус из load_dialogs
        :param passages: фрагменты README из load_passages
        :param word_vectors: дообученная модель Word2Vec
        :param generator: обученный ReplyGenerator или None
        :param tone: ToneAnalyzer или None
        :param answer_threshold: минимальная оценка темы, при которой бот отвечает из корпуса
        :param passage_threshold: минимальная оценка фрагмента README
        :param tone_penalty: штраф за противоположную тональность
        :param vote_margin: насколько тема seq2seq может отставать от лучшей темы поиска
        """
        self.normalize = normalize
        self.dialogs = dialogs
        self.passages = passages
        self.generator = generator
        self.tone = tone
        self.answer_threshold = answer_threshold
        self.passage_threshold = passage_threshold
        self.tone_penalty = tone_penalty
        self.vote_margin = vote_margin
        self.reply_topic = {d["reply"]: i for i, d in enumerate(dialogs)}
        self.exact = {tuple(normalize(st)): i for i, d in enumerate(dialogs) for st in d["statements"]}

        self.doc_entry, self.documents, texts = [], [], []
        for i, entry in enumerate(dialogs):
            reply = normalize(entry["reply"])
            for statement in entry["statements"]:
                self.doc_entry.append(i)
                self.documents.append(normalize(statement) + reply)
                texts.append(statement)
        self.doc_tone = tone.labels(texts) if tone else ["neutral"] * len(texts)
        self.dialog_search = HybridSearch(word_vectors)
        self.dialog_search.fit(self.documents)
        self.passage_search = HybridSearch(word_vectors)
        self.passage_search.fit([normalize.content(p["section"] + " " + p["text"]) for p in passages])
        self.previous = []
        self.clarify_count = 0

    def reset(self):
        """
        Забывает контекст разговора.
        """
        self.previous = []
        self.clarify_count = 0

    def rank_topics(self, lemmas: List[str], query_tone: str = "neutral",
                    use_index: bool = True) -> List[Tuple[int, float]]:
        """
        Ранжирует темы корпуса: оценка темы равна лучшей оценке ее высказываний с учетом
        штрафа за противоположную тональность.

        :param lemmas: леммы реплики
        :param query_tone: уверенная тональность реплики
        :param use_index: использовать индекс Voyager
        :return: список пар (номер темы, оценка) по убыванию оценки
        """
        best = {}
        for doc, score in self.dialog_search.search(lemmas, k=40, use_index=use_index):
            if (query_tone, self.doc_tone[doc]) in OPPOSITE:
                score -= self.tone_penalty
            entry = self.doc_entry[doc]
            best[entry] = max(best.get(entry, -1.0), score)
        return sorted(best.items(), key=itemgetter(1), reverse=True)

    def generated_topic(self, lemmas: List[str]) -> Tuple[str, int]:
        """
        Ответ порождающей модели и тема корпуса, если модель воспроизвела ответ этой темы.

        :param lemmas: леммы реплики
        :return: (текст ответа seq2seq, номер темы или -1)
        """
        if self.generator is None:
            return "", -1
        text, _ = self.generator.generate(lemmas)
        return text, self.reply_topic.get(text, -1)

    def choose(self, lemmas: List[str], query_tone: str = "neutral", use_index: bool = True) -> Dict:
        """
        Выбирает тему корпуса: лучшая тема поиска, но если ответ seq2seq совпадает с ответом
        темы, которая отстает от лучшей не больше чем на vote_margin, выбирается она.

        :param lemmas: леммы реплики
        :param query_tone: уверенная тональность реплики
        :param use_index: использовать индекс Voyager
        :return: словарь: entry (номер темы), score, source, ranked (три лучшие темы), generated (ответ seq2seq)
        """
        ranked = self.rank_topics(lemmas, query_tone, use_index)
        generated, generated_topic = self.generated_topic(lemmas)
        entry, score = ranked[0]
        source = "поиск по диалогам"
        scores = dict(ranked)
        if generated_topic in scores and scores[generated_topic] >= score - self.vote_margin:
            source = "поиск + seq2seq" if generated_topic == entry else "seq2seq среди близких тем поиска"
            entry, score = generated_topic, scores[generated_topic]
        return {"entry": entry, "score": score, "source": source, "ranked": ranked[:3], "generated": generated}

    def respond(self, text: str) -> Dict:
        """
        Формирует ответ на реплику пользователя.

        :param text: реплика
        :return: словарь: reply (ответ), source (каким подходом получен), topic, score, tone,
            а также lemmas, ranked и generated для подробного вывода
        """
        lemmas = self.normalize(text)
        if text.strip().lower().startswith(("а ", "и ")) and self.previous:
            lemmas = self.previous + lemmas
        query_tone = self.tone.labels([text])[0] if self.tone else "neutral"
        result = {"tone": query_tone, "topic": None, "score": 0.0, "lemmas": lemmas, "ranked": [], "generated": ""}
        if not lemmas:
            return dict(result, reply=self.clarify(query_tone), source="уточнение")
        self.previous = lemmas

        choice = self.choose(lemmas, query_tone)
        result.update(ranked=choice["ranked"], generated=choice["generated"])
        entry, score = choice["entry"], choice["score"]
        if tuple(lemmas) in self.exact:
            entry, score, choice["source"] = self.exact[tuple(lemmas)], 1.0, "дословно есть в корпусе"
        if score >= self.answer_threshold:
            return dict(result, reply=self.dialogs[entry]["reply"], source=choice["source"],
                        topic=self.dialogs[entry]["topic"], score=score)

        passage, passage_score = self.passage_search.search(self.normalize.content(text), k=1)[0]
        if passage_score >= self.passage_threshold:
            return dict(result, reply=self.passages[passage]["text"], source="поиск по README",
                        topic="README: " + self.passages[passage]["section"], score=passage_score)
        return dict(result, reply=self.clarify(query_tone), source="уточнение", score=max(score, passage_score))

    def clarify(self, query_tone: str = "neutral") -> str:
        """
        Уточняющий вопрос. Варианты чередуются, чтобы бот не повторял одну фразу (совет из лекции).

        :param query_tone: уверенная тональность реплики
        :return: текст уточняющего вопроса
        """
        variants = CLARIFY_NEGATIVE if query_tone == "negative" else CLARIFY
        reply = variants[self.clarify_count % len(variants)]
        self.clarify_count += 1
        return reply
