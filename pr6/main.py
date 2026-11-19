import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize as l2_normalize
from voyager import Index, Space

PR6_DIR = Path(__file__).resolve().parent
ROOT_DIR = PR6_DIR.parent
for folder in ("pr1/src", "pr2/src", "pr3/src", "pr4/src", "pr5"):
    sys.path.append(str(ROOT_DIR / folder))

from normalizer import TextNormalizer
from tokenizer import RegexTokenizer
from lsa import LSAModel
from word_vectors import WordVectors
from networks import CNNClassifier, EMBEDDING_DIMS, MAXLEN
from sequences import pad_trunc, tokenize_and_vectorize

from src.generator import TemplateGenerator
from src.lstm_classifier import LSTMClassifier
from src.patterns import extract_facts

from chatbot.dialog import DialogEngine
from chatbot.generator import ReplyGenerator
from chatbot.knowledge import DialogNormalizer, load_dialogs, load_passages
from chatbot.search import HybridSearch, fine_tune_word_vectors
from chatbot.tone import CLASSES, ToneAnalyzer

DATA_DIR = PR6_DIR / "data"
MODELS_DIR = PR6_DIR / "models"
PR3_WORD2VEC = ROOT_DIR / "pr3" / "models" / "word2vec.model"
PR4_CNN = ROOT_DIR / "pr4" / "models" / "cnn"
PR5_LSTM = ROOT_DIR / "pr5" / "models" / "lstm"
DOMAIN_WORD2VEC = MODELS_DIR / "word2vec_domain.model"
GENERATOR_PATH = MODELS_DIR / "generator"


class Assistant:
    """
    Чат-бот генератора synthfire в сборе: модели прошлых работ, дообученный Word2Vec,
    порождающая модель seq2seq и гибридный диалоговый движок.
    """

    def __init__(self, readme_path: Path = DATA_DIR / "synthfire_readme.md", retrain: bool = False):
        """
        Загружает модели ПР3–ПР5 и данные, обучает недостающие модели ПР6.

        :param readme_path: README генератора, по которому бот ищет ответы
        :param retrain: обучить Word2Vec и seq2seq заново, даже если они сохранены
        """
        for path, command in ((PR3_WORD2VEC, "pr3"), (PR4_CNN.with_suffix(".json"), "pr4"),
                              (PR5_LSTM.with_suffix(".json"), "pr5")):
            if not path.exists():
                raise FileNotFoundError(f"Не найдена модель {path.relative_to(ROOT_DIR)}. "
                                        f"Сначала выполните: python ../{command}/main.py --train")
        self.pr3_vectors = WordVectors()
        self.pr3_vectors.load(str(PR3_WORD2VEC))
        self.normalize = DialogNormalizer(RegexTokenizer(), TextNormalizer())
        self.dialogs = load_dialogs(DATA_DIR / "dialogs.csv")
        self.passages = load_passages(readme_path)
        self.statements = [self.normalize(s) for entry in self.dialogs for s in entry["statements"]]
        self.replies = [entry["reply"] for entry in self.dialogs for _ in entry["statements"]]
        MODELS_DIR.mkdir(exist_ok=True)

        self.domain_vectors = self.prepare_word_vectors(retrain)
        self.generator = self.prepare_generator(retrain)

        cnn = CNNClassifier(CLASSES)
        cnn.load(PR4_CNN)
        self.lstm = LSTMClassifier(CLASSES)
        self.lstm.load(PR5_LSTM)
        review_normalizer = TextNormalizer()
        tokenizer = RegexTokenizer()
        self.normalize_review = ReviewNormalizer(tokenizer, review_normalizer)
        self.tone = ToneAnalyzer(self.normalize_review, self.pr3_vectors, [cnn, self.lstm], pad_trunc,
                                 tokenize_and_vectorize, MAXLEN, EMBEDDING_DIMS)
        self.engine = DialogEngine(self.normalize, self.dialogs, self.passages, self.domain_vectors,
                                   self.generator, self.tone)

    def prepare_word_vectors(self, retrain: bool) -> WordVectors:
        """
        Загружает Word2Vec, дообученный на текстах предметной области, или дообучает модель ПР3.

        :param retrain: дообучить заново
        :return: модель WordVectors
        """
        if not retrain and DOMAIN_WORD2VEC.exists():
            vectors = WordVectors()
            vectors.load(str(DOMAIN_WORD2VEC))
            return vectors
        print("\n=== Дообучение Word2Vec из ПР3 на текстах предметной области ===")
        sentences = [self.normalize(p["section"] + " " + p["text"]) for p in self.passages]
        sentences += self.statements + [self.normalize(entry["reply"]) for entry in self.dialogs]
        known = {w for s in sentences for w in s if self.pr3_vectors.has_word(w)}
        total = {w for s in sentences for w in s}
        start = time.time()
        vectors = fine_tune_word_vectors(self.pr3_vectors, sentences)
        vectors.save(str(DOMAIN_WORD2VEC))
        print(f"Лемм в текстах: {len(total)}, из них нет в словаре ПР3: {len(total - known)} "
              f"({len(total - known) / len(total):.0%})")
        print(f"Словарь: {self.pr3_vectors.vocab_size} -> {vectors.vocab_size} слов, "
              f"время {time.time() - start:.1f} с, модель сохранена в models/word2vec_domain.model")
        return vectors

    def prepare_generator(self, retrain: bool) -> ReplyGenerator:
        """
        Загружает порождающую модель seq2seq или обучает ее на парах «реплика – ответ».

        :param retrain: обучить заново
        :return: модель ReplyGenerator
        """
        generator = ReplyGenerator(self.domain_vectors)
        if not retrain and generator.exists(GENERATOR_PATH):
            generator.load(GENERATOR_PATH)
            return generator
        print("\n=== Обучение порождающей модели seq2seq ===")
        start = time.time()
        history = generator.train(self.statements, self.replies)
        generator.model.summary()
        generator.save(GENERATOR_PATH)
        print(f"Пар «реплика – ответ»: {len(self.statements)}, словарь кодировщика: {len(generator.source_vocab)}, "
              f"декодировщика: {len(generator.target_vocab)}")
        print(f"Эпох: {len(history.history['loss'])}, точность на обучающих данных: "
              f"{history.history['accuracy'][-1]:.3f}, время обучения: {time.time() - start:.0f} с")
        print("Модель сохранена в models/generator.json и models/generator.weights.h5")
        return generator

    def template_reply(self, template: TemplateGenerator, text: str):
        """
        Ответ генератора на шаблонах из ПР5: тональность по сети LSTM и факты из регулярных выражений.

        :param template: TemplateGenerator из ПР5
        :param text: реплика пользователя
        :return: (ответ, тональность по LSTM)
        """
        sequences = tokenize_and_vectorize([self.normalize_review(text)], self.pr3_vectors)
        probs = self.lstm.predict(pad_trunc(sequences, MAXLEN, EMBEDDING_DIMS))[0]
        label = CLASSES[int(probs.argmax())]
        return template.reply({"label": label, "facts": extract_facts(text)}), label


class ReviewNormalizer:
    """
    Нормализация ПР1 без изменений (со стоп-словами и всеми синонимами): на таких леммах
    обучены сети тональности ПР4 и ПР5.
    """

    def __init__(self, tokenizer, normalizer):
        """
        Создает нормализатор.

        :param tokenizer: RegexTokenizer из ПР1
        :param normalizer: TextNormalizer из ПР1
        """
        self.tokenizer = tokenizer
        self.normalizer = normalizer

    def __call__(self, text: str):
        """
        Переводит текст в леммы так же, как в ПР1.

        :param text: исходный текст
        :return: леммы без стоп-слов
        """
        return self.normalizer.normalize(self.tokenizer.tokenize(text))


def read_messages(path):
    """
    Читает реплики пользователя из файла, по одной на строку.

    :param path: путь к TXT-файлу
    :return: список реплик
    """
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def print_wrapped(title, text, width=110, indent=""):
    """
    Выводит текст с переносом строк по словам.

    :param title: начало первой строки
    :param text: текст
    :param width: ширина строки
    :param indent: отступ следующих строк
    """
    line = title
    for word in text.split():
        if len(line) + len(word) + 1 > width and line.strip():
            print(line)
            line = indent + word
        else:
            line = f"{line} {word}" if line.strip() else line + word
    print(line)


def describe(result):
    """
    Краткое описание того, как получен ответ.

    :param result: результат DialogEngine.respond
    :return: строка вида «[поиск + seq2seq | тема run | оценка 0.62 | тональность neutral]»
    """
    parts = [result["source"]]
    if result["topic"]:
        parts.append(f"тема {result['topic']}")
    parts += [f"оценка {result['score']:.2f}", f"тональность {result['tone']}"]
    return "[" + " | ".join(parts) + "]"


def print_answer(assistant, text):
    """
    Отвечает на одну реплику и показывает, как работал каждый подход.

    :param assistant: Assistant
    :param text: реплика пользователя
    """
    probs = assistant.tone.probabilities([text])[0]
    result = assistant.engine.respond(text)
    print("\n--- Ответ на реплику ---")
    print(f"Реплика:              {text}")
    print(f"Леммы:                {result['lemmas']}")
    print(f"Тональность CNN+LSTM: {result['tone']} (" + ", ".join(f"{c} {p:.2f}" for c, p in zip(CLASSES, probs)) + ")")
    topics = ", ".join(f"{assistant.dialogs[i]['topic']} {s:.2f}" for i, s in result["ranked"])
    print(f"Лучшие темы поиска:   {topics}")
    print_wrapped("Ответ seq2seq:       ", result["generated"], indent=" " * 22)
    print(f"\nОтвет бота {describe(result)}:")
    print_wrapped("", result["reply"])


def print_dialog(assistant, path):
    """
    Проводит диалог по репликам пользователя из файла (контекст сохраняется между репликами).

    :param assistant: Assistant
    :param path: TXT-файл с репликами
    """
    print(f"\n--- Диалог по репликам из {path} ---")
    for text in read_messages(path):
        result = assistant.engine.respond(text)
        print(f"\nПользователь: {text}")
        print_wrapped("Бот:", result["reply"], indent="     ")
        print(f"     {describe(result)}")


def chat(assistant):
    """
    Интерактивный режим: диалог с ботом в консоли.

    :param assistant: Assistant
    """
    print("\nБот-помощник генератора synthfire. Задайте вопрос, «сброс» забывает контекст, «выход» завершает работу.")
    while True:
        try:
            text = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("выход", "exit", "quit"):
            break
        if text.lower() == "сброс":
            assistant.engine.reset()
            print("Бот: Контекст разговора сброшен.")
            continue
        if text:
            result = assistant.engine.respond(text)
            print_wrapped("Бот:", result["reply"], indent="     ")
            print(f"     {describe(result)}")


def print_comparison(assistant, path, seed):
    """
    Сравнивает генератор на шаблонах из ПР5 и диалоговую систему ПР6 на репликах пользователя.

    :param assistant: Assistant
    :param path: TXT-файл с репликами
    :param seed: зерно для выбора вариантов шаблонов
    """
    template = TemplateGenerator({"lstm": "LSTM"}, seed=seed)
    messages = read_messages(path)
    print(f"\n--- Генератор на шаблонах (ПР5) и диалоговая система (ПР6): {path} ({len(messages)} реплик) ---")
    template_replies, labels, results = [], [], []
    for i, text in enumerate(messages):
        reply, label = assistant.template_reply(template, text)
        result = assistant.engine.respond(text)
        template_replies.append(reply)
        labels.append(label)
        results.append(result)
        print(f"\n{i + 1}. Пользователь: {text}")
        print_wrapped(f"   Шаблоны ({label}):", reply, indent="      ")
        print_wrapped("   Бот:", result["reply"], indent="      ")
        print(f"      {describe(result)}")

    bot_replies = [r["reply"] for r in results]
    tones = Counter(labels)
    sources = Counter(r["source"] for r in results)
    print("\nСводка:")
    print(f"Шаблоны (ПР5): тональность по LSTM: " + ", ".join(f"{c} {tones[c]}" for c in CLASSES)
          + f"; разных ответов {len(set(template_replies))} из {len(messages)}, "
          f"средняя длина {np.mean([len(r.split()) for r in template_replies]):.0f} слов")
    print(f"Бот (ПР6): " + ", ".join(f"{name} {count}" for name, count in sources.most_common())
          + f"; разных ответов {len(set(bot_replies))} из {len(messages)}, "
          f"средняя длина {np.mean([len(r.split()) for r in bot_replies]):.0f} слов")


def search_topics(assistant, search, queries, use_index=False):
    """
    Тема лучшего найденного документа для каждой реплики.

    :param assistant: Assistant
    :param search: HybridSearch по документам корпуса диалогов
    :param queries: реплики в виде списков лемм
    :param use_index: искать через индекс Voyager (иначе полный перебор)
    :return: список тем
    """
    engine = assistant.engine
    return [assistant.dialogs[engine.doc_entry[search.search(q, 1, use_index)[0][0]]]["topic"] for q in queries]


def lsa_topics(assistant, queries):
    """
    Тема лучшего документа по векторам тем LSA из ПР2.

    :param assistant: Assistant
    :param queries: реплики в виде списков лемм
    :return: список тем (None, если LSA не нашла похожих документов)
    """
    engine = assistant.engine
    lsa = LSAModel(n_topics=100)
    lsa.fit([" ".join(doc) for doc in engine.documents])
    topics = []
    for query in queries:
        found = lsa.search(" ".join(query), top_n=1)
        topics.append(assistant.dialogs[engine.doc_entry[found[0][0]]]["topic"] if found else None)
    return topics


def print_evaluation(assistant):
    """
    Сравнивает подходы к выбору ответа на отложенных репликах, проверяет пороги на корпусе
    и сравнивает индекс Voyager с полным перебором.

    :param assistant: Assistant
    """
    engine = assistant.engine
    test = pd.read_csv(DATA_DIR / "test_questions.csv")
    known = test[test["topic"] != "unknown"]
    queries = [assistant.normalize(q) for q in known["question"]]
    answers = known["topic"].tolist()
    tones = assistant.tone.labels(known["question"].tolist())

    rows = []
    for title, vectors, alpha in (("TF-IDF (векторизатор ПР2)", assistant.domain_vectors, 1.0),
                                  ("Word2Vec ПР3, среднее векторов", assistant.pr3_vectors, 0.0),
                                  ("Word2Vec, дообученный", assistant.domain_vectors, 0.0)):
        search = HybridSearch(vectors, alpha=alpha)
        search.fit(engine.documents)
        rows.append((title, search_topics(assistant, search, queries)))
    rows.insert(1, ("LSA, 100 тем (ПР2)", lsa_topics(assistant, queries)))
    rows.append(("TF-IDF + Word2Vec, индекс Voyager", search_topics(assistant, engine.dialog_search, queries, True)))
    generated = [engine.generated_topic(q)[1] for q in queries]
    rows.append(("Seq2seq (порождающая модель)", [assistant.dialogs[i]["topic"] if i >= 0 else None for i in generated]))
    rows.append(("Гибрид: поиск + seq2seq", [assistant.dialogs[engine.choose(q)["entry"]]["topic"] for q in queries]))
    rows.append(("Гибрид + фильтр тональности", [assistant.dialogs[engine.choose(q, t)["entry"]]["topic"]
                                                 for q, t in zip(queries, tones)]))

    print(f"\n--- Выбор темы ответа на {len(queries)} отложенных репликах (data/test_questions.csv) ---")
    print(f"{'Подход':<36}Accuracy")
    for title, predicted in rows:
        accuracy = np.mean([p == a for p, a in zip(predicted, answers)])
        print(f"{title:<36}{accuracy:.3f}")
    novel = sum(1 for i in generated if i < 0)
    print(f"Seq2seq написал ответ, которого нет в корпусе, для {novel} реплик из {len(queries)}")

    outcomes = Counter()
    for question, topic in zip(test["question"], test["topic"]):
        engine.reset()
        result = engine.respond(question)
        if topic == "unknown":
            outcomes["вне темы: " + ("уточнение" if result["source"] == "уточнение" else "ответ")] += 1
        elif result["source"] in ("уточнение", "поиск по README"):
            outcomes[result["source"]] += 1
        else:
            outcomes["верно" if result["topic"] == topic else "неверно"] += 1
    engine.reset()
    answered = outcomes["верно"] + outcomes["неверно"]
    print(f"\n--- Ответы бота на все {len(test)} отложенных реплик (порог {engine.answer_threshold}) ---")
    print(f"По теме ({len(queries)}): верный ответ из корпуса {outcomes['верно']}, неверный {outcomes['неверно']}, "
          f"фрагмент README {outcomes['поиск по README']}, уточняющий вопрос {outcomes['уточнение']}")
    print(f"Не по теме ({len(test) - len(queries)}): уточняющий вопрос {outcomes['вне темы: уточнение']}, "
          f"ответ {outcomes['вне темы: ответ']}")
    print(f"Точность ответов из корпуса: {outcomes['верно'] / max(answered, 1):.2f}")

    print_thresholds(assistant)
    print_index_check(assistant, queries)


def print_thresholds(assistant):
    """
    Подбор порога ответа по корпусу: каждое высказывание по очереди убирается из поиска
    и ищется среди остальных (скользящий контроль). Seq2seq здесь не участвует, так как
    обучен на этих же высказываниях.

    :param assistant: Assistant
    """
    engine = assistant.engine
    scores, correct = [], []
    for doc, query in enumerate(assistant.statements):
        exact = engine.dialog_search.exact_scores(query)
        exact[doc] = -1.0
        best = int(exact.argmax())
        scores.append(exact[best])
        correct.append(engine.doc_entry[best] == engine.doc_entry[doc])
    scores, correct = np.array(scores), np.array(correct)
    print(f"\n--- Порог ответа: скользящий контроль на {len(scores)} высказываниях корпуса ---")
    print(f"{'Порог':<8}{'Доля ответов':<15}Точность ответов")
    for threshold in (0.3, 0.35, 0.4, 0.45, 0.5):
        answered = scores >= threshold
        print(f"{threshold:<8}{answered.mean():<15.2f}{correct[answered].mean():.2f}")


def print_index_check(assistant, queries):
    """
    Сравнивает приближенный поиск ближайших соседей (Voyager, HNSW) с полным перебором:
    на корпусе диалогов и на всем словаре Word2Vec, как в лекции 13.

    :param assistant: Assistant
    :param queries: отложенные реплики в виде списков лемм
    """
    search = assistant.engine.dialog_search
    same = sum(search.search(q, 1)[0][0] == search.search(q, 1, use_index=False)[0][0] for q in queries)
    print("\n--- Индекс Voyager и полный перебор ---")
    print(f"Корпус диалогов ({len(search.dense)} документов): лучший документ совпал для {same} реплик из {len(queries)}")

    words = assistant.domain_vectors.model.wv
    vectors = l2_normalize(words.vectors).astype("float32")
    start = time.time()
    index = Index(Space.Cosine, num_dimensions=vectors.shape[1], random_seed=1337)
    index.add_items(vectors)
    build = time.time() - start
    sample = np.random.default_rng(0).choice(len(vectors), 500, replace=False)
    start = time.time()
    exact = [np.argpartition(-(vectors @ vectors[i]), 10)[:10] for i in sample]
    exact_time = (time.time() - start) / len(sample)
    start = time.time()
    approx = [index.query(vectors[i], k=10)[0] for i in sample]
    approx_time = (time.time() - start) / len(sample)
    recall = np.mean([len(set(e) & set(a)) / 10 for e, a in zip(exact, approx)])
    print(f"Словарь Word2Vec ({len(vectors)} слов): построение индекса {build:.1f} с, recall@10 = {recall:.3f}")
    print(f"Время запроса: полный перебор {exact_time * 1000:.2f} мс, Voyager {approx_time * 1000:.2f} мс")
    for word in ("разметка", "видеокарта", "кадр"):
        if word in words.key_to_index:
            ids, _ = index.query(vectors[words.key_to_index[word]], k=6)
            neighbours = [words.index_to_key[int(i)] for i in ids if words.index_to_key[int(i)] != word]
            print(f"Ближайшие к «{word}»: " + ", ".join(neighbours))


def main():
    """
    Точка входа: разбор аргументов командной строки и запуск нужного режима.
    """
    parser = argparse.ArgumentParser(description="Диалоговая подсистема (чат-бот) генератора synthfire")
    parser.add_argument("--chat", action="store_true", help="Диалог с ботом в консоли")
    parser.add_argument("--ask", type=str, help="Ответить на одну реплику с подробным выводом")
    parser.add_argument("--dialog", type=str, help="Провести диалог по репликам из файла (одна на строку)")
    parser.add_argument("--compare", type=str, help="Сравнить с генератором на шаблонах из ПР5 на репликах из файла")
    parser.add_argument("--evaluate", action="store_true", help="Сравнить подходы на отложенных репликах")
    parser.add_argument("--train", action="store_true", help="Дообучить Word2Vec и обучить seq2seq заново")
    parser.add_argument("--readme", type=str, default=str(DATA_DIR / "synthfire_readme.md"),
                        help="README генератора для поиска ответов (по умолчанию data/synthfire_readme.md)")
    parser.add_argument("--seed", type=int, default=0, help="Зерно для генератора на шаблонах (по умолчанию 0)")
    args = parser.parse_args()

    if not any([args.chat, args.ask, args.dialog, args.compare, args.evaluate, args.train]):
        parser.print_help()
        return
    for path in (args.dialog, args.compare, args.readme):
        if path and not Path(path).exists():
            print(f"Файл {path} не найден!")
            return
    try:
        assistant = Assistant(Path(args.readme), retrain=args.train)
    except FileNotFoundError as error:
        print(error)
        return

    if args.ask:
        print_answer(assistant, args.ask)
    elif args.dialog:
        print_dialog(assistant, args.dialog)
    elif args.compare:
        print_comparison(assistant, args.compare, args.seed)
    elif args.evaluate:
        print_evaluation(assistant)
    elif args.chat:
        chat(assistant)


if __name__ == "__main__":
    main()
