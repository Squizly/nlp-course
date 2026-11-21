from itertools import product

import gensim

MODEL_FILE = "cbow.txt"
WORD_1 = "шишка"
WORD_2 = "лягушка"

NEIGHBORS = 60


model = gensim.models.KeyedVectors.load_word2vec_format(
    MODEL_FILE, binary=False
)


def closest_nouns(positive, negative=None, topn=10):
    """Находит ближайшие существительные к линейной комбинации векторов."""

    if negative is None:
        negative = []

    positive = [word + "_NOUN" for word in positive]
    negative = [word + "_NOUN" for word in negative]

    similar = model.most_similar(
        positive=positive,
        negative=negative,
        topn=topn * 20
    )

    nouns = [
        (word[:-5], score)
        for word, score in similar
        if word.endswith("_NOUN")
    ]

    return nouns[:topn]


def neighbours(word):
    """Находит ближайшие существительные, исключая заданные слова"""

    found = closest_nouns([word], topn=NEIGHBORS + 2)

    return [
        w for w, score in found
        if w not in (WORD_1, WORD_2)
    ][:NEIGHBORS]


near_1 = neighbours(WORD_1)
near_2 = neighbours(WORD_2)
best = None

for word_1, word_2 in product(near_1, near_2):

    # Проверяем три варианта линейной комбинации
    operations = [
        (f"{word_1} + {word_2}", [word_1, word_2], []),
        (f"{word_1} - {word_2}", [word_1], [word_2]),
        (f"{word_2} - {word_1}", [word_2], [word_1]),
    ]

    for formula, positive, negative in operations:

        top = closest_nouns(positive, negative, topn=10)
        words = [w for w, score in top]

        # Проверяем, попали ли оба исходных слова в первую десятку
        if WORD_1 in words and WORD_2 in words:

            places = words.index(WORD_1) + words.index(WORD_2)

            if best is None or places < best[0]:
                best = (places, formula, top)


if best is None:
    print("Формула не найдена, попробуйте увеличить NEIGHBORS.")
else:

    _, formula, top = best

    print("Найдена формула:", formula)
    print("\nТоп-10 ближайших существительных:")

    for place, (word, score) in enumerate(top, 1):

        mark = " <-- заданное слово" if word in (WORD_1, WORD_2) else ""

        print(f"{place:2d}. {word:15} {score:.4f}{mark}")

    print(f"\n{WORD_1} в топ-10: {WORD_1 in [w for w, s in top]}")
    print(f"{WORD_2} в топ-10: {WORD_2 in [w for w, s in top]}")
