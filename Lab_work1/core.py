import nltk
import pymorphy3
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

FILE_PATH = "input.txt"

morph = pymorphy3.MorphAnalyzer()

ALLOWED_POS = {"NOUN", "ADJF"}

def find_pairs():
    pairs = []

    for sentence in read_sentences(FILE_PATH):
        words = [t.lower() for t in word_tokenize(sentence, language="russian") if t.isalpha()]

        for i in range(len(words) - 1):
            w1 = morph.parse(words[i])[0]
            w2 = morph.parse(words[i + 1])[0]

            if w1.tag.POS not in ALLOWED_POS or w2.tag.POS not in ALLOWED_POS:
                continue

            same_number = w1.tag.number == w2.tag.number
            same_case = w1.tag.case == w2.tag.case
            same_gender = w1.tag.number == "plur" or w1.tag.gender == w2.tag.gender # во множественном числе рода нет

            if same_number and same_case and same_gender:
                pairs.append(w1.normal_form + " " + w2.normal_form)
            
    return pairs

def read_sentences(filename):
    with open(filename, encoding="utf-8") as f:
        return sent_tokenize(f.read(), language="russian")

if __name__ == "__main__":
    for pair in find_pairs():
        print(pair)