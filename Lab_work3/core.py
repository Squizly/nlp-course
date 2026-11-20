from transformers import pipeline, AutoTokenizer

MODEL = "DeepPavlov/rubert-base-cased-conversational"
W1, W2 = "стойло", "бряцания"

tok = AutoTokenizer.from_pretrained(MODEL)

print("Проверка токенизации:")
for w in (W1, W2):
    pieces = tok.tokenize(w)
    in_vocab = w in tok.get_vocab()
    print(f"  {w} -> {pieces}  | в словаре как целый токен: {in_vocab}")

fill = pipeline("fill-mask", model=MODEL, top_k=10)

def top10(s):
    return [(r["token_str"].strip(), round(r["score"], 4)) for r in fill(s)]

templates = [
    "Каждой лошади в конюшне отведено своё [MASK].",
    "В тишине конюшни слышалось только ржание и [MASK].",
    "В конюшне для каждой лошади было своё [MASK].",
    "Лошадь завели в своё [MASK].",
    "Конюх приготовил для коня отдельное [MASK].",
]

print("\nРезультаты:")
for t in templates:
    res = top10(t)
    words = [w for w, _ in res]
    status = "OK  " if W1 in words and W2 in words else ("W1  " if W1 in words else "--  ")
    print(status + t)
    print("    ", res)
    print()