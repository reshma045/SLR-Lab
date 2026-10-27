"""Sentence corpus for gloss-free translation, built from the 50-sign vocabulary.

Each entry pairs a sign sequence (JSL word order, used only to *stitch* the motion) with a natural
Japanese sentence and an English translation. The gloss-free model never sees the sign sequence:
it learns motion -> Japanese text directly (PoseStitch-SLT-style synthetic pretraining data).
Templates are hand-written so every Japanese target is grammatical; slots are filled with words that
make sense in that slot.
"""
from __future__ import annotations

import itertools
import random

SPORT = {"ゴルフ": "golf", "サッカー": "soccer", "スキー": "skiing", "スケート": "skating", "バレーボール": "volleyball"}
SUMMER_SPORT = {k: SPORT[k] for k in ("ゴルフ", "サッカー", "バレーボール")}
SKILL = {**SPORT, "ピアノ": "the piano"}
DRINK = {"コーヒー": "coffee", "ビール": "beer"}
FOOD = {"ラーメン": "ramen", "魚": "fish"}
SHOW = {"テレビ": "TV", "映画": "a movie"}
GIVER = {"兄": ("兄", "my older brother"), "家族": ("家族", "my family"), "みんな": ("みんな", "everyone")}
WELL = {"兄": ("兄", "My older brother"), "家族": ("家族", "My family"), "みんな": ("みんな", "Everyone")}
ANIMAL = {"犬": "a dog", "ネズミ": "a mouse", "魚": "a fish"}
THINGS = {**DRINK, **FOOD}


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def build() -> list[dict]:
    """All (signs, ja, en) entries, deduplicated."""
    E: list[tuple[list[str], str, str]] = []
    for d, en in DRINK.items():
        E.append((["今日", d, "飲む"], f"今日は{d}を飲みます。", f"Today I will drink {en}."))
        E.append((["会社", "休憩", d, "飲む"], f"会社の休憩で{d}を飲みます。", f"I drink {en} on my break at work."))
    for x, en in THINGS.items():
        E.append(([x, "嫌い"], f"{x}が嫌いです。", f"I don't like {en}."))
        for g, (gj, ge) in GIVER.items():
            E.append(([g, x, "もらう"], f"{gj}に{x}をもらいました。", f"I received {en} from {ge}."))
    for s, en in SKILL.items():
        E.append(([s, "苦手"], f"{s}が苦手です。", f"I'm not good at {en}."))
        E.append(([s, "とても", "苦手"], f"{s}はとても苦手です。", f"I'm really not good at {en}."))
        E.append(([s, "下手"], f"{s}が下手です。", f"I'm bad at {en}."))
        E.append(([s, "楽しい"], f"{s}は楽しいです。", f"{_cap(en)} is fun."))
        E.append(([s, "とても", "楽しい"], f"{s}はとても楽しいです。", f"{_cap(en)} is a lot of fun."))
        E.append(([s, "下手", "けれども", "楽しい"], f"{s}は下手だけど楽しいです。", f"I'm bad at {en}, but it's fun."))
        E.append((["兄", s, "教わる"], f"兄に{s}を教わりました。", f"My older brother taught me {en}."))
    for s, en in SUMMER_SPORT.items():
        E.append((["夏", s, "楽しい"], f"夏の{s}は楽しいです。", f"{_cap(en)} in summer is fun."))
    for v, en in SHOW.items():
        E.append((["家族", "一緒", v, "みる"], f"家族と一緒に{v}を見ます。", f"I watch {en} together with my family."))
        E.append(([v, "みる", "泣く"], f"{v}を見て泣きました。", f"I cried watching {en}."))
        E.append(([v, "みる", "楽しい"], f"{v}を見るのは楽しいです。", f"Watching {en} is fun."))
        E.append((["今日", v, "みる"], f"今日は{v}を見ます。", f"Today I will watch {en}."))
    for p, (pj, pe) in WELL.items():
        E.append(([p, "元気"], f"{pj}は元気です。", f"{pe} is doing well."))
        E.append(([p, "ありがとう"], f"{pj}、ありがとう。", f"Thank you, {pe[0].lower() + pe[1:]}."))
    for a, en in ANIMAL.items():
        aj = a
        E.append(([a, "いる"], f"{aj}がいます。", f"There is {en}."))
        E.append((["家", a, "いる"], f"家に{aj}がいます。", f"There is {en} at home."))
    E.append((["犬", "一緒", "安心"], "犬と一緒だと安心します。", "I feel at ease with my dog."))
    E.append((["家族", "一緒", "安心"], "家族と一緒だと安心します。", "I feel at ease with my family."))
    for a, b in itertools.permutations(list(DRINK) + list(FOOD), 2):
        if (a in DRINK) == (b in DRINK):
            E.append(([a, b, "どちら"], f"{a}と{b}、どちらがいいですか。",
                      f"Which would you like, {THINGS[a]} or {THINGS[b]}?"))
    E += [
        (["今日", "雨", "〜から", "家", "帰る"], "今日は雨だから家に帰ります。", "It's raining today, so I'm going home."),
        (["雨", "嫌い"], "雨が嫌いです。", "I don't like rain."),
        (["今日", "雨"], "今日は雨です。", "It's raining today."),
        (["今日", "元気"], "今日は元気です。", "I'm feeling well today."),
        (["今日", "会社", "休憩"], "今日は会社で休憩します。", "Today I'm taking a break at work."),
        (["会社", "〜から", "帰る"], "会社から帰ります。", "I'm going home from work."),
        (["スポーツ", "いろいろ", "ある"], "スポーツはいろいろあります。", "There are many kinds of sports."),
        (["サークル", "いくつか", "ある"], "サークルがいくつかあります。", "There are several clubs."),
        (["おとな", "みんな", "ビール", "飲む"], "大人はみんなビールを飲みます。", "All the adults drink beer."),
        (["兄", "ほとんど", "家", "いる"], "兄はほとんど家にいます。", "My older brother is almost always at home."),
        (["映画", "みる", "考える"], "映画を見るか考えています。", "I'm thinking about watching a movie."),
        (["夏", "とても", "楽しい"], "夏はとても楽しいです。", "Summer is a lot of fun."),
        (["テレビ", "サッカー", "みる"], "テレビでサッカーを見ます。", "I watch soccer on TV."),
        (["家族", "一緒", "ラーメン", "楽しい"], "家族と一緒にラーメンを食べるのは楽しいです。", "Eating ramen with my family is fun."),
    ]
    seen, out = set(), []
    for signs, ja, en in E:
        if ja not in seen:
            seen.add(ja)
            out.append({"signs": signs, "ja": ja, "en": en})
    return out


def split(entries: list[dict], test_frac: float = 0.2, seed: int = 3) -> tuple[list[dict], list[dict]]:
    """Hold out whole sentences (unseen word combinations) for testing; every sign stays in training."""
    rng = random.Random(seed)
    idx = list(range(len(entries)))
    rng.shuffle(idx)
    test, train_signs = [], set()
    n_test = int(len(entries) * test_frac)
    for i in idx:
        e = entries[i]
        if len(test) < n_test and all(s in train_signs for s in e["signs"]):
            test.append(e)
        else:
            train_signs.update(e["signs"])
    test_ja = {e["ja"] for e in test}
    return [e for e in entries if e["ja"] not in test_ja], test


class CharVocab:
    """Character-level tokenizer for Japanese targets (no word segmentation needed)."""
    PAD, BOS, EOS = 0, 1, 2

    def __init__(self, texts: list[str]) -> None:
        self.chars = ["<pad>", "<bos>", "<eos>"] + sorted({c for t in texts for c in t})
        self.index = {c: i for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        return [self.BOS] + [self.index[c] for c in text if c in self.index] + [self.EOS]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i == self.EOS:
                break
            if i > self.EOS:
                out.append(self.chars[i])
        return "".join(out)
