import json
import re
import unicodedata

import torch


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)


def _is_letter(ch):
    return unicodedata.category(ch).startswith("L")


def _is_number(ch):
    return unicodedata.category(ch).startswith("N")


_CONTRACTIONS = ("s", "t", "re", "ve", "m", "ll", "d")


def _match_at(text, i):
    """Match one pre-token at position i, trying alternatives in pattern order."""
    n = len(text)
    ch = text[i]
    if ch == "'":
        for contraction in _CONTRACTIONS:
            end = i + 1 + len(contraction)
            if text[i + 1:end].lower() == contraction:
                return text[i:end], end
    if ch not in "\r\n" and not _is_letter(ch) and not _is_number(ch):
        if i + 1 < n and _is_letter(text[i + 1]):
            j = i + 2
            while j < n and _is_letter(text[j]):
                j += 1
            return text[i:j], j
    if _is_letter(ch):
        j = i + 1
        while j < n and _is_letter(text[j]):
            j += 1
        return text[i:j], j
    if _is_number(ch):
        return ch, i + 1
    j = i + (1 if ch == " " else 0)
    k = j
    while k < n and text[k] not in "\r\n" and not _is_letter(text[k]) and not _is_number(text[k]) and not text[k].isspace():
        k += 1
    if k > j:
        while k < n and text[k] in "\r\n":
            k += 1
        return text[i:k], k
    if ch.isspace():
        j = i
        while j < n and text[j].isspace():
            j += 1
        last_nl = -1
        for p in range(i, j):
            if text[p] in "\r\n":
                last_nl = p
        if last_nl >= 0:
            return text[i:last_nl + 1], last_nl + 1
        if j == n:
            return text[i:j], j
        if j - 1 > i:
            return text[i:j - 1], j - 1
        return text[i:j], j
    return ch, i + 1


def _split_gpt2(text):
    """Split text following the GPT-2/Qwen pre-tokenizer pattern exactly.

    ('s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|
     ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+
    """
    tokens = []
    i, n = 0, len(text)
    while i < n:
        token, i = _match_at(text, i)
        tokens.append(token)
    return tokens


class QwenBpeTokenizer:
    """Native BPE tokenizer for Qwen-style tokenizer.json files. No third-party dependencies."""

    def __init__(self, source, device=None):
        if isinstance(source, dict):
            data = source
        else:
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
        model = data["model"]
        self.vocab = model["vocab"]
        self.merges = model["merges"]
        ranks = {}
        for i, m in enumerate(self.merges):
            pair = tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1))
            ranks[pair] = i
        self.bpe_ranks = ranks
        self.byte_encoder = bytes_to_unicode()
        self.added_tokens = {}
        for tok in data.get("added_tokens", []):
            self.added_tokens[tok["content"]] = tok["id"]
        self._added_re = None
        if self.added_tokens:
            self._added_re = re.compile("|".join(re.escape(t) for t in sorted(self.added_tokens, key=len, reverse=True)))
        self.cache = {}
        self.device = device

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        result = " ".join(word)
        self.cache[token] = result
        return result

    def _encode_chunk(self, text):
        ids = []
        for token in _split_gpt2(text):
            token_bytes = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            for bpe_token in self._bpe(token_bytes).split(" "):
                ids.append(self.vocab[bpe_token])
        return ids

    def _encode_str(self, text):
        text = unicodedata.normalize("NFC", text)
        ids = []
        if self._added_re is not None:
            pos = 0
            for m in self._added_re.finditer(text):
                if m.start() > pos:
                    ids.extend(self._encode_chunk(text[pos:m.start()]))
                ids.append(self.added_tokens[m.group(0)])
                pos = m.end()
            if pos < len(text):
                ids.extend(self._encode_chunk(text[pos:]))
        else:
            ids = self._encode_chunk(text)
        return ids

    def encode(self, text):
        return torch.tensor([self._encode_str(text)], dtype=torch.long, device=self.device)

    def encode_tagged(self, text):
        """Encode text, tokenizing [nonverbal] tags standalone."""
        ids = []
        pos = 0
        for m in _NONVERBAL_PATTERN.finditer(text):
            if m.start() > pos:
                ids.extend(self._encode_str(text[pos:m.start()]))
            ids.extend(self._encode_str(m.group()))
            pos = m.end()
        if pos < len(text):
            ids.extend(self._encode_str(text[pos:]))
        return torch.tensor([ids], dtype=torch.long, device=self.device)
