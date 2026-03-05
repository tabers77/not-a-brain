"""Character-level and simple BPE tokenizers."""

from collections import Counter


class CharTokenizer:
    """Character-level tokenizer with special tokens.

    Maps each character to an integer. Simple, transparent, no surprises.
    This is intentionally naive — the point is to show how tokenization
    choices affect what's learnable.
    """

    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"

    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self._special_tokens = [self.PAD, self.BOS, self.EOS, self.UNK]
        self._fitted = False

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    @property
    def pad_id(self) -> int:
        return self.char_to_id[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.char_to_id[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.char_to_id[self.EOS]

    @property
    def unk_id(self) -> int:
        return self.char_to_id[self.UNK]

    def fit(self, texts: list[str]) -> "CharTokenizer":
        """Build vocabulary from a list of strings."""
        chars = sorted(set(c for text in texts for c in text))
        self.char_to_id = {}
        for i, tok in enumerate(self._special_tokens):
            self.char_to_id[tok] = i
        for c in chars:
            if c not in self.char_to_id:
                self.char_to_id[c] = len(self.char_to_id)
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self._fitted = True
        return self

    def encode(self, text: str, add_bos: bool = False,
               add_eos: bool = False) -> list[int]:
        """Convert string to list of token IDs."""
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for c in text:
            ids.append(self.char_to_id.get(c, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        """Convert list of token IDs back to string."""
        chars = []
        for i in ids:
            tok = self.id_to_char.get(i, self.UNK)
            if strip_special and tok in self._special_tokens:
                continue
            chars.append(tok)
        return "".join(chars)

    def encode_batch(self, texts: list[str], add_bos: bool = False,
                     add_eos: bool = False,
                     pad_to: int | None = None) -> list[list[int]]:
        """Encode multiple strings, optionally padding to same length."""
        encoded = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]
        if pad_to is not None:
            for seq in encoded:
                while len(seq) < pad_to:
                    seq.append(self.pad_id)
        return encoded


class SimpleBPE:
    """Minimal byte-pair encoding for educational purposes.

    Shows how subword tokenization works: start with characters,
    iteratively merge the most frequent pair.
    """

    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"

    def __init__(self, num_merges: int = 50):
        self.num_merges = num_merges
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def fit(self, texts: list[str]) -> "SimpleBPE":
        """Learn BPE merges from texts."""
        # Start with character-level tokens
        words = []
        for text in texts:
            tokens = list(text)
            words.append(tokens)

        # Initialize vocab with characters + special tokens
        special = [self.PAD, self.BOS, self.EOS, self.UNK]
        all_chars = sorted(set(c for text in texts for c in text))
        self.vocab = {tok: i for i, tok in enumerate(special + all_chars)}

        # Iteratively merge most frequent pair
        self.merges = []
        for _ in range(self.num_merges):
            pair_counts: Counter = Counter()
            for tokens in words:
                for i in range(len(tokens) - 1):
                    pair_counts[(tokens[i], tokens[i + 1])] += 1

            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            merged = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)

            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)

            # Apply merge to all words
            new_words = []
            for tokens in words:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (i < len(tokens) - 1 and
                            tokens[i] == best_pair[0] and
                            tokens[i + 1] == best_pair[1]):
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_words.append(new_tokens)
            words = new_words

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        return self

    def _apply_merges(self, text: str) -> list[str]:
        """Tokenize a string by applying learned merges."""
        tokens = list(text)
        for a, b in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                        tokens[i] == a and tokens[i + 1] == b):
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str) -> list[int]:
        tokens = self._apply_merges(text)
        unk_id = self.vocab[self.UNK]
        return [self.vocab.get(t, unk_id) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        special = {self.PAD, self.BOS, self.EOS, self.UNK}
        return "".join(
            self.id_to_token.get(i, self.UNK)
            for i in ids
            if self.id_to_token.get(i, self.UNK) not in special
        )
