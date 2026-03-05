"""Tests for tokenizers."""

from not_a_brain.models.tokenizer import CharTokenizer, SimpleBPE


class TestCharTokenizer:
    def test_fit_and_encode(self):
        tok = CharTokenizer()
        tok.fit(["hello", "world"])
        ids = tok.encode("hello")
        assert len(ids) == 5
        assert all(isinstance(i, int) for i in ids)

    def test_roundtrip(self):
        tok = CharTokenizer()
        tok.fit(["hello world", "test 123"])
        for text in ["hello", "world", "test", "123"]:
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text

    def test_special_tokens(self):
        tok = CharTokenizer()
        tok.fit(["abc"])
        ids = tok.encode("abc", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id
        # Decode with strip should remove them
        decoded = tok.decode(ids, strip_special=True)
        assert decoded == "abc"

    def test_unknown_chars(self):
        tok = CharTokenizer()
        tok.fit(["abc"])
        ids = tok.encode("xyz")
        assert all(i == tok.unk_id for i in ids)

    def test_vocab_size(self):
        tok = CharTokenizer()
        tok.fit(["abc"])
        # 4 special + 3 chars
        assert tok.vocab_size == 7

    def test_batch_encode(self):
        tok = CharTokenizer()
        tok.fit(["hello", "hi"])
        batch = tok.encode_batch(["hello", "hi"], pad_to=5)
        assert len(batch) == 2
        assert len(batch[0]) == 5
        assert len(batch[1]) == 5


class TestSimpleBPE:
    def test_fit(self):
        bpe = SimpleBPE(num_merges=10)
        bpe.fit(["hello hello hello world world"])
        assert bpe.vocab_size > 0

    def test_roundtrip(self):
        bpe = SimpleBPE(num_merges=20)
        texts = ["hello hello hello world world world"]
        bpe.fit(texts)
        for text in ["hello", "world"]:
            ids = bpe.encode(text)
            decoded = bpe.decode(ids)
            assert decoded == text

    def test_merges_reduce_tokens(self):
        bpe = SimpleBPE(num_merges=50)
        text = "abababababab"
        bpe.fit([text])
        ids_merged = bpe.encode(text)
        # After merges, should be fewer tokens than characters
        assert len(ids_merged) < len(text)
