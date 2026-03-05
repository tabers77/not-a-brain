"""Tests for n-gram language models."""

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.ngram import BigramModel, TrigramModel, NgramAgent


def make_tokenizer_and_corpus():
    corpus = [
        "ADD 5 3 =8",
        "ADD 1 2 =3",
        "ADD 7 1 =8",
        "COPY: abc|abc",
        "COPY: xyz|xyz",
        "COPY: hello|hello",
    ]
    tok = CharTokenizer()
    tok.fit(corpus)
    return tok, corpus


class TestBigramModel:
    def test_train_and_generate(self):
        tok, corpus = make_tokenizer_and_corpus()
        model = BigramModel(tok)
        model.train(corpus)
        output = model.generate("ADD ", max_len=10)
        assert len(output) > 4  # Should generate something beyond prompt

    def test_predict_next_returns_valid_id(self):
        tok, corpus = make_tokenizer_and_corpus()
        model = BigramModel(tok)
        model.train(corpus)
        next_id = model.predict_next(tok.bos_id)
        assert isinstance(next_id, int)
        assert next_id in tok.id_to_char

    def test_unseen_context_returns_eos(self):
        tok, corpus = make_tokenizer_and_corpus()
        model = BigramModel(tok)
        model.train(corpus)
        # Use an ID that was never seen as context
        next_id = model.predict_next(999)
        assert next_id == tok.eos_id


class TestTrigramModel:
    def test_train_and_generate(self):
        tok, corpus = make_tokenizer_and_corpus()
        model = TrigramModel(tok)
        model.train(corpus)
        output = model.generate("COPY: ", max_len=20)
        assert len(output) > 6

    def test_falls_back_to_bigram(self):
        tok, corpus = make_tokenizer_and_corpus()
        model = TrigramModel(tok)
        model.train(corpus)
        # Use an unseen pair — should fall back to bigram
        next_id = model.predict_next(tok.bos_id, tok.bos_id)
        assert isinstance(next_id, int)


class TestNgramAgent:
    def test_agent_interface(self):
        tok, corpus = make_tokenizer_and_corpus()
        model = BigramModel(tok)
        model.train(corpus)
        agent = NgramAgent(model, "bigram_test")
        result = agent.run("ADD 5 3 =")
        assert result.answer is not None
        assert agent.name == "bigram_test"

    def test_copy_task_attempt(self):
        tok = CharTokenizer()
        # Train on many copy examples to give it a chance
        corpus = [f"COPY: {w}|{w}" for w in ["abc", "xyz", "ab", "xy"] * 50]
        tok.fit(corpus)
        model = TrigramModel(tok)
        model.train(corpus)
        agent = NgramAgent(model, "trigram_test")
        result = agent.run("COPY: abc|")
        # It should at least generate something
        assert isinstance(result.answer, str)
