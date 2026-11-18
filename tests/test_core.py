import pytest

from app.core import NGramModel, load_corpus, preprocess, tokenize


def test_milestone_1_loads_original_corpus_and_preprocesses_text(tmp_path):
    corpus = load_corpus()
    assert len(corpus) >= 5
    assert any("photosynthesis" in document.lower() for document in corpus)
    assert tokenize("Leaves, WATER, and a plant's roots!") == ["leaves", "water", "and", "a", "plant's", "roots"]
    assert preprocess("  Earth spins.  ") == ["earth", "spins"]

    path = tmp_path / "small.txt"
    path.write_text("First lesson.\n\nSecond lesson.\n", encoding="utf-8")
    assert load_corpus(path) == ["First lesson.", "Second lesson."]


def test_milestone_2_counts_ngrams_and_calculates_smoothed_probabilities():
    model = NGramModel(k=0.5).fit(["red blue red blue green"])
    assert model.bigram_counts[("red", "blue")] == 2
    assert model.trigram_counts[("red", "blue", "red")] == 1
    assert model.probability("blue", ["red"]) == pytest.approx((2 + 0.5) / (2 + 0.5 * model.vocabulary_size))
    assert model.probability("green", ["missing", "context"]) > 0
    assert model.probability("not-in-vocabulary", ["red", "blue"]) > 0
    assert sum(model.probability(word, ["red"]) for word in model.vocabulary) == pytest.approx(1.0)


def test_milestone_2_handles_invalid_smoothing_and_contexts():
    with pytest.raises(ValueError, match="non-negative"):
        NGramModel(k=-1)
    model = NGramModel().fit(["science facts"])
    with pytest.raises(ValueError, match="at most two"):
        model.probability("facts", ["too", "many", "tokens"])
