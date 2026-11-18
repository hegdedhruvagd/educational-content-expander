from app.core import load_corpus, preprocess, tokenize


def test_milestone_1_loads_original_corpus_and_preprocesses_text(tmp_path):
    corpus = load_corpus()
    assert len(corpus) >= 5
    assert any("photosynthesis" in document.lower() for document in corpus)
    assert tokenize("Leaves, WATER, and a plant's roots!") == ["leaves", "water", "and", "a", "plant's", "roots"]
    assert preprocess("  Earth spins.  ") == ["earth", "spins"]

    path = tmp_path / "small.txt"
    path.write_text("First lesson.\n\nSecond lesson.\n", encoding="utf-8")
    assert load_corpus(path) == ["First lesson.", "Second lesson."]
