import pytest

from app.core import NGramModel, generate_text, load_corpus, load_model, preprocess, save_model, tokenize, train_model


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


def test_milestone_3_expands_prompt_with_relevant_local_continuation():
    prompt = "Plants use sunlight"
    expanded = generate_text(prompt, max_words=5)
    assert expanded.startswith(prompt)
    assert expanded[len(prompt):].strip().split()[:3] == ["water", "and", "carbon"]
    assert len(expanded[len(prompt):].split()) == 5
    assert generate_text(prompt, max_words=0) == prompt


def test_milestone_3_rejects_invalid_generation_requests():
    with pytest.raises(ValueError, match="prompt"):
        generate_text("   ")
    with pytest.raises(ValueError, match="non-negative"):
        generate_text("lesson", max_words=-1)


def test_milestone_4_cli_trains_saves_loads_and_expands(tmp_path, capsys):
    from app.cli import main

    model_path = tmp_path / "trained.json"
    main(["train", "--output", str(model_path)])
    assert model_path.exists()
    assert "saved to" in capsys.readouterr().out

    main(["expand", "Plants", "use", "sunlight", "--model", str(model_path), "--words", "3"])
    output = capsys.readouterr().out.strip()
    assert output.startswith("Plants use sunlight")
    assert output.endswith("water and") or output.endswith("water and carbon")


def test_milestone_4_model_save_load_round_trip(tmp_path):
    original = train_model(k=0.25)
    model_path = tmp_path / "model.json"

    save_model(original, model_path)
    restored = load_model(model_path)

    assert restored.k == original.k
    assert restored.vocabulary == original.vocabulary
    assert restored.bigram_counts == original.bigram_counts
    assert restored.trigram_counts == original.trigram_counts
    assert restored.probability("water", ["sunlight"]) == original.probability("water", ["sunlight"])


def test_milestone_4_cli_help_and_actionable_errors(capsys):
    from app.cli import main

    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    assert "train" in capsys.readouterr().out

    with pytest.raises(SystemExit) as error_exit:
        main(["expand", "lesson", "--model", "missing-model.json"])
    assert error_exit.value.code == 2
    assert "Model file not found" in capsys.readouterr().err
