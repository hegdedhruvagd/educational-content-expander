"""Command-line interface for training and using the local n-gram model."""

from __future__ import annotations

import argparse

from app.core import generate_text, load_model, save_model, train_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="educational-content-expander",
        description="Expand short educational prompts with a local bigram/trigram model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train the model from the bundled corpus")
    train.add_argument("--output", "-o", default="data/model.json", help="where to save the model (default: data/model.json)")
    train.add_argument("--k", type=float, default=1.0, help="add-k smoothing value (default: 1.0)")

    expand = sub.add_parser("expand", help="expand a short educational prompt")
    expand.add_argument("text", nargs="+", help="prompt text to expand")
    expand.add_argument("--model", "-m", help="load a saved model (default: train from bundled corpus)")
    expand.add_argument("--words", "-w", type=int, default=40, help="number of words to append (default: 40)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            if args.k < 0:
                parser.error("--k must be non-negative")
            model = train_model(k=args.k)
            save_model(model, args.output)
            print(f"Trained model on {sum(model.unigram_counts.values())} token occurrences; saved to {args.output}")
        elif args.command == "expand":
            model = load_model(args.model) if args.model else train_model()
            print(generate_text(" ".join(args.text), model=model, max_words=args.words))
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
