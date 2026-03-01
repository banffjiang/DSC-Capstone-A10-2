import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Optional

from make_wiki import chunk_sentences

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train_100M"
DEFAULT_OUT_PATH = BASE_DIR / "train_100m_passages.jsonl"


def normalize_source_name(path: Path) -> str:
    if path.stem == "simple_wiki":
        return "simplewiki"
    return path.stem


def clean_common_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\{\{[^}]*\}\}", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"={2,}.*?={2,}", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not re.search(r"[A-Za-z]", text):
        return ""

    return text


def clean_line(source: str, line: str) -> str:
    text = line.strip()
    if not text:
        return ""

    if source == "childes":
        text = re.sub(r"^\*[A-Z]{3}:\s*", "", text)
    elif source == "switchboard":
        text = re.sub(r"^[A-Z]:\s*", "", text)
    elif source == "open_subtitles":
        text = re.sub(r"^-+\s*", "", text)

    text = clean_common_text(text)
    if not text:
        return ""

    upper_ratio = sum(1 for ch in text if ch.isupper()) / max(1, sum(1 for ch in text if ch.isalpha()))
    if source == "gutenberg":
        if "project gutenberg" in text.lower():
            return ""
        if upper_ratio > 0.75 and len(text.split()) >= 4:
            return ""

    return text


def iter_train_files(train_dir: Path) -> Iterable[Path]:
    for path in sorted(train_dir.glob("*.train")):
        if path.is_file():
            yield path


def build_dataset(
    train_dir: Path,
    output_path: Path,
    min_words: int,
    max_words: Optional[int],
    target_passage_words: int,
    max_instances: Optional[int],
) -> int:
    n = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for in_path in iter_train_files(train_dir):
            source = normalize_source_name(in_path)
            with in_path.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    line = clean_line(source, raw_line)
                    if not line:
                        continue

                    for passage in chunk_sentences(line, target_words=target_passage_words):
                        wc = len(passage.split())
                        if wc < min_words:
                            continue
                        if max_words is not None and wc > max_words:
                            continue

                        ex = {"id": n, "source": source, "passage": passage}
                        out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        n += 1

                        if max_instances is not None and n >= max_instances:
                            return n
    return n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all train_100M/*.train files into wiki-style passage JSONL."
    )
    parser.add_argument(
        "--train_dir",
        default=str(TRAIN_DIR),
        help="Directory containing .train files.",
    )
    parser.add_argument(
        "--output_path",
        default=str(DEFAULT_OUT_PATH),
        help="Output JSONL path.",
    )
    parser.add_argument("--min_words", type=int, default=20)
    parser.add_argument("--max_words", type=int, default=None)
    parser.add_argument("--target_passage_words", type=int, default=80)
    parser.add_argument("--max_instances", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = Path(args.train_dir)
    output_path = Path(args.output_path)
    written = build_dataset(
        train_dir=train_dir,
        output_path=output_path,
        min_words=args.min_words,
        max_words=args.max_words,
        target_passage_words=args.target_passage_words,
        max_instances=args.max_instances,
    )
    print(f"Wrote {written} passages to {output_path}")


if __name__ == "__main__":
    main()
