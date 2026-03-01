import argparse
import json
import math
import os
import re
from collections import Counter

from data.dataloader_wiki import SimpleWikiPassageLoader

TOKEN_RE = re.compile(r"[a-z0-9]+")  # lowercase only (we lowercase input)

def tokens_set(text: str, min_len: int = 2):
    text = text.lower()
    toks = TOKEN_RE.findall(text)
    return set(t for t in toks if len(t) >= min_len)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki_jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min_len", type=int, default=2)
    args = ap.parse_args()

    df = Counter()
    N = 0
    for ex in SimpleWikiPassageLoader(path=args.wiki_jsonl, limit=args.limit):
        txt = ex.get("passage", "")
        if not txt:
            continue
        N += 1
        df.update(tokens_set(txt, min_len=args.min_len))

    # smooth IDF avoids divide by zero
    idf = {t: math.log((N + 1.0) / (c + 1.0)) + 1.0 for t, c in df.items()}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"N_docs": N, "min_len": args.min_len, "idf": idf}, f)

    print(f"saved idf: docs={N} vocab={len(idf)} -> {args.out}")

if __name__ == "__main__":
    main()