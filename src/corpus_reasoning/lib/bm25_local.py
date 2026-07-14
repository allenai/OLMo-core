"""BM25 over an arbitrary in-memory corpus, backed by pyserini Lucene.

Companion to `scripts.lib.bm25.BM25Searcher`, which uses the pre-built
`wikipedia-dpr-100w` index. This wrapper instead indexes a caller-supplied
corpus (a list of `{_id, text}` dicts) the first time it is invoked, caches
the Lucene index under `index_dir`, and exposes the same batch_search /
search interface so downstream generators can swap the two with minimal
glue.

Used by OBLIQ-Bench data generation: BM25 distractor mining over each
subset's own corpus rather than Wikipedia.

Requires pyserini (lives in the corpus-reasoning-eval env).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from pyserini.search.lucene import LuceneSearcher


def build_lucene_index(corpus, index_dir, threads=8, force=False):
    """Index a list of `{_id, text}` dicts as a pyserini JsonCollection.

    Skips work if `index_dir/_SUCCESS` already exists (and `force=False`).
    """
    index_dir = Path(index_dir)
    marker = index_dir / "_SUCCESS"
    if marker.exists() and not force:
        return str(index_dir)

    index_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        coll_path = Path(tmpdir) / "corpus.jsonl"
        with open(coll_path, "w") as f:
            for doc in corpus:
                f.write(json.dumps({
                    "id": str(doc["_id"]),
                    "contents": doc["text"],
                }) + "\n")
        print(f"  Building Lucene index over {len(corpus):,} docs -> {index_dir}")
        subprocess.run([
            sys.executable, "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", tmpdir,
            "--index", str(index_dir),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(threads),
            "--storePositions", "--storeDocvectors", "--storeRaw",
        ], check=True)
    marker.write_text("done\n")
    return str(index_dir)


class LocalBM25Searcher:
    """BM25 over an arbitrary corpus, indexed once and cached on disk."""

    def __init__(self, corpus, index_dir, k1=0.9, b=0.4, threads=8):
        build_lucene_index(corpus, index_dir, threads=threads)
        self.searcher = LuceneSearcher(str(index_dir))
        self.searcher.set_bm25(k1=k1, b=b)
        self.num_docs = self.searcher.num_docs
        print(f"  Lucene index loaded: {self.num_docs:,} docs")

    def search(self, query, k=100):
        hits = self.searcher.search(query, k=k)
        return [{"docid": h.docid, "score": h.score} for h in hits]

    def batch_search(self, queries, k=100, threads=8):
        qids = [str(i) for i in range(len(queries))]
        raw = self.searcher.batch_search(queries, qids, k=k, threads=threads)
        return [
            [{"docid": h.docid, "score": h.score} for h in raw.get(qid, [])]
            for qid in qids
        ]
