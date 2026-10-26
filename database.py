"""
database.py
============
VectorDB (16D demo index over BruteForce + KD-Tree + HNSW) and
DocumentDB (HNSW-only index over real Ollama embeddings), plus the
text chunker and the built-in demo dataset. Ported from main.cpp.
"""

import threading
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

from vector_engine import (
    VectorItem, BruteForce, KDTree, HNSW, DistFn, get_dist_fn, cosine
)

DIMS = 16  # dimensionality of the demo vectors


# =====================================================================
#  VECTOR DATABASE (demo 16D index)
# =====================================================================

class VectorDB:
    def __init__(self, dims: int):
        self.dims = dims
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.next_id = 1
        self.lock = threading.RLock()

    def insert(self, meta: str, cat: str, emb: List[float], dist: DistFn) -> int:
        with self.lock:
            v = VectorItem(self.next_id, meta, cat, emb)
            self.next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, id_: int) -> bool:
        with self.lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.bf.remove(id_)
            self.hnsw.remove(id_)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: List[float], k: int, metric: str, algo: str) -> dict:
        with self.lock:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)
            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, id_ in raw:
                v = self.store.get(id_)
                if v is not None:
                    hits.append({"id": id_, "meta": v.metadata, "cat": v.category,
                                 "emb": v.emb, "dist": d})
            return {"hits": hits, "us": us, "algo": algo, "metric": metric}

    def benchmark(self, q: List[float], k: int, metric: str) -> dict:
        with self.lock:
            dfn = get_dist_fn(metric)

            def timeit(fn):
                t0 = time.perf_counter()
                fn()
                return int((time.perf_counter() - t0) * 1_000_000)

            bf_us = timeit(lambda: self.bf.knn(q, k, dfn))
            kd_us = timeit(lambda: self.kdt.knn(q, k, dfn))
            hnsw_us = timeit(lambda: self.hnsw.knn(q, k, 50, dfn))
            return {"bfUs": bf_us, "kdUs": kd_us, "hnswUs": hnsw_us, "n": len(self.store)}

    def all(self) -> List[VectorItem]:
        with self.lock:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        with self.lock:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.lock:
            return len(self.store)


# =====================================================================
#  DOCUMENT DATABASE — HNSW over real Ollama embeddings
# =====================================================================

@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: List[float]


class DocumentDB:
    def __init__(self):
        self.store: Dict[int, DocItem] = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.next_id = 1
        self.dims = 0
        self.lock = threading.RLock()

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self.lock:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: List[float], k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        with self.lock:
            if not self.store:
                return []
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            else:
                raw = self.hnsw.knn(q, k, 50, cosine)
            out = []
            for d, id_ in raw:
                item = self.store.get(id_)
                if item is not None and d <= max_dist:
                    out.append((d, item))
            return out

    def remove(self, id_: int) -> bool:
        with self.lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.hnsw.remove(id_)
            self.bf.remove(id_)
            return True

    def all(self) -> List[DocItem]:
        with self.lock:
            return list(self.store.values())

    def size(self) -> int:
        with self.lock:
            return len(self.store)

    def get_dims(self) -> int:
        return self.dims


# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = chunk_words - overlap_words
    i = 0
    n = len(words)
    while i < n:
        end = min(i + chunk_words, n)
        chunks.append(" ".join(words[i:end]))
        if end == n:
            break
        i += step
    return chunks


# =====================================================================
#  DEMO DATA (16D categorical vectors)
# =====================================================================

def load_demo(db: VectorDB) -> None:
    dist = get_dist_fn("cosine")

    def add(meta, cat, emb):
        db.insert(meta, cat, emb, dist)

    # Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
    add("Linked List: nodes connected by pointers", "cs",
        [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06])
    add("Binary Search Tree: O(log n) search and insert", "cs",
        [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10])
    add("Dynamic Programming: memoization overlapping subproblems", "cs",
        [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07])
    add("Graph BFS and DFS: breadth and depth first traversal", "cs",
        [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07])
    add("Hash Table: O(1) lookup with collision chaining", "cs",
        [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09])
    add("Calculus: derivatives integrals and limits", "math",
        [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10])
    add("Linear Algebra: matrices eigenvalues eigenvectors", "math",
        [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09])
    add("Probability: distributions random variables Bayes theorem", "math",
        [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08])
    add("Number Theory: primes modular arithmetic RSA cryptography", "math",
        [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06])
    add("Combinatorics: permutations combinations generating functions", "math",
        [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07])
    add("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
        [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07])
    add("Sushi: vinegared rice raw fish and nori rolls", "food",
        [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08])
    add("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
        [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06])
    add("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
        [0.07, 0.09, 0.08, 0.06, 0.06, 0.07, 0.09, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09])
    add("Croissant: laminated pastry with buttery flaky layers", "food",
        [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06])
    add("Basketball: fast-paced shooting dribbling slam dunks", "sports",
        [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72])
    add("Football: tackles touchdowns field goals and strategy", "sports",
        [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76])
    add("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
        [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82])
    add("Chess: openings endgames tactics strategic board game", "sports",
        [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90])
    add("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
        [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80])
