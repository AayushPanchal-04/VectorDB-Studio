"""
vector_engine.py
=================
Core search engine: distance metrics + three nearest-neighbor algorithms
(Brute Force, KD-Tree, HNSW). This is a direct, behavior-preserving port
of the original C++ implementation (main.cpp) to Python.
"""

import math
import heapq
import random
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Optional


# =====================================================================
#  DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: List[float]


DistFn = Callable[[List[float], List[float]], float]


# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: List[float], b: List[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


def cosine(a: List[float], b: List[float]) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(name: str) -> DistFn:
    if name == "cosine":
        return cosine
    if name == "manhattan":
        return manhattan
    return euclidean


# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem) -> None:
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        r = [(dist(q, v.emb), v.id) for v in self.items]
        r.sort(key=lambda p: p[0])
        return r[:k]

    def remove(self, id_: int) -> None:
        self.items = [v for v in self.items if v.id != id_]


# =====================================================================
#  KD-TREE
# =====================================================================

class _KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem):
        self.item = item
        self.left: Optional["_KDNode"] = None
        self.right: Optional["_KDNode"] = None


class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root: Optional[_KDNode] = None

    def insert(self, v: VectorItem) -> None:
        self.root = self._ins(self.root, v, 0)

    def _ins(self, n: Optional[_KDNode], v: VectorItem, d: int) -> _KDNode:
        if n is None:
            return _KDNode(v)
        ax = d % self.dims
        if v.emb[ax] < n.item.emb[ax]:
            n.left = self._ins(n.left, v, d + 1)
        else:
            n.right = self._ins(n.right, v, d + 1)
        return n

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        # max-heap of size k, emulated with negated distances
        heap: List[Tuple[float, int]] = []
        self._knn(self.root, q, k, 0, dist, heap)
        res = [(-d, i) for d, i in heap]
        res.sort(key=lambda p: p[0])
        return res

    def _knn(self, n: Optional[_KDNode], q: List[float], k: int, d: int,
             dist: DistFn, heap: List[Tuple[float, int]]) -> None:
        if n is None:
            return
        dn = dist(q, n.item.emb)
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, n.item.id))
            if len(heap) > k:
                heapq.heappop(heap)
        ax = d % self.dims
        diff = q[ax] - n.item.emb[ax]
        closer, farther = (n.left, n.right) if diff < 0 else (n.right, n.left)
        self._knn(closer, q, k, d + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, d + 1, dist, heap)

    def rebuild(self, items: List[VectorItem]) -> None:
        self.root = None
        for v in items:
            self.insert(v)


# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class _HNSWNode:
    __slots__ = ("item", "max_lyr", "nbrs")

    def __init__(self, item: VectorItem, max_lyr: int):
        self.item = item
        self.max_lyr = max_lyr
        self.nbrs: List[List[int]] = [[] for _ in range(max_lyr + 1)]


class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200, seed: int = 42):
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m))
        self.rng = random.Random(seed)
        self.G: Dict[int, _HNSWNode] = {}
        self.top_layer = -1
        self.entry_pt = -1

    def _rand_level(self) -> int:
        u = self.rng.random()
        while u <= 0.0:
            u = self.rng.random()
        return int(math.floor(-math.log(u) * self.mL))

    def _search_layer(self, q: List[float], ep: int, ef: int, lyr: int,
                       dist: DistFn) -> List[Tuple[float, int]]:
        visited = {ep}
        d0 = dist(q, self.G[ep].item.emb)
        cands = [(d0, ep)]
        found = [(-d0, ep)]  # max-heap via negation

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            node = self.G.get(cid)
            if node is None or lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if nid in visited or nid not in self.G:
                    continue
                visited.add(nid)
                nd = dist(q, self.G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        res = [(-d, i) for d, i in found]
        res.sort(key=lambda p: p[0])
        return res

    @staticmethod
    def _select_nbrs(cands: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [cid for _, cid in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn) -> None:
        id_ = item.id
        lvl = self._rand_level()
        self.G[id_] = _HNSWNode(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt = id_
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            node = self.G[ep]
            if lc < len(node.nbrs):
                w = self._search_layer(item.emb, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            w = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(w, max_m)
            self.G[id_].nbrs[lc] = sel

            for nid in sel:
                nnode = self.G.get(nid)
                if nnode is None:
                    continue
                if len(nnode.nbrs) <= lc:
                    nnode.nbrs.extend([[] for _ in range(lc + 1 - len(nnode.nbrs))])
                conn = nnode.nbrs[lc]
                conn.append(id_)
                if len(conn) > max_m:
                    ds = [(dist(nnode.item.emb, self.G[c].item.emb), c)
                          for c in conn if c in self.G]
                    ds.sort(key=lambda p: p[0])
                    nnode.nbrs[lc] = [c for _, c in ds[:max_m]]
            if w:
                ep = w[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = id_

    def knn(self, q: List[float], k: int, ef: int, dist: DistFn) -> List[Tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            node = self.G[ep]
            if lc < len(node.nbrs):
                w = self._search_layer(q, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]
        w = self._search_layer(q, ep, max(ef, k), 0, dist)
        return w[:k]

    def remove(self, id_: int) -> None:
        if id_ not in self.G:
            return
        for nd in self.G.values():
            for layer in nd.nbrs:
                if id_ in layer:
                    layer.remove(id_)
        if self.entry_pt == id_:
            self.entry_pt = -1
            for nid in self.G:
                if nid != id_:
                    self.entry_pt = nid
                    break
        del self.G[id_]

    def get_info(self) -> dict:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []
        for id_, nd in self.G.items():
            nodes.append({"id": id_, "metadata": nd.item.metadata,
                          "category": nd.item.category, "maxLyr": nd.max_lyr})
            top = min(nd.max_lyr, max_l - 1)
            for lc in range(0, top + 1):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id_, "dst": nid, "lyr": lc})
        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
        }

    def size(self) -> int:
        return len(self.G)
