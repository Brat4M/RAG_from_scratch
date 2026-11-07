"""Unit tests for Vector Indexing and Approximate Nearest Neighbor (ANN) search.

Tests exact k-NN flat search, Inverted File Index (IVF), and
Hierarchical Navigable Small World (HNSW) graph prototypes.
"""

import heapq
import numpy as np
import pytest


def exact_knn_search(query: np.ndarray, dataset: np.ndarray, k: int = 5):
    """Computes exact Euclidean distances from query to all vectors in dataset."""
    diffs = dataset - query
    distances = np.linalg.norm(diffs, axis=1)
    top_indices = np.argsort(distances)[:k]
    return [(int(idx), float(distances[idx])) for idx in top_indices]


def compute_recall_at_k(retrieved_ids: list[int], ground_truth_ids: list[int]) -> float:
    """Computes Recall@k: fraction of true top-k neighbors retrieved by ANN."""
    if not ground_truth_ids:
        return 0.0
    k = len(ground_truth_ids)
    hits = len(set(retrieved_ids[:k]).intersection(set(ground_truth_ids)))
    return float(hits / k)


class EducationalIVFIndex:
    """Inverted File Index (IVF) using k-means clustering for Voronoi partitioning."""

    def __init__(self, n_clusters: int = 4, max_iters: int = 20, seed: int = 42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.seed = seed
        self.centroids = None
        self.posting_lists = {}
        self.vectors = None

    def fit(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype=np.float64)
        n_samples, _ = self.vectors.shape
        rng = np.random.RandomState(self.seed)

        init_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        self.centroids = self.vectors[init_indices].copy()

        assignments = np.zeros(n_samples, dtype=int)
        for _ in range(self.max_iters):
            dists = np.linalg.norm(
                self.vectors[:, None, :] - self.centroids[None, :, :], axis=2
            )
            new_assignments = np.argmin(dists, axis=1)
            if np.array_equal(assignments, new_assignments):
                break
            assignments = new_assignments
            for c in range(self.n_clusters):
                cluster_vecs = self.vectors[assignments == c]
                if len(cluster_vecs) > 0:
                    self.centroids[c] = np.mean(cluster_vecs, axis=0)

        self.posting_lists = {c: [] for c in range(self.n_clusters)}
        for vec_id, c in enumerate(assignments):
            self.posting_lists[c].append(vec_id)

    def search(self, query: np.ndarray, k: int = 5, n_probe: int = 1):
        if self.centroids is None or not self.posting_lists:
            return [], 0

        c_dists = np.linalg.norm(self.centroids - query, axis=1)
        probed_clusters = np.argsort(c_dists)[: min(n_probe, self.n_clusters)]

        candidates = []
        for c in probed_clusters:
            candidates.extend(self.posting_lists[c])

        if not candidates:
            return [], len(self.centroids)

        candidate_vecs = self.vectors[candidates]
        dists = np.linalg.norm(candidate_vecs - query, axis=1)
        dist_comps = len(self.centroids) + len(candidates)

        sorted_order = np.argsort(dists)[:k]
        results = [(int(candidates[idx]), float(dists[idx])) for idx in sorted_order]
        return results, dist_comps


class EducationalHNSWIndex:
    """Educational Hierarchical Navigable Small World (HNSW) index."""

    def __init__(self, dim: int, m: int = 6, ef_construction: int = 16, seed: int = 42):
        self.dim = dim
        self.m = m
        self.m0 = 2 * m
        self.ef_construction = ef_construction
        self.ml = 1.0 / np.log(m)
        self.rng = np.random.RandomState(seed)
        self.vectors = {}
        self.levels = {}
        self.graphs = []
        self.enter_point = None
        self.max_level = -1

    def _dist(self, u: np.ndarray, v: np.ndarray) -> float:
        return float(np.linalg.norm(u - v))

    def _get_random_level(self) -> int:
        r = self.rng.uniform(0.0001, 1.0)
        return int(-np.log(r) * self.ml)

    def insert(self, vec_id: int, vec: np.ndarray):
        vec = np.asarray(vec, dtype=np.float64)
        self.vectors[vec_id] = vec
        level = self._get_random_level()
        self.levels[vec_id] = level

        while len(self.graphs) <= max(level, self.max_level):
            self.graphs.append({})
        for l in range(level + 1):
            if vec_id not in self.graphs[l]:
                self.graphs[l][vec_id] = set()

        if self.enter_point is None:
            self.enter_point = vec_id
            self.max_level = level
            return

        curr_obj = self.enter_point
        # 1. Greedy routing in upper layers above node level
        for l in range(self.max_level, level, -1):
            changed = True
            while changed:
                changed = False
                curr_dist = self._dist(vec, self.vectors[curr_obj])
                for n_id in self.graphs[l].get(curr_obj, set()):
                    d = self._dist(vec, self.vectors[n_id])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = n_id
                        changed = True

        # 2. Search and connect from min(max_level, level) down to 0
        ep = [curr_obj]
        for l in range(min(self.max_level, level), -1, -1):
            v = set(ep)
            candidates = []
            results = []
            for node in ep:
                d = self._dist(vec, self.vectors[node])
                heapq.heappush(candidates, (d, node))
                heapq.heappush(results, (-d, node))

            while candidates:
                c_dist, c_id = heapq.heappop(candidates)
                furthest = -results[0][0]
                if c_dist > furthest:
                    break
                for n_id in self.graphs[l].get(c_id, set()):
                    if n_id not in v:
                        v.add(n_id)
                        n_dist = self._dist(vec, self.vectors[n_id])
                        furthest = -results[0][0]
                        if n_dist < furthest or len(results) < self.ef_construction:
                            heapq.heappush(candidates, (n_dist, n_id))
                            heapq.heappush(results, (-n_dist, n_id))
                            if len(results) > self.ef_construction:
                                heapq.heappop(results)

            ep = [item[1] for item in sorted([(-d, nid) for d, nid in results])]
            m_max = self.m0 if l == 0 else self.m
            neighbors = [n_id for n_id in ep[:m_max] if n_id != vec_id]
            self.graphs[l][vec_id] = set(neighbors)
            for n_id in neighbors:
                self.graphs[l][n_id].add(vec_id)
                if len(self.graphs[l][n_id]) > m_max:
                    n_nbrs = list(self.graphs[l][n_id])
                    n_dists = [
                        self._dist(self.vectors[n_id], self.vectors[other])
                        for other in n_nbrs
                    ]
                    sorted_nbrs = [x for _, x in sorted(zip(n_dists, n_nbrs))]
                    self.graphs[l][n_id] = set(sorted_nbrs[:m_max])

        if level > self.max_level:
            self.max_level = level
            self.enter_point = vec_id

    def search(self, query: np.ndarray, k: int = 5, ef_search: int = 16):
        if self.enter_point is None:
            return [], 0

        query = np.asarray(query, dtype=np.float64)
        dist_comps = 0
        dist_cache = {}

        def get_dist(node_id):
            nonlocal dist_comps
            if node_id not in dist_cache:
                dist_comps += 1
                dist_cache[node_id] = self._dist(query, self.vectors[node_id])
            return dist_cache[node_id]

        curr_obj = self.enter_point
        # 1. Greedy routing in upper layers
        for l in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                curr_dist = get_dist(curr_obj)
                for n_id in self.graphs[l].get(curr_obj, set()):
                    d = get_dist(n_id)
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = n_id
                        changed = True

        # 2. Layer 0 beam search
        ef = max(ef_search, k)
        v = {curr_obj}
        candidates = []
        results = []
        d0 = get_dist(curr_obj)
        heapq.heappush(candidates, (d0, curr_obj))
        heapq.heappush(results, (-d0, curr_obj))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest = -results[0][0]
            if c_dist > furthest:
                break
            for n_id in self.graphs[0].get(c_id, set()):
                if n_id not in v:
                    v.add(n_id)
                    n_dist = get_dist(n_id)
                    furthest = -results[0][0]
                    if n_dist < furthest or len(results) < ef:
                        heapq.heappush(candidates, (n_dist, n_id))
                        heapq.heappush(results, (-n_dist, n_id))
                        if len(results) > ef:
                            heapq.heappop(results)

        top_k = sorted([(n_id, -d) for d, n_id in results], key=lambda x: x[1])[:k]
        return top_k, dist_comps


# --- Tests ---


def test_exact_knn_search_returns_sorted_nearest():
    dataset = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [10.0, 10.0],
    ])
    query = np.array([0.9, 0.9])
    results = exact_knn_search(query, dataset, k=3)
    assert len(results) == 3
    # Nearest should be [1.0, 1.0] (idx 1), followed by [0.0, 0.0] (idx 0), then [2.0, 2.0] (idx 2)
    assert results[0][0] == 1
    assert results[1][0] == 0
    assert results[2][0] == 2
    # Verify monotonically increasing distances
    assert results[0][1] <= results[1][1] <= results[2][1]


def test_compute_recall_at_k():
    ground_truth = [1, 2, 3, 4, 5]
    # Perfect retrieval
    assert compute_recall_at_k([1, 2, 3, 4, 5], ground_truth) == 1.0
    # Partial retrieval (3 out of 5)
    assert compute_recall_at_k([1, 9, 3, 8, 5], ground_truth) == 0.6
    # Zero retrieval
    assert compute_recall_at_k([10, 11, 12, 13, 14], ground_truth) == 0.0
    # Empty ground truth
    assert compute_recall_at_k([1, 2], []) == 0.0


def test_ivf_index_construction_and_full_probe_recall():
    rng = np.random.RandomState(42)
    dataset = rng.randn(60, 4)
    ivf = EducationalIVFIndex(n_clusters=4, max_iters=20, seed=42)
    ivf.fit(dataset)

    # All vectors must be assigned to posting lists
    total_indexed = sum(len(posting) for posting in ivf.posting_lists.values())
    assert total_indexed == 60

    query = rng.randn(4)
    exact_res = exact_knn_search(query, dataset, k=5)
    exact_ids = [idx for idx, _ in exact_res]

    # When n_probe == n_clusters, all posting lists are scanned, giving 100% recall
    ivf_res, dist_comps = ivf.search(query, k=5, n_probe=4)
    ivf_ids = [idx for idx, _ in ivf_res]

    recall = compute_recall_at_k(ivf_ids, exact_ids)
    assert np.isclose(recall, 1.0)
    assert dist_comps == 4 + 60


def test_ivf_nprobe_tradeoff():
    rng = np.random.RandomState(42)
    dataset = rng.randn(100, 8)
    ivf = EducationalIVFIndex(n_clusters=10, max_iters=20, seed=42)
    ivf.fit(dataset)

    query = rng.randn(8)
    _, dist_p1 = ivf.search(query, k=5, n_probe=1)
    _, dist_p4 = ivf.search(query, k=5, n_probe=4)
    _, dist_p10 = ivf.search(query, k=5, n_probe=10)

    # Increasing n_probe scans more posting lists
    assert dist_p1 < dist_p4 < dist_p10


def test_hnsw_geometric_layer_distribution():
    dim = 8
    hnsw = EducationalHNSWIndex(dim=dim, m=6, ef_construction=16, seed=42)
    rng = np.random.RandomState(42)
    n_nodes = 150
    for i in range(n_nodes):
        hnsw.insert(i, rng.randn(dim))

    assert hnsw.max_level >= 1
    # Layer 0 must contain all nodes
    assert len(hnsw.graphs[0]) == n_nodes
    # Upper layers must have fewer nodes (exponential decay)
    for l in range(hnsw.max_level):
        assert len(hnsw.graphs[l]) >= len(hnsw.graphs[l + 1])


def test_hnsw_search_reaches_high_recall():
    dim = 16
    hnsw = EducationalHNSWIndex(dim=dim, m=6, ef_construction=24, seed=42)
    rng = np.random.RandomState(42)
    dataset = rng.randn(100, dim)
    for i in range(100):
        hnsw.insert(i, dataset[i])

    query = rng.randn(dim)
    exact_res = exact_knn_search(query, dataset, k=5)
    exact_ids = [idx for idx, _ in exact_res]

    # Search with ef_search=32
    hnsw_res, comps = hnsw.search(query, k=5, ef_search=32)
    hnsw_ids = [idx for idx, _ in hnsw_res]

    recall = compute_recall_at_k(hnsw_ids, exact_ids)
    assert recall >= 0.80
    assert comps < 100  # Evaluates fewer distances than exact search


def test_hnsw_empty_and_single_element_edge_cases():
    hnsw = EducationalHNSWIndex(dim=4, m=4, ef_construction=8, seed=42)
    # Search on empty index
    res, comps = hnsw.search(np.zeros(4), k=3)
    assert res == []
    assert comps == 0

    # Insert single element
    vec0 = np.array([1.0, 2.0, 3.0, 4.0])
    hnsw.insert(0, vec0)
    res, comps = hnsw.search(vec0, k=3)
    assert len(res) == 1
    assert res[0][0] == 0
    assert np.isclose(res[0][1], 0.0)
