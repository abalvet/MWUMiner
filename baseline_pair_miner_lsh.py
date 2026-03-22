#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import multiprocessing as mp
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class PairResult:
    pair_id: str
    i: int
    j: int
    segment_1: str
    segment_2: str
    candidate_strategy: str
    mode: str
    rel_lev: Optional[float]
    lev_dist: Optional[int]
    est_jaccard: Optional[float]
    exact_jaccard: Optional[float]
    shared_bands: Optional[int]
    len_1_units: int
    len_2_units: int
    len_1_chars: int
    len_2_chars: int
    length_ratio: float
    xx: str
    yy: str
    lcs_stable: str
    lcs_moved: str
    lcs_sequence: str
    lcs_length: int
    lcs_cov_seg1: float
    lcs_cov_seg2: float
    lcs_cov_mean: float
    lcs_cov_min: float
    same_position_hits: int
    moved_hits: int
    score_similarity: Optional[float]
    score_lev_sim: Optional[float]
    score_lcs_mean_minus_distance: Optional[float]


@dataclass(frozen=True)
class LCSView:
    xx: str
    yy: str
    lcs_stable: str
    lcs_moved: str
    lcs_sequence: str
    lcs_length: int
    same_position_hits: int
    moved_hits: int
    len_1_units: int
    len_2_units: int


def read_segments(path: Path, encoding: str = "utf-8") -> List[str]:
    return path.read_text(encoding=encoding, errors="replace").splitlines()


def prepare_segments(lines: Iterable[str], dedup: bool = True) -> List[str]:
    stripped = [line.strip() for line in lines if line.strip()]
    return sorted(set(stripped) if dedup else stripped)


def tokenize(text: str, mode: str = "token", token_pattern: str = r"\S+") -> List[str]:
    if mode == "char":
        return list(text)
    if mode != "token":
        raise ValueError(f"Unsupported mode: {mode!r}")
    return re.findall(token_pattern, text)


def make_shingles_from_units(units: Sequence[str], shingle_size: int) -> Set[str]:
    if not units:
        return set()
    if shingle_size <= 1:
        return set(units)
    if len(units) < shingle_size:
        return {" ".join(units)}
    return {" ".join(units[i:i + shingle_size]) for i in range(len(units) - shingle_size + 1)}


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:d}h{m:02d}m{s:05.2f}s"
    if m > 0:
        return f"{m:d}m{s:05.2f}s"
    return f"{s:.2f}s"


def progress_bar(current: int, total: int, width: int = 32) -> str:
    total = max(1, total)
    ratio = min(1.0, max(0.0, current / total))
    filled = int(width * ratio)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {current}/{total} ({ratio * 100:5.1f}%)"


def print_progress(label: str, current: int, total: int, start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    rate = current / elapsed if elapsed > 0 else 0.0
    remaining = ((total - current) / rate) if rate > 0 else 0.0
    print(
        f"\r[progress] {label}: {progress_bar(current, total)} | elapsed={format_seconds(elapsed)} | eta={format_seconds(remaining)}",
        file=sys.stderr,
        end="",
        flush=True,
    )


def relative_levenshtein_bounded(a: str, b: str, rel_threshold: float) -> Tuple[Optional[int], Optional[float]]:
    if a == b:
        return 0, 0.0
    la, lb = len(a), len(b)
    max_len = max(la, lb)
    if max_len == 0:
        return 0, 0.0
    max_dist = int(math.floor(rel_threshold * max_len))
    if max_dist <= 0:
        return None, None
    if abs(la - lb) > max_dist:
        return None, None
    if lb > la:
        a, b = b, a
        la, lb = lb, la
    if la - lb > max_dist:
        return None, None

    inf = max_dist + 1
    prev = [inf] * (lb + 1)
    curr = [inf] * (lb + 1)
    for j in range(min(lb, max_dist) + 1):
        prev[j] = j

    for i in range(1, la + 1):
        lower = max(1, i - max_dist)
        upper = min(lb, i + max_dist)
        for j in range(lb + 1):
            curr[j] = inf
        if lower == 1:
            curr[0] = i
        row_min = inf
        ai = a[i - 1]
        for j in range(lower, upper + 1):
            cost = 0 if ai == b[j - 1] else 1
            deletion = prev[j] + 1
            insertion = curr[j - 1] + 1
            substitution = prev[j - 1] + cost
            v = min(deletion, insertion, substitution)
            curr[j] = v
            row_min = min(row_min, v)
        if row_min > max_dist:
            return None, None
        prev, curr = curr, prev

    dist = prev[lb]
    if dist > max_dist:
        return None, None
    return dist, dist / max(la, lb)


def _join_units(units: Sequence[str], mode: str) -> str:
    return "".join(units) if mode == "char" else " ".join(units)


def lcs_views(seq1: Sequence[str], seq2: Sequence[str], mode: str = "token") -> LCSView:
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return LCSView(
            xx=_join_units(["-"] * m, mode),
            yy=_join_units(["-"] * n, mode),
            lcs_stable="",
            lcs_moved="",
            lcs_sequence="",
            lcs_length=0,
            same_position_hits=0,
            moved_hits=0,
            len_1_units=m,
            len_2_units=n,
        )

    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        row_i = L[i]
        row_i1 = L[i + 1]
        s1 = seq1[i]
        for j in range(n):
            if s1 == seq2[j]:
                row_i1[j + 1] = row_i[j] + 1
            else:
                row_i1[j + 1] = max(row_i1[j], row_i[j + 1])

    i, j = m, n
    matches = []
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            matches.append((i - 1, j - 1, seq1[i - 1]))
            i -= 1
            j -= 1
        elif L[i - 1][j] >= L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    matches.reverse()

    xx_units = ["-"] * m
    yy_units = ["-"] * n
    stable_units = []
    moved_units = []
    lcs_units = []
    same_position_hits = 0
    moved_hits = 0
    for i1, j1, unit in matches:
        xx_units[i1] = unit
        yy_units[j1] = unit
        lcs_units.append(unit)
        if i1 == j1:
            stable_units.append(unit)
            same_position_hits += 1
        else:
            moved_units.append(unit)
            moved_hits += 1

    return LCSView(
        xx=_join_units(xx_units, mode),
        yy=_join_units(yy_units, mode),
        lcs_stable=_join_units(stable_units, mode),
        lcs_moved=_join_units(moved_units, mode),
        lcs_sequence=_join_units(lcs_units, mode),
        lcs_length=len(matches),
        same_position_hits=same_position_hits,
        moved_hits=moved_hits,
        len_1_units=m,
        len_2_units=n,
    )


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return (len(a & b) / len(union)) if union else 1.0


def stable_u64(text: str, salt: int) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(f"{salt}\t{text}".encode("utf-8", errors="replace"))
    return int.from_bytes(h.digest(), "big", signed=False)


def minhash_signature(shingles: Set[str], num_perm: int) -> List[int]:
    if not shingles:
        return [0] * num_perm
    return [min(stable_u64(sh, k) for sh in shingles) for k in range(num_perm)]


def auto_bands(num_perm: int) -> int:
    for rows in (4, 5, 6, 8):
        if num_perm % rows == 0:
            return num_perm // rows
    for bands in range(int(math.sqrt(num_perm)), 0, -1):
        if num_perm % bands == 0:
            return bands
    return 1


def lsh_jaccard_threshold(num_perm: int, bands: int) -> float:
    """Expected Jaccard similarity threshold for the (bands, rows) configuration.

    Derived from the band trick: t ≈ (1/bands)^(1/rows), where rows = num_perm // bands.
    Pairs whose true Jaccard exceeds this value have a high probability of sharing at least
    one LSH band and therefore of becoming candidates.
    """
    rows = num_perm // bands
    return (1.0 / bands) ** (1.0 / rows)


def build_result_from_units(
    i: int,
    j: int,
    s1: str,
    s2: str,
    units1: Sequence[str],
    units2: Sequence[str],
    mode: str,
    candidate_strategy: str,
    rel_lev: Optional[float],
    lev_dist: Optional[int],
    est_jaccard: Optional[float],
    exact_jaccard: Optional[float],
    shared_bands: Optional[int],
) -> PairResult:
    lcs = lcs_views(units1, units2, mode=mode)
    denom_1 = len(units1) or 1
    denom_2 = len(units2) or 1
    cov1 = lcs.lcs_length / denom_1
    cov2 = lcs.lcs_length / denom_2
    cov_mean = (cov1 + cov2) / 2.0
    cov_min = min(cov1, cov2)
    max_units = max(len(units1), len(units2))
    length_ratio = (min(len(units1), len(units2)) / max_units) if max_units else 1.0

    sim_candidates = [x for x in (est_jaccard, exact_jaccard) if x is not None]
    score_similarity = max(sim_candidates) if sim_candidates else None
    score_lev_sim = (1.0 - rel_lev) if rel_lev is not None else None
    score_lcs_mean_minus_distance = (cov_mean - rel_lev) if rel_lev is not None else None

    return PairResult(
        pair_id=f"{i}-{j}",
        i=i,
        j=j,
        segment_1=s1,
        segment_2=s2,
        candidate_strategy=candidate_strategy,
        mode=mode,
        rel_lev=rel_lev,
        lev_dist=lev_dist,
        est_jaccard=est_jaccard,
        exact_jaccard=exact_jaccard,
        shared_bands=shared_bands,
        len_1_units=len(units1),
        len_2_units=len(units2),
        len_1_chars=len(s1),
        len_2_chars=len(s2),
        length_ratio=length_ratio,
        xx=lcs.xx,
        yy=lcs.yy,
        lcs_stable=lcs.lcs_stable,
        lcs_moved=lcs.lcs_moved,
        lcs_sequence=lcs.lcs_sequence,
        lcs_length=lcs.lcs_length,
        lcs_cov_seg1=cov1,
        lcs_cov_seg2=cov2,
        lcs_cov_mean=cov_mean,
        lcs_cov_min=cov_min,
        same_position_hits=lcs.same_position_hits,
        moved_hits=lcs.moved_hits,
        score_similarity=score_similarity,
        score_lev_sim=score_lev_sim,
        score_lcs_mean_minus_distance=score_lcs_mean_minus_distance,
    )


_WORKER_SEGMENTS: List[str] = []
_WORKER_THRESHOLD: float = 0.0
_WORKER_MODE: str = "token"
_WORKER_TOKEN_PATTERN: str = r"\S+"


def _init_worker(segments: List[str], threshold: float, mode: str, token_pattern: str) -> None:
    global _WORKER_SEGMENTS, _WORKER_THRESHOLD, _WORKER_MODE, _WORKER_TOKEN_PATTERN
    _WORKER_SEGMENTS = segments
    _WORKER_THRESHOLD = threshold
    _WORKER_MODE = mode
    _WORKER_TOKEN_PATTERN = token_pattern


def _process_range_exhaustive(task: Tuple[int, int]) -> List[PairResult]:
    start_i, end_i = task
    segs = _WORKER_SEGMENTS
    threshold = _WORKER_THRESHOLD
    mode = _WORKER_MODE
    token_pattern = _WORKER_TOKEN_PATTERN
    n = len(segs)
    out: List[PairResult] = []
    for i in range(start_i, min(end_i, n)):
        s1 = segs[i]
        units1 = tokenize(s1, mode=mode, token_pattern=token_pattern)
        len1 = len(s1)
        for j in range(i + 1, n):
            s2 = segs[j]
            max_len = max(len1, len(s2))
            if max_len == 0:
                continue
            if abs(len1 - len(s2)) / max_len >= threshold:
                continue
            lev_dist, rel_lev = relative_levenshtein_bounded(s1, s2, threshold)
            if lev_dist is None or rel_lev is None or rel_lev <= 0.0 or rel_lev >= threshold:
                continue
            units2 = tokenize(s2, mode=mode, token_pattern=token_pattern)
            out.append(build_result_from_units(i, j, s1, s2, units1, units2, mode, "exhaustive", rel_lev, lev_dist, None, None, None))
    return out


def make_index_ranges(n: int, workers: int, min_chunk: int = 32) -> List[Tuple[int, int]]:
    if n <= 0:
        return []
    if workers <= 1:
        return [(0, n)]
    chunk = max(min_chunk, math.ceil(n / workers))
    return [(start, min(n, start + chunk)) for start in range(0, n, chunk)]


def mine_pairs_exhaustive(
    segments: Sequence[str],
    rel_threshold: float,
    mode: str = "token",
    token_pattern: str = r"\S+",
    workers: int = 1,
    show_progress: bool = True,
) -> List[PairResult]:
    start = time.perf_counter()
    if workers <= 1:
        _init_worker(list(segments), rel_threshold, mode, token_pattern)
        results = _process_range_exhaustive((0, len(segments)))
        if show_progress:
            print_progress("pair mining", 1, 1, start)
            print(file=sys.stderr)
        return results

    ranges = make_index_ranges(len(segments), workers=workers)
    with mp.Pool(processes=workers, initializer=_init_worker, initargs=(list(segments), rel_threshold, mode, token_pattern)) as pool:
        out: List[PairResult] = []
        total = len(ranges)
        for done, part in enumerate(pool.imap_unordered(_process_range_exhaustive, ranges), start=1):
            out.extend(part)
            if show_progress:
                print_progress("pair mining", done, total, start)
        if show_progress:
            print(file=sys.stderr)
    return out


def build_shingle_sets(segments: Sequence[str], mode: str, token_pattern: str, shingle_size: int, show_progress: bool = True) -> List[Set[str]]:
    start = time.perf_counter()
    out = []
    total = len(segments)
    step = max(1, total // 100)
    for idx, segment in enumerate(segments, start=1):
        out.append(make_shingles_from_units(tokenize(segment, mode=mode, token_pattern=token_pattern), shingle_size))
        if show_progress and (idx == total or idx % step == 0):
            print_progress("shingling", idx, total, start)
    if show_progress:
        print(file=sys.stderr)
    return out


def build_signatures(shingle_sets: Sequence[Set[str]], num_perm: int, show_progress: bool = True) -> List[List[int]]:
    start = time.perf_counter()
    out = []
    total = len(shingle_sets)
    step = max(1, total // 100)
    for idx, shingles in enumerate(shingle_sets, start=1):
        out.append(minhash_signature(shingles, num_perm))
        if show_progress and (idx == total or idx % step == 0):
            print_progress("minhash signatures", idx, total, start)
    if show_progress:
        print(file=sys.stderr)
    return out


def lsh_candidates(signatures: Sequence[Sequence[int]], bands: int, show_progress: bool = True) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    if not signatures:
        return set(), {}
    num_perm = len(signatures[0])
    if num_perm % bands != 0:
        raise ValueError("num_perm must be divisible by bands")
    rows = num_perm // bands

    start = time.perf_counter()
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
    total_segments = len(signatures)
    step = max(1, total_segments // 100)
    for idx, sig in enumerate(signatures, start=1):
        for band_id in range(bands):
            lo = band_id * rows
            key = (band_id, tuple(sig[lo:lo + rows]))
            buckets.setdefault(key, []).append(idx - 1)
        if show_progress and (idx == total_segments or idx % step == 0):
            print_progress("lsh bucketing", idx, total_segments, start)
    if show_progress:
        print(file=sys.stderr)

    cand_start = time.perf_counter()
    pair_counts: Dict[Tuple[int, int], int] = {}
    items = list(buckets.items())
    total_buckets = len(items)
    step_b = max(1, total_buckets // 100)
    for idx, (_, members) in enumerate(items, start=1):
        if len(members) > 1:
            for i, j in combinations(sorted(members), 2):
                pair_counts[(i, j)] = pair_counts.get((i, j), 0) + 1
        if show_progress and (idx == total_buckets or idx % step_b == 0):
            print_progress("candidate pairs", idx, total_buckets, cand_start)
    if show_progress:
        print(file=sys.stderr)

    return set(pair_counts.keys()), pair_counts


def estimate_jaccard_from_signatures(sig1: Sequence[int], sig2: Sequence[int]) -> float:
    return sum(1 for a, b in zip(sig1, sig2) if a == b) / min(len(sig1), len(sig2))


def score_lsh_candidates(
    segments: Sequence[str],
    shingle_sets: Sequence[Set[str]],
    signatures: Sequence[Sequence[int]],
    candidates: Sequence[Tuple[int, int]],
    pair_band_counts: Dict[Tuple[int, int], int],
    mode: str,
    token_pattern: str,
    min_est_jaccard: float,
    min_exact_jaccard: Optional[float],
    show_progress: bool = True,
) -> List[PairResult]:
    start = time.perf_counter()
    out: List[PairResult] = []
    total = len(candidates)
    step = max(1, total // 100) if total else 1

    for idx, (i, j) in enumerate(candidates, start=1):
        est = estimate_jaccard_from_signatures(signatures[i], signatures[j])
        if est >= min_est_jaccard:
            exact = jaccard_similarity(shingle_sets[i], shingle_sets[j])
            if min_exact_jaccard is None or exact >= min_exact_jaccard:
                s1 = segments[i]
                s2 = segments[j]
                units1 = tokenize(s1, mode=mode, token_pattern=token_pattern)
                units2 = tokenize(s2, mode=mode, token_pattern=token_pattern)
                lev_dist, rel_lev = relative_levenshtein_bounded(s1, s2, 1.0)
                out.append(build_result_from_units(i, j, s1, s2, units1, units2, mode, "lsh", rel_lev, lev_dist, est, exact, pair_band_counts.get((i, j))))
        if show_progress and total and (idx == total or idx % step == 0):
            print_progress("scoring lsh candidates", idx, total, start)
    if show_progress:
        print(file=sys.stderr)
    return out


def mine_pairs_lsh(
    segments: Sequence[str],
    mode: str = "token",
    token_pattern: str = r"\S+",
    shingle_size: int = 2,
    num_perm: int = 64,
    bands: Optional[int] = None,
    min_est_jaccard: float = 0.3,
    min_exact_jaccard: Optional[float] = None,
    show_progress: bool = True,
) -> List[PairResult]:
    if num_perm <= 0:
        raise ValueError("num_perm must be > 0")
    if bands is None:
        bands = auto_bands(num_perm)
    if num_perm % bands != 0:
        raise ValueError("num_perm must be divisible by bands")
    shingle_sets = build_shingle_sets(segments, mode, token_pattern, shingle_size, show_progress)
    signatures = build_signatures(shingle_sets, num_perm, show_progress)
    candidates, pair_band_counts = lsh_candidates(signatures, bands, show_progress)
    print(f"[info] LSH candidate pairs: {len(candidates)}", file=sys.stderr)
    return score_lsh_candidates(
        segments,
        shingle_sets,
        signatures,
        sorted(candidates),
        pair_band_counts,
        mode,
        token_pattern,
        min_est_jaccard,
        min_exact_jaccard,
        show_progress,
    )


DEFAULT_COLUMNS = [
    "pair_id", "i", "j", "segment_1", "segment_2", "candidate_strategy", "mode",
    "rel_lev", "lev_dist", "est_jaccard", "exact_jaccard", "shared_bands",
    "len_1_units", "len_2_units", "len_1_chars", "len_2_chars", "length_ratio",
    "xx", "yy", "lcs_stable", "lcs_moved", "lcs_sequence", "lcs_length",
    "lcs_cov_seg1", "lcs_cov_seg2", "lcs_cov_mean", "lcs_cov_min",
    "same_position_hits", "moved_hits",
    "score_similarity", "score_lev_sim", "score_lcs_mean_minus_distance",
]


def sort_results(results: List[PairResult], sort_by: str = "score_similarity") -> List[PairResult]:
    descending = {
        "est_jaccard", "exact_jaccard", "shared_bands", "lcs_length", "lcs_cov_seg1",
        "lcs_cov_seg2", "lcs_cov_mean", "lcs_cov_min", "same_position_hits",
        "moved_hits", "score_similarity", "score_lev_sim", "score_lcs_mean_minus_distance", "length_ratio",
    }
    reverse = sort_by in descending
    return sorted(results, key=lambda r: (getattr(r, sort_by) is None, getattr(r, sort_by)), reverse=reverse)


def write_tsv(results: Sequence[PairResult], output_path: Path, columns: Sequence[str] = DEFAULT_COLUMNS) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extraction of similar segment pairs + LCS patterns (exhaustive or LSH).")
    p.add_argument("input", type=Path, help="Text file with one segment per line.")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output TSV path.")
    p.add_argument("--candidate-strategy", choices=["exhaustive", "lsh"], default="exhaustive", help="How to identify candidate pairs.")
    p.add_argument("-t", "--threshold", type=float, default=0.4, help="Maximum relative Levenshtein distance for exhaustive strategy.")
    p.add_argument("-m", "--mode", choices=["token", "char"], default="token", help="LCS mode and shingling unit mode.")
    p.add_argument("--token-pattern", default=r"\S+", help=r"Regex used to extract tokens in token mode.")
    p.add_argument("--encoding", default="utf-8", help="Input encoding.")
    p.add_argument("-w", "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Worker processes for exhaustive strategy.")
    p.add_argument("--sort-by", default="lcs_cov_mean", choices=DEFAULT_COLUMNS, help="Column used to sort results before export.")
    p.add_argument("--limit", type=int, default=0, help="Optional cap on the number of rows written.")
    p.add_argument("--min-lcs-cov", type=float, default=0.0, help="Minimum lcs_cov_mean to keep a pair (0 = disabled).")
    p.add_argument("--no-dedup", action="store_true", help="Disable input deduplication (keep duplicate segments).")
    p.add_argument("--shingle-size", type=int, default=2, help="Shingle size for LSH.")
    p.add_argument("--num-perm", type=int, default=64, help="Number of MinHash permutations.")
    p.add_argument("--bands", type=int, default=0, help="Number of LSH bands. 0 = automatic choice.")
    p.add_argument("--min-est-jaccard", type=float, default=0.3, help="Minimum estimated Jaccard in LSH mode.")
    p.add_argument("--min-exact-jaccard", type=float, default=-1.0, help="Optional exact Jaccard threshold in LSH mode. -1 disables it.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.candidate_strategy == "exhaustive" and not (0.0 < args.threshold < 1.0):
        parser.error("--threshold must be in ]0,1[ for exhaustive strategy.")

    raw_lines = read_segments(args.input, encoding=args.encoding)
    segments = prepare_segments(raw_lines, dedup=not args.no_dedup)

    print(f"[info] input lines: {len(raw_lines)}", file=sys.stderr)
    print(f"[info] deduplication: {'disabled' if args.no_dedup else 'enabled'}", file=sys.stderr)
    print(f"[info] unique non-empty segments: {len(segments)}", file=sys.stderr)
    if len(segments) >= 2:
        print(f"[info] theoretical candidate pairs: {len(segments) * (len(segments) - 1) // 2}", file=sys.stderr)
    print(f"[info] candidate strategy: {args.candidate_strategy}", file=sys.stderr)
    print(f"[info] mode: {args.mode}", file=sys.stderr)

    total_start = time.perf_counter()
    mining_start = time.perf_counter()

    if args.candidate_strategy == "exhaustive":
        print(f"[info] workers: {args.workers}", file=sys.stderr)
        results = mine_pairs_exhaustive(
            segments=segments,
            rel_threshold=args.threshold,
            mode=args.mode,
            token_pattern=args.token_pattern,
            workers=args.workers,
            show_progress=True,
        )
    else:
        bands = args.bands if args.bands > 0 else auto_bands(args.num_perm)
        min_exact = None if args.min_exact_jaccard < 0 else args.min_exact_jaccard
        print(f"[info] shingle size: {args.shingle_size}", file=sys.stderr)
        print(f"[info] num_perm: {args.num_perm}", file=sys.stderr)
        print(f"[info] bands: {bands}", file=sys.stderr)
        print(f"[info] effective Jaccard threshold ≈ {lsh_jaccard_threshold(args.num_perm, bands):.3f}", file=sys.stderr)
        print(f"[info] min_est_jaccard: {args.min_est_jaccard}", file=sys.stderr)
        if min_exact is not None:
            print(f"[info] min_exact_jaccard: {min_exact}", file=sys.stderr)
        results = mine_pairs_lsh(
            segments=segments,
            mode=args.mode,
            token_pattern=args.token_pattern,
            shingle_size=args.shingle_size,
            num_perm=args.num_perm,
            bands=bands,
            min_est_jaccard=args.min_est_jaccard,
            min_exact_jaccard=min_exact,
            show_progress=True,
        )

    print(f"[time] pair mining: {format_seconds(time.perf_counter() - mining_start)}", file=sys.stderr)

    if args.min_lcs_cov > 0.0:
        before = len(results)
        results = [r for r in results if r.lcs_cov_mean >= args.min_lcs_cov]
        print(f"[info] lcs_cov_mean filter (>= {args.min_lcs_cov}): {before} → {len(results)} pairs", file=sys.stderr)

    sort_start = time.perf_counter()
    results = sort_results(results, sort_by=args.sort_by)
    print(f"[time] sorting: {format_seconds(time.perf_counter() - sort_start)}", file=sys.stderr)
    if args.limit > 0:
        results = results[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_start = time.perf_counter()
    write_tsv(results, args.output)
    print(f"[time] writing TSV: {format_seconds(time.perf_counter() - write_start)}", file=sys.stderr)
    print(f"[time] total: {format_seconds(time.perf_counter() - total_start)}", file=sys.stderr)
    print(f"[info] retained pairs: {len(results)}", file=sys.stderr)
    print(f"[info] wrote: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
