#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
UNIT_SEP = "\u241F"  # visible internal separator unlikely to appear in corpora
NUM_RE = re.compile(r"^[+-]?(?:\d+[\d.,:/-]*|\d*[.,]\d+)$")
@dataclass(frozen=True)
class PairResult:
    pair_id: str
    i: int
    j: int
    segment_id_1: str
    segment_id_2: str
    segment_1: str
    segment_2: str
    repr_1: str
    repr_2: str
    annotation_level: str
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
    intersection: str
    lcs_sequence: str
    lcs_length: int
    lcs_cov_seg1: float
    lcs_cov_seg2: float
    lcs_cov_mean: float
    lcs_cov_min: float
    same_position_hits: int
    moved_hits: int
    score_similarity: float
    score_lcs_mean_minus_distance: Optional[float]
@dataclass(frozen=True)
class LCSView:
    xx: str
    yy: str
    intersection: str
    lcs_sequence: str
    lcs_length: int
    same_position_hits: int
    moved_hits: int
    len_1_units: int
    len_2_units: int
@dataclass(frozen=True)
class SegmentRecord:
    idx: int
    seg_id: str
    raw_text: str
    units: Tuple[str, ...]
    repr_text: str
    meta: Dict[str, str]
@dataclass(frozen=True)
class UDToken:
    id_: str
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: str
    deprel: str
    deps: str
    misc: str
@dataclass(frozen=True)
class UDSentence:
    sent_id: str
    text: str
    tokens: Tuple[UDToken, ...]
    meta: Dict[str, str]
# -------------------------
# Parsing / normalization
# -------------------------
def normalize_surface_token(tok: str, lowercase: bool = False, normalize_num: bool = False) -> str:
    out = tok
    if lowercase:
        out = out.lower()
    if normalize_num and NUM_RE.match(out):
        out = "<NUM>"
    return out
def is_real_ud_token(token_id: str) -> bool:
    return "." not in token_id and "-" not in token_id
def read_plain_segments(path: Path, encoding: str = "utf-8") -> List[Tuple[str, Dict[str, str]]]:
    lines = path.read_text(encoding=encoding, errors="replace").splitlines()
    out = []
    for k, line in enumerate(lines, start=1):
        line = line.strip()
        if line:
            out.append((line, {"line_no": str(k)}))
    return out
def parse_conllu(path: Path, encoding: str = "utf-8") -> List[UDSentence]:
    text = path.read_text(encoding=encoding, errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip(), flags=re.MULTILINE)
    sentences: List[UDSentence] = []
    sent_counter = 0
    for block in blocks:
        if not block.strip():
            continue
        meta: Dict[str, str] = {}
        toks: List[UDToken] = []
        for line in block.splitlines():
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                m = re.match(r"#\s*([^=]+?)\s*=\s*(.*)", line)
                if m:
                    meta[m.group(1).strip()] = m.group(2).strip()
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue
            tok = UDToken(*cols)
            if is_real_ud_token(tok.id_):
                toks.append(tok)
        if not toks:
            continue
        sent_counter += 1
        sent_id = meta.get("sent_id", str(sent_counter))
        sent_text = meta.get("text") or " ".join(t.form for t in toks)
        sentences.append(UDSentence(sent_id=sent_id, text=sent_text, tokens=tuple(toks), meta=meta))
    return sentences
def detect_input_format(path: Path, encoding: str = "utf-8") -> str:
    sample = path.read_text(encoding=encoding, errors="replace")[:4000]
    if "\t" in sample and re.search(r"(?m)^\d+(?:[-.]\d+)?\t", sample):
        return "conllu"
    return "plain"
def build_units_from_ud_sentence(
    sent: UDSentence,
    annotation_level: str,
    lowercase: bool = False,
    normalize_num: bool = False,
) -> Tuple[str, ...]:
    units: List[str] = []
    for tok in sent.tokens:
        surf = normalize_surface_token(tok.form, lowercase=lowercase, normalize_num=normalize_num)
        lemma = normalize_surface_token(tok.lemma if tok.lemma != "_" else tok.form, lowercase=lowercase, normalize_num=normalize_num)
        upos = tok.upos if tok.upos != "_" else "X"
        if annotation_level == "token":
            units.append(surf)
        elif annotation_level == "lemma":
            units.append(lemma)
        elif annotation_level == "lemma_upos":
            units.append(f"{lemma}|{upos}")
        elif annotation_level == "token_lemma":
            units.append(f"{surf}|{lemma}")
        elif annotation_level == "token_lemma_upos":
            units.append(f"{surf}|{lemma}|{upos}")
        else:
            raise ValueError(f"Unsupported annotation level for UD: {annotation_level!r}")
    return tuple(units)
def build_units_from_plain_segment(
    text: str,
    annotation_level: str,
    token_pattern: str,
    lowercase: bool = False,
    normalize_num: bool = False,
) -> Tuple[str, ...]:
    if annotation_level != "token":
        raise ValueError("Without UD annotations, only annotation_level=token is supported.")
    toks = re.findall(token_pattern, text)
    return tuple(normalize_surface_token(tok, lowercase=lowercase, normalize_num=normalize_num) for tok in toks)
def load_segments(
    path: Path,
    input_format: str,
    annotation_level: str,
    token_pattern: str,
    encoding: str = "utf-8",
    lowercase: bool = False,
    normalize_num: bool = False,
    deduplicate: bool = True,
) -> List[SegmentRecord]:
    if input_format == "auto":
        input_format = detect_input_format(path, encoding=encoding)
    segments: List[SegmentRecord] = []
    seen: Set[str] = set()
    if input_format == "plain":
        raw_segments = read_plain_segments(path, encoding=encoding)
        for idx, (line, meta) in enumerate(raw_segments):
            units = build_units_from_plain_segment(
                line,
                annotation_level=annotation_level,
                token_pattern=token_pattern,
                lowercase=lowercase,
                normalize_num=normalize_num,
            )
            if not units:
                continue
            repr_text = UNIT_SEP.join(units)
            if deduplicate and repr_text in seen:
                continue
            seen.add(repr_text)
            segments.append(SegmentRecord(idx=len(segments), seg_id=meta.get("line_no", str(idx + 1)), raw_text=line, units=units, repr_text=repr_text, meta=meta))
    elif input_format == "conllu":
        sentences = parse_conllu(path, encoding=encoding)
        for idx, sent in enumerate(sentences):
            units = build_units_from_ud_sentence(
                sent,
                annotation_level=annotation_level,
                lowercase=lowercase,
                normalize_num=normalize_num,
            )
            if not units:
                continue
            repr_text = UNIT_SEP.join(units)
            if deduplicate and repr_text in seen:
                continue
            seen.add(repr_text)
            segments.append(SegmentRecord(idx=len(segments), seg_id=sent.sent_id, raw_text=sent.text, units=units, repr_text=repr_text, meta=sent.meta))
    else:
        raise ValueError(f"Unsupported input format: {input_format!r}")
    return segments
# -------------------------
# Utilities
# -------------------------
def make_shingles_from_units(units: Sequence[str], shingle_size: int) -> Set[str]:
    if not units:
        return set()
    if shingle_size <= 1:
        return set(units)
    if len(units) < shingle_size:
        return {UNIT_SEP.join(units)}
    return {UNIT_SEP.join(units[i:i + shingle_size]) for i in range(len(units) - shingle_size + 1)}
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
def relative_levenshtein_bounded_units(a: Sequence[str], b: Sequence[str], rel_threshold: float) -> Tuple[Optional[int], Optional[float]]:
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
    if mode == "char":
        return "".join(units)
    return " ".join(units)
def lcs_views(seq1: Sequence[str], seq2: Sequence[str], mode: str = "token") -> LCSView:
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return LCSView(
            xx=_join_units(["-"] * m, mode),
            yy=_join_units(["-"] * n, mode),
            intersection="",
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
    inter_units = []
    lcs_units = []
    same_position_hits = 0
    moved_hits = 0
    for i1, j1, unit in matches:
        xx_units[i1] = unit
        yy_units[j1] = unit
        lcs_units.append(unit)
        if i1 == j1:
            inter_units.append(unit)
            same_position_hits += 1
        else:
            inter_units.append(f'<MOV UNIT="{unit}" ORIG="{i1}" DEST="{j1}">')
            moved_hits += 1
    return LCSView(
        xx=_join_units(xx_units, mode),
        yy=_join_units(yy_units, mode),
        intersection=_join_units(inter_units, mode),
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
def build_result_from_records(
    r1: SegmentRecord,
    r2: SegmentRecord,
    annotation_level: str,
    mode: str,
    candidate_strategy: str,
    rel_lev: Optional[float],
    lev_dist: Optional[int],
    est_jaccard: Optional[float],
    exact_jaccard: Optional[float],
    shared_bands: Optional[int],
) -> PairResult:
    lcs = lcs_views(r1.units, r2.units, mode=mode)
    denom_1 = len(r1.units) or 1
    denom_2 = len(r2.units) or 1
    cov1 = lcs.lcs_length / denom_1
    cov2 = lcs.lcs_length / denom_2
    cov_mean = (cov1 + cov2) / 2.0
    cov_min = min(cov1, cov2)
    max_units = max(len(r1.units), len(r2.units))
    length_ratio = (min(len(r1.units), len(r2.units)) / max_units) if max_units else 1.0
    sim_candidates = [x for x in (est_jaccard, exact_jaccard, (1.0 - rel_lev) if rel_lev is not None else None) if x is not None]
    score_similarity = max(sim_candidates) if sim_candidates else 0.0
    score_lcs_mean_minus_distance = (cov_mean - rel_lev) if rel_lev is not None else None
    return PairResult(
        pair_id=f"{r1.idx}-{r2.idx}",
        i=r1.idx,
        j=r2.idx,
        segment_id_1=r1.seg_id,
        segment_id_2=r2.seg_id,
        segment_1=r1.raw_text,
        segment_2=r2.raw_text,
        repr_1=r1.repr_text.replace(UNIT_SEP, " "),
        repr_2=r2.repr_text.replace(UNIT_SEP, " "),
        annotation_level=annotation_level,
        candidate_strategy=candidate_strategy,
        mode=mode,
        rel_lev=rel_lev,
        lev_dist=lev_dist,
        est_jaccard=est_jaccard,
        exact_jaccard=exact_jaccard,
        shared_bands=shared_bands,
        len_1_units=len(r1.units),
        len_2_units=len(r2.units),
        len_1_chars=len(r1.repr_text),
        len_2_chars=len(r2.repr_text),
        length_ratio=length_ratio,
        xx=lcs.xx,
        yy=lcs.yy,
        intersection=lcs.intersection,
        lcs_sequence=lcs.lcs_sequence,
        lcs_length=lcs.lcs_length,
        lcs_cov_seg1=cov1,
        lcs_cov_seg2=cov2,
        lcs_cov_mean=cov_mean,
        lcs_cov_min=cov_min,
        same_position_hits=lcs.same_position_hits,
        moved_hits=lcs.moved_hits,
        score_similarity=score_similarity,
        score_lcs_mean_minus_distance=score_lcs_mean_minus_distance,
    )
# -------------------------
# Exhaustive mining
# -------------------------
_WORKER_SEGMENTS: List[SegmentRecord] = []
_WORKER_THRESHOLD: float = 0.0
_WORKER_MODE: str = "token"
_WORKER_ANNOTATION_LEVEL: str = "token"
def _init_worker(segments: List[SegmentRecord], threshold: float, mode: str, annotation_level: str) -> None:
    global _WORKER_SEGMENTS, _WORKER_THRESHOLD, _WORKER_MODE, _WORKER_ANNOTATION_LEVEL
    _WORKER_SEGMENTS = segments
    _WORKER_THRESHOLD = threshold
    _WORKER_MODE = mode
    _WORKER_ANNOTATION_LEVEL = annotation_level
def _process_range_exhaustive(task: Tuple[int, int]) -> List[PairResult]:
    start_i, end_i = task
    segs = _WORKER_SEGMENTS
    threshold = _WORKER_THRESHOLD
    mode = _WORKER_MODE
    annotation_level = _WORKER_ANNOTATION_LEVEL
    n = len(segs)
    out: List[PairResult] = []
    for i in range(start_i, min(end_i, n)):
        r1 = segs[i]
        len1 = len(r1.units)
        for j in range(i + 1, n):
            r2 = segs[j]
            max_len = max(len1, len(r2.units))
            if max_len == 0:
                continue
            if abs(len1 - len(r2.units)) / max_len >= threshold:
                continue
            lev_dist, rel_lev = relative_levenshtein_bounded_units(r1.units, r2.units, threshold)
            if lev_dist is None or rel_lev is None or rel_lev <= 0.0 or rel_lev >= threshold:
                continue
            out.append(build_result_from_records(r1, r2, annotation_level, mode, "exhaustive", rel_lev, lev_dist, None, None, None))
    return out
def make_index_ranges(n: int, workers: int, min_chunk: int = 32) -> List[Tuple[int, int]]:
    if n <= 0:
        return []
    if workers <= 1:
        return [(0, n)]
    chunk = max(min_chunk, math.ceil(n / workers))
    return [(start, min(n, start + chunk)) for start in range(0, n, chunk)]
def mine_pairs_exhaustive(
    segments: Sequence[SegmentRecord],
    annotation_level: str,
    rel_threshold: float,
    mode: str = "token",
    workers: int = 1,
    show_progress: bool = True,
) -> List[PairResult]:
    start = time.perf_counter()
    if workers <= 1:
        _init_worker(list(segments), rel_threshold, mode, annotation_level)
        results = _process_range_exhaustive((0, len(segments)))
        if show_progress:
            print_progress("pair mining", 1, 1, start)
            print(file=sys.stderr)
        return results
    ranges = make_index_ranges(len(segments), workers=workers)
    with mp.Pool(processes=workers, initializer=_init_worker, initargs=(list(segments), rel_threshold, mode, annotation_level)) as pool:
        out: List[PairResult] = []
        total = len(ranges)
        for done, part in enumerate(pool.imap_unordered(_process_range_exhaustive, ranges), start=1):
            out.extend(part)
            if show_progress:
                print_progress("pair mining", done, total, start)
        if show_progress:
            print(file=sys.stderr)
    return out
# -------------------------
# LSH / graph
# -------------------------
def build_shingle_sets(segments: Sequence[SegmentRecord], shingle_size: int, show_progress: bool = True) -> List[Set[str]]:
    start = time.perf_counter()
    out = []
    total = len(segments)
    step = max(1, total // 100)
    for idx, segment in enumerate(segments, start=1):
        out.append(make_shingles_from_units(segment.units, shingle_size))
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
def lsh_candidates(signatures: Sequence[Sequence[int]], bands: int, show_progress: bool = True) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], int], Dict[str, List[int]]]:
    if not signatures:
        return set(), {}, {}
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
    bucket_map: Dict[str, List[int]] = {}
    items = list(buckets.items())
    total_buckets = len(items)
    step_b = max(1, total_buckets // 100)
    for idx, ((band_id, sig_key), members) in enumerate(items, start=1):
        bucket_name = f"band{band_id}:" + hashlib.blake2b(repr(sig_key).encode("utf-8"), digest_size=6).hexdigest()
        bucket_map[bucket_name] = sorted(members)
        if len(members) > 1:
            for i, j in combinations(sorted(members), 2):
                pair_counts[(i, j)] = pair_counts.get((i, j), 0) + 1
        if show_progress and (idx == total_buckets or idx % step_b == 0):
            print_progress("candidate pairs", idx, total_buckets, cand_start)
    if show_progress:
        print(file=sys.stderr)
    return set(pair_counts.keys()), pair_counts, bucket_map
def estimate_jaccard_from_signatures(sig1: Sequence[int], sig2: Sequence[int]) -> float:
    return sum(1 for a, b in zip(sig1, sig2) if a == b) / min(len(sig1), len(sig2))
def score_lsh_candidates(
    segments: Sequence[SegmentRecord],
    annotation_level: str,
    shingle_sets: Sequence[Set[str]],
    signatures: Sequence[Sequence[int]],
    candidates: Sequence[Tuple[int, int]],
    pair_band_counts: Dict[Tuple[int, int], int],
    mode: str,
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
                out.append(build_result_from_records(
                    segments[i], segments[j], annotation_level, mode, "lsh", None, None, est, exact, pair_band_counts.get((i, j))
                ))
        if show_progress and total and (idx == total or idx % step == 0):
            print_progress("scoring lsh candidates", idx, total, start)
    if show_progress:
        print(file=sys.stderr)
    return out
def mine_pairs_lsh(
    segments: Sequence[SegmentRecord],
    annotation_level: str,
    mode: str = "token",
    shingle_size: int = 2,
    num_perm: int = 64,
    bands: Optional[int] = None,
    min_est_jaccard: float = 0.3,
    min_exact_jaccard: Optional[float] = None,
    show_progress: bool = True,
) -> Tuple[List[PairResult], Dict[str, List[int]]]:
    if num_perm <= 0:
        raise ValueError("num_perm must be > 0")
    if bands is None:
        bands = auto_bands(num_perm)
    if num_perm % bands != 0:
        raise ValueError("num_perm must be divisible by bands")
    shingle_sets = build_shingle_sets(segments, shingle_size, show_progress)
    signatures = build_signatures(shingle_sets, num_perm, show_progress)
    candidates, pair_band_counts, bucket_map = lsh_candidates(signatures, bands, show_progress)
    print(f"[info] LSH candidate pairs: {len(candidates)}", file=sys.stderr)
    results = score_lsh_candidates(
        segments,
        annotation_level,
        shingle_sets,
        signatures,
        sorted(candidates),
        pair_band_counts,
        mode,
        min_est_jaccard,
        min_exact_jaccard,
        show_progress,
    )
    return results, bucket_map
DEFAULT_COLUMNS = [
    "pair_id", "i", "j", "segment_id_1", "segment_id_2", "segment_1", "segment_2", "repr_1", "repr_2", "annotation_level",
    "candidate_strategy", "mode", "rel_lev", "lev_dist", "est_jaccard", "exact_jaccard", "shared_bands",
    "len_1_units", "len_2_units", "len_1_chars", "len_2_chars", "length_ratio",
    "xx", "yy", "intersection", "lcs_sequence", "lcs_length",
    "lcs_cov_seg1", "lcs_cov_seg2", "lcs_cov_mean", "lcs_cov_min",
    "same_position_hits", "moved_hits", "score_similarity", "score_lcs_mean_minus_distance",
]
def sort_results(results: List[PairResult], sort_by: str = "score_similarity") -> List[PairResult]:
    descending = {
        "est_jaccard", "exact_jaccard", "shared_bands", "lcs_length", "lcs_cov_seg1",
        "lcs_cov_seg2", "lcs_cov_mean", "lcs_cov_min", "same_position_hits",
        "moved_hits", "score_similarity", "score_lcs_mean_minus_distance", "length_ratio",
    }
    reverse = sort_by in descending
    return sorted(results, key=lambda r: (getattr(r, sort_by) is None, getattr(r, sort_by)), reverse=reverse)
def write_tsv(results: Sequence[PairResult], output_path: Path, columns: Sequence[str] = DEFAULT_COLUMNS) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
def connected_components(n: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    adj: Dict[int, Set[int]] = defaultdict(set)
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    seen: Set[int] = set()
    comps: List[List[int]] = []
    for node in range(n):
        if node in seen:
            continue
        stack = [node]
        comp = []
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(sorted(comp))
    return comps
def _tokenize_lcs_path(lcs_sequence: str) -> List[str]:
    return [tok for tok in lcs_sequence.split() if tok]
def _collect_lcs_paths(results: Sequence[PairResult], min_len: int = 1) -> List[Tuple[List[str], PairResult]]:
    collected: List[Tuple[List[str], PairResult]] = []
    for r in results:
        toks = _tokenize_lcs_path(r.lcs_sequence)
        if len(toks) >= min_len:
            collected.append((toks, r))
    return collected
def build_lcs_token_graph(results: Sequence[PairResult], min_len: int = 1) -> Dict[str, object]:
    """Build a strongly factorised weighted automaton-like graph.
    Node identity is the token label itself, so repeated subsequences such as
    'de -> le -> surface' are merged globally and their edge/node weights grow.
    This is intentionally more compact than a prefix trie and makes recurring
    local grammar fragments much more visible.
    """
    START = '__START__'
    END = '__END__'
    paths = _collect_lcs_paths(results, min_len=min_len)
    node_freq: Counter[str] = Counter()
    edge_freq: Counter[Tuple[str, str]] = Counter()
    edge_examples: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    path_freq: Counter[Tuple[str, ...]] = Counter()
    path_examples: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for toks, r in paths:
        path = tuple(toks)
        path_freq[path] += 1
        if len(path_examples[path]) < 5:
            path_examples[path].append(r.pair_id)
        full = [START] + toks + [END]
        for tok in toks:
            node_freq[tok] += 1
        node_freq[START] += 1
        node_freq[END] += 1
        for a, b in zip(full, full[1:]):
            edge_freq[(a, b)] += 1
            if len(edge_examples[(a, b)]) < 5:
                edge_examples[(a, b)].append(r.pair_id)
    nodes = []
    node_ids: Dict[str, str] = {START: 'n0', END: 'n1'}
    nodes.append({'id': 'n0', 'token': 'START', 'weight': node_freq[START], 'special': True})
    nodes.append({'id': 'n1', 'token': 'END', 'weight': node_freq[END], 'special': True})
    sorted_tokens = sorted(tok for tok in node_freq.keys() if tok not in {START, END})
    for idx, tok in enumerate(sorted_tokens, start=2):
        nid = f'n{idx}'
        node_ids[tok] = nid
        nodes.append({'id': nid, 'token': tok, 'weight': node_freq[tok], 'special': False})
    edges = []
    for (a, b), w in sorted(edge_freq.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
        edges.append({
            'source': node_ids[a],
            'target': node_ids[b],
            'source_token': 'START' if a == START else ('END' if a == END else a),
            'target_token': 'START' if b == START else ('END' if b == END else b),
            'weight': w,
            'examples': edge_examples[(a, b)],
        })
    path_payload = []
    for path, freq in sorted(path_freq.items(), key=lambda x: (-x[1], -len(x[0]), x[0])):
        path_payload.append({
            'sequence': ' '.join(path),
            'length': len(path),
            'freq': freq,
            'examples': path_examples[path],
        })
    return {
        'graph_kind': 'lcs_token_graph',
        'factorisation': 'global_token_paths',
        'min_lcs_length': min_len,
        'nodes': nodes,
        'edges': edges,
        'paths': path_payload,
    }
def build_similarity_graph_payload(
    segments: Sequence[SegmentRecord],
    results: Sequence[PairResult],
    bucket_map: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, object]:
    edges = []
    for r in results:
        edges.append({
            'source': r.i,
            'target': r.j,
            'pair_id': r.pair_id,
            'score_similarity': r.score_similarity,
            'lcs_length': r.lcs_length,
            'lcs_sequence': r.lcs_sequence,
            'shared_bands': r.shared_bands,
            'exact_jaccard': r.exact_jaccard,
            'est_jaccard': r.est_jaccard,
            'rel_lev': r.rel_lev,
        })
    comps = connected_components(len(segments), [(r.i, r.j) for r in results])
    node_to_comp = {}
    for comp_id, comp in enumerate(comps):
        for node in comp:
            node_to_comp[node] = comp_id
    return {
        'graph_kind': 'similarity',
        'nodes': [
            {
                'id': s.idx,
                'segment_id': s.seg_id,
                'label': s.raw_text,
                'repr': s.repr_text.replace(UNIT_SEP, ' '),
                'n_units': len(s.units),
                'component': node_to_comp.get(s.idx, -1),
                'meta': s.meta,
            }
            for s in segments
        ],
        'edges': edges,
        'components': comps,
        'lsh_buckets': {k: v for k, v in (bucket_map or {}).items() if len(v) > 1},
    }
def write_graph_json(
    path: Path,
    segments: Sequence[SegmentRecord],
    results: Sequence[PairResult],
    bucket_map: Optional[Dict[str, List[int]]] = None,
    graph_kind: str = 'lcs_token_graph',
    min_lcs_length: int = 1,
) -> None:
    if graph_kind == 'similarity':
        payload = build_similarity_graph_payload(segments, results, bucket_map)
    else:
        payload = build_lcs_token_graph(results, min_len=min_lcs_length)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
def write_graphviz_dot(
    path: Path,
    segments: Sequence[SegmentRecord],
    results: Sequence[PairResult],
    graph_kind: str = 'lcs_token_graph',
    min_lcs_length: int = 1,
) -> None:
    if graph_kind == 'similarity':
        lines = ['graph corpus_similarity {']
        lines.append('  graph [overlap=false, splines=true];')
        lines.append('  node [shape=ellipse, fontsize=10];')
        for s in segments:
            label = s.raw_text.replace('"', "'")
            lines.append(f'  n{s.idx} [label="{s.idx}: {label}"];')
        for r in results:
            w = f"{r.score_similarity:.3f}"
            lcs = r.lcs_sequence.replace('"', "'")
            lines.append(f'  n{r.i} -- n{r.j} [label="{w} | {lcs}"];')
        lines.append('}')
        path.write_text('\n'.join(lines), encoding='utf-8')
        return
    payload = build_lcs_token_graph(results, min_len=min_lcs_length)
    lines = ['digraph lcs_grammar {']
    lines.append('  rankdir=LR;')
    lines.append('  graph [overlap=false, splines=true];')
    lines.append('  node [shape=circle, fontsize=10];')
    for node in payload['nodes']:
        token = str(node['token']).replace('"', "'")
        weight = int(node['weight'])
        shape = 'doublecircle' if node.get('special') else 'circle'
        lines.append(f'  {node["id"]} [shape={shape}, label="{token}\n[{weight}]"];')
    for edge in payload['edges']:
        tgt = str(edge['target_token']).replace('"', "'")
        w = int(edge['weight'])
        lines.append(f'  {edge["source"]} -> {edge["target"]} [label="{tgt} | {w}"];')
    lines.append('}')
    path.write_text('\n'.join(lines), encoding='utf-8')
# -------------------------
# CLI
# -------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extraction of similar segment pairs + LCS patterns (plain text or UD/CoNLL-U; exhaustive or MinHash/LSH).")
    p.add_argument("input", type=Path, help="Input file: plain text (one segment per line) or CoNLL-U.")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output TSV path.")
    p.add_argument("--input-format", choices=["auto", "plain", "conllu"], default="auto", help="Input format.")
    p.add_argument("--annotation-level", choices=["token", "lemma", "lemma_upos", "token_lemma", "token_lemma_upos"], default="token", help="Linguistic representation used for comparison.")
    p.add_argument("--candidate-strategy", choices=["exhaustive", "lsh"], default="exhaustive", help="How to identify candidate pairs.")
    p.add_argument("-t", "--threshold", type=float, default=0.4, help="Maximum relative unit Levenshtein distance for exhaustive strategy.")
    p.add_argument("-m", "--mode", choices=["token"], default="token", help="LCS mode. Token mode is recommended for linguistic work.")
    p.add_argument("--token-pattern", default=r"\S+", help=r"Regex used to extract tokens in plain-text mode.")
    p.add_argument("--encoding", default="utf-8", help="Input encoding.")
    p.add_argument("--lowercase", action="store_true", help="Lowercase surface forms and lemmas before comparison.")
    p.add_argument("--normalize-num", action="store_true", help="Replace numeric tokens with <NUM>.")
    p.add_argument("--keep-duplicates", action="store_true", help="Keep duplicate segments (default: deduplicate by representation).")
    p.add_argument("-w", "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Worker processes for exhaustive strategy.")
    p.add_argument("--sort-by", default="score_similarity", choices=DEFAULT_COLUMNS, help="Column used to sort results before export.")
    p.add_argument("--limit", type=int, default=0, help="Optional cap on the number of rows written.")
    p.add_argument("--shingle-size", type=int, default=2, help="Shingle size for LSH.")
    p.add_argument("--num-perm", type=int, default=64, help="Number of MinHash permutations.")
    p.add_argument("--bands", type=int, default=0, help="Number of LSH bands. 0 = automatic choice.")
    p.add_argument("--min-est-jaccard", type=float, default=0.3, help="Minimum estimated Jaccard in LSH mode.")
    p.add_argument("--min-exact-jaccard", type=float, default=-1.0, help="Optional exact Jaccard threshold in LSH mode. -1 disables it.")
    p.add_argument("--graph-json", type=Path, default=None, help="Optional JSON graph export. Default graph kind = factorised LCS token graph.")
    p.add_argument("--graph-dot", type=Path, default=None, help="Optional Graphviz DOT export. Default graph kind = factorised LCS token graph.")
    p.add_argument("--graph-kind", choices=["lcs_token_graph", "similarity"], default="lcs_token_graph", help="Type of graph to export. lcs_token_graph merges repeated tokens/edges globally across extracted LCS paths.")
    p.add_argument("--graph-min-lcs-length", type=int, default=1, help="Minimum LCS length included in the graph exports.")
    return p
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.candidate_strategy == "exhaustive" and not (0.0 < args.threshold < 1.0):
        parser.error("--threshold must be in ]0,1[ for exhaustive strategy.")
    if args.input_format == "plain" and args.annotation_level != "token":
        parser.error("For plain text without UD annotations, use --annotation-level token.")
    total_start = time.perf_counter()
    load_start = time.perf_counter()
    segments = load_segments(
        path=args.input,
        input_format=args.input_format,
        annotation_level=args.annotation_level,
        token_pattern=args.token_pattern,
        encoding=args.encoding,
        lowercase=args.lowercase,
        normalize_num=args.normalize_num,
        deduplicate=not args.keep_duplicates,
    )
    print(f"[time] loading/preparation: {format_seconds(time.perf_counter() - load_start)}", file=sys.stderr)
    print(f"[info] prepared segments: {len(segments)}", file=sys.stderr)
    if len(segments) >= 2:
        print(f"[info] theoretical candidate pairs: {len(segments) * (len(segments) - 1) // 2}", file=sys.stderr)
    print(f"[info] candidate strategy: {args.candidate_strategy}", file=sys.stderr)
    print(f"[info] input format: {args.input_format}", file=sys.stderr)
    print(f"[info] annotation level: {args.annotation_level}", file=sys.stderr)
    print(f"[info] lowercase: {args.lowercase}", file=sys.stderr)
    print(f"[info] normalize_num: {args.normalize_num}", file=sys.stderr)
    mining_start = time.perf_counter()
    bucket_map: Dict[str, List[int]] = {}
    if args.candidate_strategy == "exhaustive":
        print(f"[info] workers: {args.workers}", file=sys.stderr)
        results = mine_pairs_exhaustive(
            segments=segments,
            annotation_level=args.annotation_level,
            rel_threshold=args.threshold,
            mode=args.mode,
            workers=args.workers,
            show_progress=True,
        )
    else:
        bands = args.bands if args.bands > 0 else auto_bands(args.num_perm)
        min_exact = None if args.min_exact_jaccard < 0 else args.min_exact_jaccard
        print(f"[info] shingle size: {args.shingle_size}", file=sys.stderr)
        print(f"[info] num_perm: {args.num_perm}", file=sys.stderr)
        print(f"[info] bands: {bands}", file=sys.stderr)
        print(f"[info] min_est_jaccard: {args.min_est_jaccard}", file=sys.stderr)
        if min_exact is not None:
            print(f"[info] min_exact_jaccard: {min_exact}", file=sys.stderr)
        results, bucket_map = mine_pairs_lsh(
            segments=segments,
            annotation_level=args.annotation_level,
            mode=args.mode,
            shingle_size=args.shingle_size,
            num_perm=args.num_perm,
            bands=bands,
            min_est_jaccard=args.min_est_jaccard,
            min_exact_jaccard=min_exact,
            show_progress=True,
        )
    print(f"[time] pair mining: {format_seconds(time.perf_counter() - mining_start)}", file=sys.stderr)
    sort_start = time.perf_counter()
    results = sort_results(results, sort_by=args.sort_by)
    print(f"[time] sorting: {format_seconds(time.perf_counter() - sort_start)}", file=sys.stderr)
    if args.limit > 0:
        results = results[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_start = time.perf_counter()
    write_tsv(results, args.output)
    print(f"[time] writing TSV: {format_seconds(time.perf_counter() - write_start)}", file=sys.stderr)
    if args.graph_json:
        args.graph_json.parent.mkdir(parents=True, exist_ok=True)
        write_graph_json(args.graph_json, segments, results, bucket_map, graph_kind=args.graph_kind, min_lcs_length=args.graph_min_lcs_length)
        print(f"[info] wrote graph JSON: {args.graph_json}", file=sys.stderr)
    if args.graph_dot:
        args.graph_dot.parent.mkdir(parents=True, exist_ok=True)
        write_graphviz_dot(args.graph_dot, segments, results, graph_kind=args.graph_kind, min_lcs_length=args.graph_min_lcs_length)
        print(f"[info] wrote graph DOT: {args.graph_dot}", file=sys.stderr)
    print(f"[time] total: {format_seconds(time.perf_counter() - total_start)}", file=sys.stderr)
    print(f"[info] retained pairs: {len(results)}", file=sys.stderr)
    print(f"[info] wrote: {args.output}", file=sys.stderr)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())