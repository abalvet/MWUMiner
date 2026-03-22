# `baseline_pair_miner_lsh.py` — v1.2

Extraction of similar segment pairs with LCS-based MWU candidate patterns.  
Supports two candidate strategies: **exhaustive** (baseline) and **LSH** (MinHash + Locality-Sensitive Hashing).

---

## Table of Contents

1. [Overview and Motivation](#1-overview-and-motivation)
2. [Processing Pipeline](#2-processing-pipeline)
   - 2.1 [Exhaustive strategy](#21-exhaustive-strategy)
   - 2.2 [LSH strategy](#22-lsh-strategy)
3. [Output Format](#3-output-format)
   - 3.1 [Pair identification and provenance](#31-pair-identification-and-provenance)
   - 3.2 [Similarity measures](#32-similarity-measures)
   - 3.3 [LCS alignment view](#33-lcs-alignment-view)
   - 3.4 [Scores](#34-scores)
4. [Command-line Options](#4-command-line-options)
5. [Usage Examples](#5-usage-examples)
6. [Known Limitations and Remaining Issues](#6-known-limitations-and-remaining-issues)
7. [Roadmap](#7-roadmap)
8. [Changelog](#8-changelog)

---

## 1. Overview and Motivation

In corpus linguistics, **multiword units (MWUs)** — fixed or semi-fixed sequences such as
collocations, idioms, formulaic chunks, and support-verb constructions — are typically identified
by frequency-based association measures applied to n-gram counts.  A complementary approach,
explored here, works from **pairwise similarity**: two corpus segments (sentences, utterances,
or other units) that share a large common subsequence are likely to contain a recurring
phraseological pattern.  The longest common subsequence (LCS) of a similar pair is a
hypothesis about that shared pattern; aggregating LCS candidates across many pairs gives a
ranked inventory of MWU candidates without pre-committing to a fixed window size.

This tool implements that pipeline:

1. Read a corpus as a flat list of segments (one per line).
2. Identify **candidate pairs** that are sufficiently similar to be worth comparing
   (either exhaustively or via approximate hashing).
3. For each retained pair, compute the **LCS** at the token or character level, along with a
   rich set of similarity and coverage metrics.
4. Write all results to a TSV file, sorted by a chosen score.

The LSH strategy makes the approach **scalable**: instead of evaluating all *n(n-1)/2* pairs
(O(n²)), MinHash signatures and band bucketing reduce the search space to a small fraction of
candidate pairs while preserving, with high probability, pairs whose Jaccard similarity of
shingle sets exceeds a configurable threshold.  This makes it practical on corpora of tens of
thousands of segments where exhaustive comparison would be prohibitively slow.

---

## 2. Processing Pipeline

### 2.1 Exhaustive strategy

```
Input segments
    │
    ▼
Deduplication + sorting  (optionally disabled with --no-dedup)
    │
    ▼  for every pair (i, j), i < j
Character-level length filter
    │   abs(len(s1) - len(s2)) / max_len < threshold
    ▼
Bounded relative Levenshtein distance
    │   rel_lev(s1, s2) = edit_dist / max_char_len
    │   pair kept if 0 < rel_lev < threshold
    ▼
Token-level LCS computation
    │
    ▼
LCS coverage filter  (optional, --min-lcs-cov)
    │
    ▼
PairResult  ──►  TSV output
```

The exhaustive strategy distributes work across multiple processes using Python's
`multiprocessing.Pool`.  Each worker handles a contiguous slice of the sorted segment list and
filters pairs independently.  Because filtering is character-level (Levenshtein on raw strings),
it acts as a cheap pre-filter before the more informative token-level LCS step.

> **Note on the threshold parameter.**  The `--threshold` value is a *maximum* relative
> Levenshtein distance, i.e. a value of `0.4` means "keep pairs that differ by at most 40 % of
> the length of the longer string".  Identical strings (`rel_lev = 0.0`) are *excluded*; only
> pairs with `0 < rel_lev < threshold` are retained.

### 2.2 LSH strategy

```
Input segments
    │
    ▼
Deduplication + sorting  (optionally disabled with --no-dedup)
    │
    ▼
Shingling
    │   Each segment → set of k-shingles (overlapping n-grams of tokens or chars)
    ▼
MinHash signatures
    │   Each shingle set → vector of length num_perm
    │   Each component = min hash value over all shingles (Blake2b, salted)
    ▼
LSH band bucketing
    │   Signature split into `bands` sub-vectors of length (num_perm / bands)
    │   Pairs landing in the same bucket at least once are candidate pairs
    │   Effective Jaccard threshold displayed on stderr: t ≈ (1/bands)^(1/rows)
    ▼
Jaccard filter
    │   Estimated Jaccard  (from signature agreement)  ≥ min_est_jaccard
    │   Exact Jaccard      (from shingle sets)         ≥ min_exact_jaccard  [optional]
    ▼
Token-level LCS + character-level rel_lev computation
    │
    ▼
LCS coverage filter  (optional, --min-lcs-cov)
    │
    ▼
PairResult  ──►  TSV output
```

The key insight of LSH is the **band trick**: the probability that two segments share at least
one identical band sub-vector grows sharply around the Jaccard threshold `t ≈ (1/bands)^(1/rows)`
(where `rows = num_perm / bands`).  The `auto_bands` function selects a band count that gives a
reasonable amplification curve for the chosen `num_perm`.  Increasing `bands` raises recall but
also increases the number of candidate pairs evaluated; decreasing `bands` (i.e. increasing
`rows`) makes the filter stricter and faster.

The effective threshold is printed to `stderr` at startup, making it easy to calibrate without
manual computation.

---

## 3. Output Format

The output is a UTF-8 TSV file.  Each row represents one retained segment pair.  Column order
follows the `DEFAULT_COLUMNS` list defined in the script.

### 3.1 Pair identification and provenance

| Column | Type | Description |
|--------|------|-------------|
| `pair_id` | string | `"i-j"` identifier for the pair. |
| `i` | int | Index of segment 1 in the sorted segment list (0-based). |
| `j` | int | Index of segment 2 (always `j > i`). |
| `segment_1` | string | Text of segment 1. |
| `segment_2` | string | Text of segment 2. |
| `candidate_strategy` | string | `"exhaustive"` or `"lsh"`. |
| `mode` | string | `"token"` or `"char"`: tokenization and LCS unit granularity. |

### 3.2 Similarity measures

Both strategies now populate `rel_lev`, `lev_dist`, `est_jaccard`, `exact_jaccard`,
and `shared_bands` where they are naturally available; in LSH mode, `rel_lev` and `lev_dist`
are computed as a post-filter step on retained candidates.

| Column | Exhaustive | LSH | Description |
|--------|:----------:|:---:|-------------|
| `rel_lev` | ✓ | ✓ | Relative Levenshtein distance: `edit_dist / max(len(s1), len(s2))` at the **character** level.  Lower = more similar. |
| `lev_dist` | ✓ | ✓ | Raw character-level edit distance. |
| `est_jaccard` | ✗ | ✓ | Estimated Jaccard similarity derived from MinHash signature agreement. |
| `exact_jaccard` | ✗ | ✓ | Exact Jaccard similarity computed from shingle sets: `|A ∩ B| / |A ∪ B|`. |
| `shared_bands` | ✗ | ✓ | Number of LSH bands in which the pair co-occurred in the same bucket. |
| `len_1_units` | ✓ | ✓ | Number of tokens (or chars) in segment 1. |
| `len_2_units` | ✓ | ✓ | Number of tokens (or chars) in segment 2. |
| `len_1_chars` | ✓ | ✓ | Raw character length of segment 1. |
| `len_2_chars` | ✓ | ✓ | Raw character length of segment 2. |
| `length_ratio` | ✓ | ✓ | `min(len_1_units, len_2_units) / max(len_1_units, len_2_units)`. |

### 3.3 LCS alignment view

The LCS is computed at the **token** (or character) level using a full dynamic-programming
algorithm.  The `intersection` column of v1.1 (which mixed plain text and XML-like `<MOV …>`
markup) has been replaced by two clean, TSV-safe columns:

| Column | Description |
|--------|-------------|
| `xx` | Segment 1 with non-LCS tokens replaced by `"-"`. |
| `yy` | Segment 2 with non-LCS tokens replaced by `"-"`. |
| `lcs_stable` | LCS tokens that appear at the **same position** in both segments (strict positional overlap).  This is the "frozen" part of the pattern, invariant across the two variants. |
| `lcs_moved` | LCS tokens that are part of the shared subsequence but appear at **different positions** in the two segments (permuted elements).  These signal syntactic mobility — constituents that can move under certain conditions (topicalisation, focus, scrambling). |
| `lcs_sequence` | Full LCS token sequence in order, without positional annotation (`lcs_stable ∪ lcs_moved` reordered by occurrence). |
| `lcs_length` | Total number of tokens in the LCS (`same_position_hits + moved_hits`). |
| `lcs_cov_seg1` | LCS coverage of segment 1: `lcs_length / len_1_units`. |
| `lcs_cov_seg2` | LCS coverage of segment 2: `lcs_length / len_2_units`. |
| `lcs_cov_mean` | Mean of the two coverage values. |
| `lcs_cov_min` | Minimum of the two coverage values (conservative estimate). |
| `same_position_hits` | Count of tokens in `lcs_stable`. |
| `moved_hits` | Count of tokens in `lcs_moved`. |

**Interpretation for MWU mining.**  The `lcs_sequence` column is the primary MWU candidate
extracted from the pair.  `lcs_stable` directly gives the positionally-fixed core of the
shared pattern — the best signal for a frozen MWU skeleton.  `lcs_moved` highlights the mobile
elements: constituents whose position shifts between the two variants, which is valuable
evidence for syntactic flexibility analysis and for future PatternDAG construction.

**Example.** For the pair *"il fait beau aujourd'hui"* / *"il fait très beau aujourd'hui"*:

- `lcs_stable` = `il fait`  — these two tokens are in the same position in both segments.
- `lcs_moved` = `beau aujourd'hui`  — these tokens slide one position to the right because
  *très* is inserted; they are still in the LCS but no longer in the same slot.
- `lcs_sequence` = `il fait beau aujourd'hui`  — the full shared subsequence.

### 3.4 Scores

The scores are now clearly separated by what they measure:

| Column | Strategies | Description |
|--------|:----------:|-------------|
| `score_similarity` | LSH only | Best Jaccard-based similarity: `max(est_jaccard, exact_jaccard)`.  `None` in exhaustive mode (use `score_lev_sim` instead). |
| `score_lev_sim` | both | Levenshtein-based similarity: `1 - rel_lev`.  Available in both strategies. |
| `score_lcs_mean_minus_distance` | both | `lcs_cov_mean - rel_lev`: rewards high LCS coverage and penalises surface distance. |

> **Rationale for the separation.**  In v1.1, `score_similarity` conflated
> `1 - rel_lev` (character-level, exhaustive) with Jaccard (shingle-level, LSH), making
> cross-strategy comparisons misleading.  The two are now kept in separate columns so that
> each can be used independently or combined in downstream metrics — for example, MWU density
> indices that weight both inter-segment similarity and LCS coverage.

---

## 4. Command-line Options

```
python baseline_pair_miner_lsh.py <input> -o <output> [options]
```

### Positional

| Argument | Description |
|----------|-------------|
| `input` | Path to the input text file.  One segment per line.  Blank lines are ignored. |

### Output

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output` | *(required)* | Path to the output TSV file.  Parent directories are created automatically. |

### Strategy selection

| Option | Default | Description |
|--------|---------|-------------|
| `--candidate-strategy` | `exhaustive` | `"exhaustive"` evaluates all O(n²) pairs filtered by relative Levenshtein distance.  `"lsh"` uses MinHash + LSH for approximate candidate selection. |

### Shared options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--mode` | `token` | Tokenization granularity for shingling and LCS computation. |
| `--token-pattern` | `\S+` | Python regex used to extract tokens in `token` mode. |
| `--encoding` | `utf-8` | Input file encoding. |
| `--sort-by` | `lcs_cov_mean` | Column used to sort the output before writing.  Similarity/coverage columns sort descending; distance columns sort ascending. |
| `--limit` | `0` (disabled) | If > 0, truncate output to the top N rows after sorting. |
| `--min-lcs-cov` | `0.0` (disabled) | Minimum `lcs_cov_mean` required to keep a pair.  Applied after mining, before sorting.  Recommended range: `0.5`–`0.7` for high-quality MWU candidates. |
| `--no-dedup` | off | Disable input deduplication.  By default, duplicate lines are silently collapsed into one representative before comparison.  Use `--no-dedup` when duplicate segments carry evidence of productivity (e.g., repeated utterances in spoken corpora). |

### Exhaustive strategy options

| Option | Default | Description |
|--------|---------|-------------|
| `-t`, `--threshold` | `0.4` | Maximum relative Levenshtein distance (character-level).  Must be in ]0, 1[. |
| `-w`, `--workers` | `cpu_count - 1` | Number of worker processes. |

### LSH strategy options

| Option | Default | Description |
|--------|---------|-------------|
| `--shingle-size` | `2` | Size of shingles (n-grams).  Size 1 = bag-of-words; size 2 = bigrams. |
| `--num-perm` | `64` | Number of MinHash permutations.  Must be divisible by `--bands`. |
| `--bands` | `0` (auto) | Number of LSH bands.  `0` triggers `auto_bands`.  The effective Jaccard threshold `t ≈ (1/bands)^(1/rows)` is printed to `stderr` on startup. |
| `--min-est-jaccard` | `0.3` | Minimum estimated Jaccard (from MinHash signatures) to keep a candidate pair. |
| `--min-exact-jaccard` | `-1` (disabled) | Minimum exact Jaccard (from shingle sets).  `-1` disables this filter. |

---

## 5. Usage Examples

**Exhaustive baseline on a small corpus, filtering by LCS coverage:**
```bash
python baseline_pair_miner_lsh.py corpus.txt \
    -o results/exhaustive.tsv \
    --candidate-strategy exhaustive \
    --threshold 0.35 \
    --mode token \
    --workers 4 \
    --min-lcs-cov 0.5 \
    --sort-by lcs_cov_mean
```

**LSH on a large corpus, bigram shingles, relaxed recall:**
```bash
python baseline_pair_miner_lsh.py corpus.txt \
    -o results/lsh_relaxed.tsv \
    --candidate-strategy lsh \
    --shingle-size 2 \
    --num-perm 128 \
    --bands 32 \
    --min-est-jaccard 0.25 \
    --mode token
```

**LSH with exact Jaccard post-filter, top 5000 pairs by LCS coverage:**
```bash
python baseline_pair_miner_lsh.py corpus.txt \
    -o results/lsh_strict.tsv \
    --candidate-strategy lsh \
    --num-perm 64 \
    --min-est-jaccard 0.3 \
    --min-exact-jaccard 0.4 \
    --min-lcs-cov 0.5 \
    --sort-by lcs_cov_mean \
    --limit 5000
```

**Spoken corpus with productive repetitions (no deduplication):**
```bash
python baseline_pair_miner_lsh.py spoken.txt \
    -o results/spoken.tsv \
    --candidate-strategy lsh \
    --no-dedup \
    --min-lcs-cov 0.6
```

**Character-level mode (morphologically rich languages):**
```bash
python baseline_pair_miner_lsh.py corpus.txt \
    -o results/char.tsv \
    --candidate-strategy lsh \
    --mode char \
    --shingle-size 3
```

**Checking the effective LSH Jaccard threshold before a run:**

The threshold is always printed to `stderr`:
```
[info] bands: 16
[info] effective Jaccard threshold ≈ 0.500
```

To compute it manually for a given `(num_perm, bands)` configuration:
```python
from baseline_pair_miner_lsh import lsh_jaccard_threshold
print(lsh_jaccard_threshold(num_perm=64, bands=16))   # 0.5
print(lsh_jaccard_threshold(num_perm=128, bands=32))  # 0.5
print(lsh_jaccard_threshold(num_perm=64, bands=8))    # 0.354
```

---

## 6. Known Limitations and Remaining Issues

### 6.1 Exhaustive filter is character-level; LCS is token-level  *(unchanged)*

In the exhaustive strategy, the pre-filter uses **character-level** relative Levenshtein
distance.  The LCS is computed over **tokens**.  Pairs with many shared tokens but different
surface forms (clitics, punctuation) may be incorrectly discarded or produce a low-quality
token-level LCS.

### 6.2 No downstream aggregation  *(partially addressed)*

This script produces **pair-level** LCS candidates; it does not aggregate them into a
frequency-ranked MWU inventory.  The `lcs_sequence`, `lcs_stable`, and `lcs_moved` columns
are designed to feed a downstream PatternDAG extraction step (handled in the non-baseline
version of the pipeline).  In this baseline, PatternDAG output is illustrative only: format
and column names are stable and documented, but no aggregation logic is implemented.

### 6.3 No segment provenance tracking  *(unchanged)*

The input is a flat list of segments.  There is no facility to record which source document(s)
or sentence positions a segment came from.

### 6.4 Multiprocessing not available for LSH strategy  *(unchanged)*

The `--workers` option only affects the exhaustive strategy.  The LSH pipeline runs in a single
process.  The non-baseline version parallelises the LSH step.

### 6.5 `auto_bands` may give a suboptimal amplification curve  *(partially addressed)*

`auto_bands` uses a small set of preferred `rows` values (`4, 5, 6, 8`).  For non-standard
`num_perm` values the resulting `(bands, rows)` pair may not give the intended Jaccard threshold.
The `lsh_jaccard_threshold()` helper now makes it easy to inspect the effective threshold and
set `--bands` explicitly when a specific operating point is required.

---

## 7. Roadmap

The following improvements are planned for future versions, in rough priority order:

### Short term

- **Cross-strategy score normalisation.**  A single composite score that is directly
  comparable between exhaustive and LSH runs (e.g., a weighted combination of
  `score_lev_sim`, `exact_jaccard`, and `lcs_cov_mean`).  This would enable merged output
  from both strategies without requiring the user to select strategy-specific columns.

- **MWU density metric.**  A textometric indicator combining `lcs_cov_mean` and the
  frequency of the `lcs_sequence` pattern across the corpus, suitable for use as a
  novel specificity or productivity measure.

- **PatternDAG integration (baseline stub).**  Even in this baseline version, an optional
  `--dag-output` flag could write a minimal DAG representation (node: lcs_sequence, edges:
  pair_id references) to a separate file, providing a stable format contract with the
  non-baseline pipeline.

### Medium term

- **Segment provenance tracking.**  A `--segment-id-col` option accepting a companion
  TSV mapping each segment text to its source document and position, propagated to
  `PairResult` as `doc_1`, `pos_1`, `doc_2`, `pos_2` columns.

- **LSH multiprocessing.**  Port the signature-building and candidate-scoring steps to
  `multiprocessing.Pool` for large-corpus runs in this baseline version (the non-baseline
  pipeline already does this).

- **Token-level Levenshtein option.**  A `--lev-unit token` flag to compute relative
  Levenshtein distance at the token level in the exhaustive strategy, eliminating the
  character/token mismatch described in §6.1.

### Longer term

- **Streaming output.**  Write results to TSV incrementally rather than accumulating all
  `PairResult` objects in memory, enabling processing of very large corpora on
  memory-constrained machines.

- **Lemma normalisation.**  An optional `--lemma-map` TSV (form → lemma) to collapse
  morphological variants before shingling and LCS computation, reducing sparsity on
  inflected languages.

---

## 8. Changelog

### v1.2  *(current)*

**Breaking changes (output schema):**

- `intersection` column removed.  Replaced by two clean columns:
  - `lcs_stable`: LCS tokens at identical positions in both segments.
  - `lcs_moved`: LCS tokens that appear in the LCS but at different positions.
  - No more XML-like `<MOV …>` markup; both columns contain plain token sequences,
    safe for any TSV consumer.

**New columns:**

- `score_lev_sim`: `1 - rel_lev`, now a separate column distinct from `score_similarity`.
  Available in both strategies.

**Changed columns:**

- `score_similarity`: now **Jaccard-only** (`max(est_jaccard, exact_jaccard)`).
  `None` in exhaustive mode.  Use `score_lev_sim` for exhaustive results.

**New features:**

- LSH mode now computes `rel_lev` and `lev_dist` for retained candidate pairs
  (character-level, unbounded).  Both strategies now share a common set of populated fields.
- New `--min-lcs-cov` option: filter output pairs by minimum `lcs_cov_mean`.
- New `--no-dedup` option: disable input deduplication (useful for spoken corpora with
  productive repetitions).
- New `lsh_jaccard_threshold(num_perm, bands)` helper function: returns the effective
  Jaccard threshold for a given LSH configuration.  Also printed to `stderr` at startup
  in LSH mode.
- Default `--sort-by` changed from `score_similarity` to `lcs_cov_mean`.

**Fixes / improvements:**

- `prepare_segments` accepts a `dedup` parameter.
- `sort_results` descending set updated with `score_lev_sim`.

### v1.1  *(previous)*

- Added LSH strategy (`--candidate-strategy lsh`) with MinHash signatures and band bucketing.
- Added `auto_bands` heuristic.
- Initial `intersection` column with `<MOV …>` markup.
- `score_similarity` conflated Levenshtein and Jaccard measures.
