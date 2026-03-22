# `ud_pair_miner_lsh.py`: v3

Extraction of similar segment pairs with LCS-based MWU candidate patterns.  
Accepts **plain text** or **Universal Dependencies CoNLL-U annotated** files as input.  
Supports two candidate strategies: **exhaustive** (baseline) and **LSH** (MinHash + Locality-Sensitive Hashing).  
Optionally exports **LCS grammar and similarity graphs** in JSON and Graphviz DOT formats.

---

## Table of Contents

1. [Overview and Motivation](#1-overview-and-motivation)
2. [Input Formats and Annotation Levels](#2-input-formats-and-annotation-levels)
   - 2.1 [Plain text](#21-plain-text)
   - 2.2 [CoNLL-U (UD)](#22-conll-u-ud)
   - 2.3 [UD annotation levels](#23-ud-annotation-levels)
3. [Processing Pipeline](#3-processing-pipeline)
   - 3.1 [Exhaustive strategy](#31-exhaustive-strategy)
   - 3.2 [LSH strategy](#32-lsh-strategy)
4. [Output Format](#4-output-format)
   - 4.1 [Pair identification and provenance](#41-pair-identification-and-provenance)
   - 4.2 [Similarity measures](#42-similarity-measures)
   - 4.3 [LCS alignment view](#43-lcs-alignment-view)
   - 4.4 [Scores](#44-scores)
   - 4.5 [Graph exports](#45-graph-exports)
5. [Command-line Options](#5-command-line-options)
6. [Usage Examples](#6-usage-examples)
7. [Known Limitations and Remaining Issues](#7-known-limitations-and-remaining-issues)
8. [Roadmap](#8-roadmap)
9. [Changelog](#9-changelog)

---

## 1. Overview and Motivation

In corpus linguistics, **multiword units (MWUs)**,  fixed or semi-fixed sequences such as
collocations, idioms, formulaic chunks, and support-verb constructions, are typically identified
by frequency-based association measures applied to n-gram counts.  A complementary approach,
explored here, works from **pairwise similarity**: two corpus segments (sentences, utterances,
or other units) that share a large common subsequence are likely to contain a recurring
phraseological pattern.  The longest common subsequence (LCS) of a similar pair is a
hypothesis about that shared pattern; aggregating LCS candidates across many pairs gives a
ranked inventory of MWU candidates without pre-committing to a fixed window size.

**This version extends the baseline** by exploiting the morphosyntactic annotations available
in the Universal Dependencies CoNLL-U format.  Where the baseline operates solely on surface
forms, `ud_pair_miner_lsh.py` can represent each segment as a sequence of **lemmas**,
**lemma|UPOS pairs**, or **form|lemma|UPOS triples**, thereby enabling:

- abstraction over inflectional variation during similarity computation and LCS extraction;
- recovery of recurring morphosyntactic patterns (constructional skeletons) rather than
  surface-form coincidences;
- preservation of raw surface text in the result columns for interpretability, regardless
  of the representation level used for comparison.

**Syntactic dependency information (head, deprel, deps) is not used** in the comparison units:
only the form, lemma, and upos columns of UD tokens are involved.  This restriction is
deliberate: similarity scores and LCS outputs must remain comparable across CoNLL-U files
produced by different parsers, and the integration of dependency arcs belongs to a later
analysis stage (structural PatternDAG construction).

Both candidate strategies inherited from the baseline are available unchanged:

1. Read the corpus as a flat list of segments (one per UD sentence, or one per line in
   plain-text mode).
2. Identify **candidate pairs** that are sufficiently similar to be worth comparing
   (either exhaustively or via approximate hashing).
3. For each retained pair, compute the **LCS** at the unit level, along with a rich set of
   similarity and coverage metrics.
4. Write all results to a TSV file, sorted by a chosen score.
5. Optionally, export **graphs** (JSON and/or Graphviz DOT) representing either the LCS
   paths factorised as a weighted automaton, or the inter-segment similarity graph.

---

## 2. Input Formats and Annotation Levels

### 2.1 Plain text

One segment per line; blank lines are ignored.  
**Note**: performing minimal pre-processing and token and separator normalization is advised, in order to boost MWU detection by reducing data sparsity.  Pre-processing steps should include: 

  - segmenting sentences (or longer stretches of text or annotations) into more manageable ``chunks''
    - example: using punctuation for basic chunking with `sed -r "s/ ?[[:punct:]] ?/\n/g"`; **warning** in this scenario, punctuations will be consumed during the chunking process
  - normalizing numbers, dates and other digits (e.g. `sed -r "s/[0-9]+/<Num>/g"`) if they are not absolutely necessary
  - converting all tokens to lower case (`tr '[:upper:]' '[:lower:]' < input.txt > output.txt`)
  - checking for pesky non-ASCII characters (French typographic apostrophe and guillemets, unbreakable spaces, etc.) and possibly normalizing these characters to their ASCII counterpart.

Parameters `--lowercase` and `normalize-num` are there to perform basic normalization for you, but no basic chunking nor non-ASCII character normalization. Moreover, these normalization steps are quite basic and might not work in the case of non-ASCII or very exotic characters.

The `seg_id` is the line number (1-based).
In plain-text mode, only the `token` annotation level is available (regex tokenisation,
`--token-pattern` option).

### 2.2 CoNLL-U (UD)

Tab-separated ten-column format conforming to the Universal Dependencies specification.
Each block of lines (separated by a blank line) constitutes one segment.  Comment lines
of the form `# key = value` are parsed as sentence-level metadata.

- **Retained tokens**: only "real" tokens (pure integer identifier, no `.` or `-`); multi-word
  tokens and empty nodes are ignored.
- **`sent_id`**: extracted from the `# sent_id = …` comment; if absent, a sequential counter
  is used.
- **`text`**: extracted from the `# text = …` comment; if absent, reconstructed by joining
  token forms with spaces.

**Automatic format detection** (`--input-format auto`, the default) identifies a CoNLL-U file
if tab characters and numeric token identifiers at the start of lines are found within the
first 4,000 characters.

### 2.3 UD annotation levels

The annotation level is set with `--annotation-level` and determines how each UD token is
converted into a comparison unit:

| Level | Unit produced | Example |
|-------|--------------|---------|
| `token` | surface form | `eats` |
| `lemma` | lemma | `eat` |
| `lemma_upos` | `lemma\|UPOS` | `eat\|VERB` |
| `token_lemma` | `form\|lemma` | `eats\|eat` |
| `token_lemma_upos` | `form\|lemma\|UPOS` | `eats\|eat\|VERB` |

All comparisons (Levenshtein, shingling, LCS) operate on these composite units, not on raw
character strings.  The `repr_1` / `repr_2` columns in the TSV output expose the effective
representation used for each segment, as space-separated units.

> **Note on missing values.**  If a UD field is `_`, the `lemma` column falls back to the
> surface form, and `upos` falls back to `X`.  Tokens with missing lemma or POS are therefore
> always included, in a degraded form.

---

## 3. Processing Pipeline

### 3.1 Exhaustive strategy

```
Input file (plain or CoNLL-U)
    │
    ▼
Parsing / SegmentRecord construction
    │   (UDSentence loading if CoNLL-U,
    │    regex tokenisation if plain)
    │   → units: Tuple[str, ...] according to --annotation-level
    ▼
Deduplication on repr_text  (unless --keep-duplicates)
    │
    ▼  for every pair (i, j), i < j
Unit-level length filter
    │   abs(len(u1) - len(u2)) / max_len ≥ threshold  → discard
    ▼
Bounded relative Levenshtein distance at the UNIT level
    │   rel_lev = edit_dist_units / max(len(u1), len(u2))
    │   pair kept if 0 < rel_lev < threshold
    ▼
Unit-level LCS computation
    │
    ▼
PairResult  ──►  TSV output (+ optional graph exports)
```
**Warning**: the exhaustive strategy has **0(n^2)** complexity at best (since we're examining 2 elements out of *n*).  The implementation is provided as a baseline. **The LSH strategy should be preferred**, as it dramatically optimizes processing time. Unless you have a very good reason to waste computing cycles. 
This strategy tries to minimize  the impact of quadratic complexity by distributing work across multiple processes (`multiprocessing.Pool`). Even so, expect to wait quite some time when processing ``large'' corpora (ca. 50 minutes for a 20,000 segments corpus). 
Each worker handles a contiguous slice of the segment list.  

**Key difference from the baseline**: the pre-selection filter now operates at the **unit** level (lemmas, pairs, or
triples depending on the chosen annotation level) rather than at the character level.  The
filter is therefore coherent with the representation used for LCS computation.

 **Note on the threshold parameter.**  `--threshold` is the *maximum* relative Levenshtein
 distance computed over units.  A value of `0.4` means "keep pairs that differ by at most
 40 % of the length of the longer unit sequence".  Identical segments (`rel_lev = 0.0`) are
 excluded; only pairs with `0 < rel_lev < threshold` are retained.

### 3.2 LSH strategy

Please use this strategy, unless you have a very good reason to use the ``exhaustive'' approach (e.g. testing purposes). 

```
Input file (plain or CoNLL-U)
    │
    ▼
Parsing / SegmentRecord construction
    │
    ▼
Deduplication on repr_text  (unless --keep-duplicates)
    │
    ▼
Shingling over UNITS
    │   Each segment → set of k-shingles (overlapping k-grams of units,
    │   not characters)
    ▼
MinHash signatures
    │   Each shingle set → vector of length num_perm
    │   Each component = minimum hash value over all shingles (Blake2b, salted)
    ▼
LSH band bucketing
    │   Signature split into `bands` sub-vectors of length (num_perm / bands)
    │   Pairs sharing at least one bucket → candidate pairs
    │   Effective Jaccard threshold displayed on stderr: t ≈ (1/bands)^(1/rows)
    ▼
Jaccard filter
    │   Estimated Jaccard  (from signatures)    ≥ min_est_jaccard
    │   Exact Jaccard      (from shingle sets)  ≥ min_exact_jaccard  [optional]
    ▼
Unit-level LCS computation
    │
    ▼
PairResult  ──►  TSV output (+ optional graph exports)
```

Shingles are formed from **units** (lemmas, pairs, etc.), not from surface tokens.  A
shingle of size 2 in `lemma_upos` mode is therefore a bigram of `lemma|UPOS` pairs, which
gives a morphosyntactic fingerprint of the segment.  The effective Jaccard threshold is
printed to `stderr` at startup.

---

## 4. Output Format

The primary output is a UTF-8 TSV file.  Each row represents one retained segment pair.
Column order follows the `DEFAULT_COLUMNS` list defined in the script.

### 4.1 Pair identification and provenance

| Column | Type | Description |
|--------|------|-------------|
| `pair_id` | string | `"i-j"` identifier for the pair. |
| `i` | int | Index of segment 1 in the loaded list (0-based). |
| `j` | int | Index of segment 2 (always `j > i`). |
| `segment_id_1` | string | Source identifier of segment 1: CoNLL-U `sent_id` or line number. |
| `segment_id_2` | string | Source identifier of segment 2. |
| `segment_1` | string | Raw text of segment 1 (CoNLL-U `text` field or raw input line). |
| `segment_2` | string | Raw text of segment 2. |
| `repr_1` | string | Effective representation of segment 1 after applying the annotation level (space-separated units). |
| `repr_2` | string | Effective representation of segment 2. |
| `annotation_level` | string | Annotation level used: `token`, `lemma`, `lemma_upos`, `token_lemma`, or `token_lemma_upos`. |
| `candidate_strategy` | string | `"exhaustive"` or `"lsh"`. |
| `mode` | string | Always `"token"` in this version. |

The `segment_id_1` / `segment_id_2` columns are new relative to the baseline and make it
possible to trace pairs back to their source sentences in the original CoNLL-U corpus.  The
`repr_1` / `repr_2` columns are essential for interpreting the LCS columns when a composite
annotation level is used.

### 4.2 Similarity measures

Both strategies populate `rel_lev` and `lev_dist` from units (no longer from characters as
in the baseline).  In LSH mode, `rel_lev` and `lev_dist` are not recomputed a posteriori on
retained candidates (unlike baseline v1.2): they remain `None` in LSH results.

| Column | Exhaustive | LSH | Description |
|--------|:----------:|:---:|-------------|
| `rel_lev` | ✓ | ✗ | Relative Levenshtein distance at the **unit** level: `edit_dist / max(len(u1), len(u2))`. |
| `lev_dist` | ✓ | ✗ | Raw edit distance on units. |
| `est_jaccard` | ✗ | ✓ | Estimated Jaccard similarity derived from MinHash signature agreement. |
| `exact_jaccard` | ✗ | ✓ | Exact Jaccard similarity computed from shingle sets: `|A ∩ B| / |A ∪ B|`. |
| `shared_bands` | ✗ | ✓ | Number of LSH bands in which the pair co-occurred in the same bucket. |
| `len_1_units` | ✓ | ✓ | Number of units in segment 1. |
| `len_2_units` | ✓ | ✓ | Number of units in segment 2. |
| `len_1_chars` | ✓ | ✓ | Character length of `repr_1` (length of the effective representation). |
| `len_2_chars` | ✓ | ✓ | Character length of `repr_2`. |
| `length_ratio` | ✓ | ✓ | `min(len_1_units, len_2_units) / max(len_1_units, len_2_units)`. |

### 4.3 LCS alignment view

The LCS is computed at the **unit** level using a full dynamic-programming algorithm.
Non-LCS units are replaced by `"-"` in the display columns.

| Column | Description |
|--------|-------------|
| `xx` | Segment 1 with non-LCS units replaced by `"-"`. |
| `yy` | Segment 2 with non-LCS units replaced by `"-"`. |
| `intersection` | Annotated LCS view: units at identical positions in both segments appear as-is; displaced units are marked `<MOV UNIT="…" ORIG="…" DEST="…">`. |
| `lcs_sequence` | Full LCS unit sequence in order, without positional annotation. |
| `lcs_length` | Total number of units in the LCS. |
| `lcs_cov_seg1` | LCS coverage of segment 1: `lcs_length / len_1_units`. |
| `lcs_cov_seg2` | LCS coverage of segment 2: `lcs_length / len_2_units`. |
| `lcs_cov_mean` | Mean of the two coverage values. |
| `lcs_cov_min` | Minimum of the two coverage values (conservative estimate). |
| `same_position_hits` | Count of LCS units at identical positions in both segments. |
| `moved_hits` | Count of LCS units at different positions (mobile elements). |

**Interpretation for MWU mining.** 

In `lemma_upos` mode, `lcs_sequence` directly yields an
abstract morphosyntactic pattern (e.g. `eat|VERB a|DET lot|NOUN of|ADP …`).
`same_position_hits` measures the positionally frozen core of the pattern; `moved_hits`
signals syntactically mobile constituents, which are valuable evidence for constructional
flexibility analysis.

**Example.**  For the pair *"he often eats green apples"* / *"he eats green apples"* in
`token` mode:

- `lcs_sequence` = `he eats green apples`
- `same_position_hits` = 2 (`he`, `eats`)
- `moved_hits` = 2 (`green`, `apples` slide one position to the left)
- `intersection` = `he eats <MOV UNIT="green" ORIG="3" DEST="2"> <MOV UNIT="apples" ORIG="4" DEST="3">`

### 4.4 Scores

| Column | Strategies | Description |
|--------|:----------:|-------------|
| `score_similarity` | both | Unified score: `max(est_jaccard, exact_jaccard, 1 - rel_lev)` over available non-null values. Provides a single sortable measure regardless of strategy. |
| `score_lcs_mean_minus_distance` | exhaustive only | `lcs_cov_mean - rel_lev`: rewards high LCS coverage and penalises surface distance. `None` in LSH mode (rel_lev not available). |

> **Difference from baseline v1.2.**  In the baseline, `score_similarity` was split into
> `score_similarity` (Jaccard only, LSH) and `score_lev_sim` (Levenshtein only, exhaustive).
> In this UD version, `score_similarity` is **unified**: it takes the best available score
> for the pair, making it possible to sort a mixed output file without having to select a
> strategy-specific column.

### 4.5 Graph exports

In addition to the main TSV, two types of graphs can be exported via `--graph-json` and/or
`--graph-dot`.

**Type `lcs_token_graph`** (default)  
A factorised weighted automaton built over LCS paths.  Nodes are **unit tokens** (surface
forms, lemmas, or composite units depending on the annotation level); edges are consecutive
transitions within LCS sequences.  Each node and each edge carries a frequency, corresponding
to the number of LCS paths that pass through it.  `__START__` and `__END__` nodes anchor the
graph.  This representation aggregates phraseological recurrences across all pairs: high-frequency
portions correspond to the frozen cores of MWU candidates.

The JSON payload contains:

- `nodes`: list of `{id, token, weight, special}`;
- `edges`: list of `{source, target, source_token, target_token, weight, examples}` (up to
  5 `pair_id` values per edge);
- `paths`: list of `{sequence, length, freq, examples}` for each distinct LCS sequence,
  sorted by descending frequency.

**Caveat**: LCS token graphs can become rather big (> 5 Mb) when converted to image format (png, svg, etc.) via `dot -Tpng input.dot -o output.png` Graphviz might complain that the graph is too big for common rendering algorithms. If you're out of luck, trying to visualize such a huge graph might even crash your session.  

**Type `similarity`**  
An inter-segment co-similarity graph.  Nodes are segments; edges connect retained pairs, with
scores as attributes.  Connected components are computed and exposed in the JSON, providing
a natural partition of the corpus into groups of phraseologically related sentences.

**Caveat**: similarity graphs can yield extremely wide, and thus unusable, graphs.

The JSON payload contains:

- `nodes`: list of `{id, segment_id, label, repr, n_units, component, meta}`;
- `edges`: list of `{source, target, pair_id, score_similarity, lcs_length, lcs_sequence, …}`;
- `components`: list of connected components (list of lists of segment indices);
- `lsh_buckets`: dictionary of non-singleton LSH buckets (LSH mode only).

The `--graph-min-lcs-length` option filters out paths that are too short before the LCS
graph is built.

---

## 5. Command-line Options

```
python ud_pair_miner_lsh.py <input> -o <o> [options]
```

### Positional

| Argument | Description |
|----------|-------------|
| `input` | Path to the input file: plain text (one segment per line) or CoNLL-U. |

### Output

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output` | *(required)* | Path to the output TSV file.  Parent directories are created automatically. |

### Input format and annotation

| Option | Default | Description |
|--------|---------|-------------|
| `--input-format` | `auto` | `"auto"` detects the format automatically.  `"plain"` forces plain-text mode.  `"conllu"` forces CoNLL-U mode. |
| `--annotation-level` | `token` | UD token representation used for comparison.  `token`, `lemma`, `lemma_upos`, `token_lemma`, or `token_lemma_upos`.  Ignored in plain-text mode (only `token` is available). |

### Strategy selection

| Option | Default | Description |
|--------|---------|-------------|
| `--candidate-strategy` | `exhaustive` | `"exhaustive"` evaluates all O(n²) pairs filtered by relative unit Levenshtein distance.  `"lsh"` uses MinHash + LSH for approximate candidate selection. |

### Shared options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--mode` | `token` | LCS mode.  Only `"token"` is available in this version. |
| `--token-pattern` | `\S+` | Python regex used to extract tokens in plain-text mode. |
| `--encoding` | `utf-8` | Input file encoding. |
| `--lowercase` | off | Lowercase surface forms and lemmas before comparison. |
| `--normalize-num` | off | Replace numeric tokens with `<NUM>` before comparison. |
| `--keep-duplicates` | off | Keep duplicate segments (default: deduplicate by effective representation).  Useful for spoken corpora with productive repetitions. |
| `--sort-by` | `score_similarity` | Column used to sort the output before writing.  Similarity/coverage columns sort descending; distance columns sort ascending. |
| `--limit` | `0` (disabled) | If > 0, truncate output to the top N rows after sorting. |

### Exhaustive strategy options

| Option | Default | Description |
|--------|---------|-------------|
| `-t`, `--threshold` | `0.4` | Maximum relative Levenshtein distance on **units**.  Must be in ]0, 1[. |
| `-w`, `--workers` | `cpu_count - 1` | Number of worker processes. |

### LSH strategy options

| Option | Default | Description |
|--------|---------|-------------|
| `--shingle-size` | `2` | Shingle size (k-grams of units).  Size 1 = bag of units; size 2 = unit bigrams. |
| `--num-perm` | `64` | Number of MinHash permutations.  Must be divisible by `--bands`. |
| `--bands` | `0` (auto) | Number of LSH bands.  `0` triggers `auto_bands`.  The effective Jaccard threshold `t ≈ (1/bands)^(1/rows)` is printed to `stderr` on startup. |
| `--min-est-jaccard` | `0.3` | Minimum estimated Jaccard (from MinHash signatures) to keep a candidate pair. |
| `--min-exact-jaccard` | `-1` (disabled) | Minimum exact Jaccard (from shingle sets).  `-1` disables this filter. |

### Graph export options

| Option | Default | Description |
|--------|---------|-------------|
| `--graph-json` | `None` | Path to the JSON graph export file (optional). |
| `--graph-dot` | `None` | Path to the Graphviz DOT export file (optional). |
| `--graph-kind` | `lcs_token_graph`, `similarity` | Type of graph to export.  `lcs_token_graph`: weighted automaton factorised over LCS paths.  `similarity`: inter-segment co-similarity graph with connected components. |
| `--graph-min-lcs-length` | `1` | Minimum LCS length (in units) included in graph exports. |

---

## 6. Usage Examples

**Note**: if the output TSV table only contains a few pairs, try lowering the similarity threshold (`-t`  and `--min-est-jaccard` parameters), and upping the number of bands and allowed permutations.

**Lemma mode on a parsed CoNLL-U file, exhaustive strategy:**

```bash
python ud_pair_miner_lsh.py corpus.conllu \
    -o results/lemma_exhaustive.tsv \
    --input-format conllu \
    --annotation-level lemma \
    --candidate-strategy exhaustive \
    --threshold 0.35 \
    --workers 4 \
    --sort-by lcs_cov_mean
```

**Lemma|UPOS mode, LSH, morphosyntactic bigrams:**

```bash
python ud_pair_miner_lsh.py corpus.conllu \

    -o results/lemma_upos_lsh.tsv \
    --input-format conllu \
    --annotation-level lemma_upos \
    --candidate-strategy lsh \
    --shingle-size 2 \
    --num-perm 128 \
    --bands 32 \
    --min-est-jaccard 0.3 \
    --sort-by score_similarity
```

**LSH with exact Jaccard post-filter and LCS graph export:**

```bash
python ud_pair_miner_lsh.py corpus.conllu \
    -o results/lsh_strict.tsv \
    --annotation-level lemma_upos \
    --candidate-strategy lsh \
    --num-perm 64 \
    --min-est-jaccard 0.3 \
    --min-exact-jaccard 0.4 \
    --limit 5000 \
    --graph-json results/lcs_graph.json \
    --graph-dot results/lcs_graph.dot \
    --graph-kind lcs_token_graph \
    --graph-min-lcs-length 2
```

**Similarity graph export (LSH, connected components):**

```bash
python ud_pair_miner_lsh.py corpus.conllu \
    -o results/pairs.tsv \
    --annotation-level lemma \
    --candidate-strategy lsh \
    --min-est-jaccard 0.25 \
    --graph-json results/sim_graph.json \
    --graph-kind similarity
```

**Numeric normalization and lowercasing (spoken corpus, token mode):**

```bash
python ud_pair_miner_lsh.py spoken.conllu \
    -o results/spoken.tsv \
    --annotation-level token \
    --lowercase \
    --normalize-num \
    --keep-duplicates \
    --candidate-strategy lsh \
    --min-est-jaccard 0.3
```

**Plain text without UD annotations (identical behaviour to the baseline):**

```bash
python ud_pair_miner_lsh.py corpus.txt \
    -o results/plain.tsv \
    --input-format plain \
    --annotation-level token \
    --candidate-strategy exhaustive \
    --threshold 0.4
```

**Checking the effective LSH Jaccard threshold before a run:**

The threshold is always printed to `stderr`:

```
[info] bands: 16
[info] shingle size: 2
[info] annotation level: lemma_upos
[info] effective Jaccard threshold ≈ 0.500
```

---

## 7. Known Limitations and Remaining Issues

### 7.1 Syntactic dependency information not used

Consistent with the purpose of this version (UD annotations *excluding* dependencies),
the `head`, `deprel`, and `deps` columns of CoNLL-U tokens are ignored.  Comparison is
therefore purely **positional and morphosyntactic**.  Pairs of structurally equivalent
segments with different constituent order may be penalised by the LCS score even though
their dependency structures would reveal functional identity.

### 7.2 `rel_lev` not available in LSH mode

Unlike baseline v1.2, which recomputes `rel_lev` a posteriori on retained LSH candidates,
this version leaves `rel_lev` and `lev_dist` as `None` for pairs produced by the LSH
strategy.  `score_lcs_mean_minus_distance` is therefore also `None` in LSH mode.  The
default sort column is `score_similarity` (Jaccard-based), which remains available.

### 7.3 Composite annotation levels and shingling

In `lemma_upos` or `token_lemma_upos` mode, shingles are k-grams of composite units.  A
shingle of size 2 in `token_lemma_upos` mode represents a pair of consecutive triples,
i.e. six implicit UD fields.  This makes the MinHash signature highly specific and can
reduce estimated Jaccard values on low-redundancy corpora.  Setting `--shingle-size` to 1
(bag of units) or increasing `--num-perm` mitigates this effect.

### 7.4 No aggregation into an MWU inventory

This script produces LCS candidates at the **pair** level; it does not aggregate LCS
sequences into a frequency-ranked MWU inventory.  The `paths` list in the `lcs_token_graph`
export provides partial aggregation (frequency of each distinct LCS sequence), but without
textometric normalisation or association measures.  Integration of an MWU density index or
structural PatternDAG construction is planned for future versions.

### 7.5 No intra-document provenance tracking

The input is a flat list of segments.  There is no option to record the position of each
sentence within its source document.  The `segment_id_1` / `segment_id_2` columns propagate
the CoNLL-U `sent_id`, but not the originating file name in the case of a multi-file corpus.

### 7.6 LSH pipeline not parallelised

The `--workers` option only affects the exhaustive strategy.  The LSH pipeline (shingling,
MinHash, band bucketing, scoring) runs in a single process.  On very large CoNLL-U corpora
(>100,000 sentences), the bottleneck is the candidate scoring phase.

### 7.7 `<MOV …>` markup in the `intersection` column

The `intersection` column still uses XML-like `<MOV UNIT="…" ORIG="…" DEST="…">` markup
for displaced units.  This format is not TSV-safe if units contain double-quote characters.
The clean separation into `lcs_stable` / `lcs_moved` introduced in baseline v1.2 has not
yet been back-ported to this version.

---

## 8. Roadmap

### Short term

- **Back-port `lcs_stable` / `lcs_moved` separation.**  Replace the `intersection` column
  with `<MOV …>` markup with two clean columns, consistent with baseline v1.2, to ensure
  a uniform output schema across both versions.

- **Recompute `rel_lev` in LSH mode.**  Port the baseline v1.2 behaviour: recompute
  unit-level Levenshtein distance for retained LSH candidates, so that `rel_lev`, `lev_dist`,
  and `score_lcs_mean_minus_distance` are populated in all modes.

- **Strategy-specific score columns.**  Introduce `score_lev_sim` (exhaustive) and keep
  `score_similarity` for Jaccard (LSH), as in baseline v1.2, to avoid misleading
  cross-strategy comparisons.

- **`--min-lcs-cov` option.**  LCS coverage threshold filter, absent from this version,
  to be aligned with the baseline.

### Medium term

- **UD morphological features (`feats`) integration.**  A `lemma_upos_feats` annotation
  level would include inflectional information (gender, number, tense, mood) in the
  comparison unit, enabling more precise morphosyntactic patterns on morphologically rich
  languages.

- **Multi-file provenance tracking.**  A `--corpus-map` option accepting a TSV file mapping
  `sent_id → document_path`, propagated to `doc_1` / `doc_2` columns in the output.

- **LSH parallelisation.**  Port the MinHash signature-building and candidate-scoring steps
  to `multiprocessing.Pool`, symmetrically with the exhaustive strategy.

- **LCS graph visualiser.**  An HTML/D3.js interface for interactive exploration of the
  `lcs_token_graph` output (frequency-threshold filtering, sequence search).

### Longer term

- **Structural PatternDAG integration.**  A `--dag-output` mode exploiting CoNLL-U
  dependency arcs to build a syntactic pattern DAG from the LCS sequences, complementing
  the token-level graph currently produced.

- **Textometric MWU density measure.**  An indicator combining `lcs_cov_mean` and the
  frequency of `lcs_sequence` across the corpus, usable as a phraseological specificity or
  productivity measure.

- **Streaming output.**  Write results to TSV incrementally rather than accumulating all
  `PairResult` objects in memory, enabling processing of very large corpora on
  memory-constrained machines.

---

## 9. Changelog

### v3  *(current)*

**Major new features:**

- **CoNLL-U / Universal Dependencies support.**  Full parsing of the ten-column format;
  extraction of `UDToken` and `UDSentence` objects; filtering of multi-word tokens and
  empty nodes (`is_real_ud_token`); retrieval of `sent_id` and `text` from sentence
  comments.
- **Automatic input format detection** (`--input-format auto`).
- **Five UD annotation levels** (`--annotation-level`): `token`, `lemma`, `lemma_upos`,
  `token_lemma`, `token_lemma_upos`.  All comparisons (Levenshtein, MinHash, LCS) operate
  on the composite units produced by the chosen level.
- **Input normalisation**: `--lowercase` (lowercase surface forms and lemmas) and
  `--normalize-num` (replace numeric tokens with `<NUM>`).
- **Graph exports**: `--graph-json` and `--graph-dot` with two graph kinds
  (`lcs_token_graph` and `similarity`).  The `lcs_token_graph` is a weighted automaton
  factorised over LCS paths; the `similarity` graph exposes corpus connected components.
- **New output columns**: `segment_id_1`, `segment_id_2`, `repr_1`, `repr_2`,
  `annotation_level`.

**Changes relative to baseline v1.2:**

- The Levenshtein distance in the exhaustive filter is now computed at the **unit level**
  (no longer at the character level), making it coherent with the effective representation
  used for LCS computation.
- `score_similarity` is **unified**: `max(est_jaccard, exact_jaccard, 1 - rel_lev)` over
  available values, replacing the `score_similarity` / `score_lev_sim` split of baseline
  v1.2.
- `--no-dedup` replaced by `--keep-duplicates` (inverted semantics, more explicit).
- `--min-lcs-cov` not yet ported to this version.
- `rel_lev` not recomputed a posteriori for LSH candidates.
- The `intersection` column retains `<MOV …>` markup (the `lcs_stable` / `lcs_moved`
  separation from baseline v1.2 has not yet been back-ported).
