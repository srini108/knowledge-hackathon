# Knowledge Hackathon — Implementation Notes

## 1. Overview

This submission implements a deterministic, snapshot-aware knowledge retrieval system in
`solution.py`. It answers questions using only the five website snapshots supplied with
the assignment. It does not make live website requests and does not require a hosted
language model, embedding service, API key, or network connection.

The implementation exposes the required interface:

- `build()` creates and returns a `Knowledge` instance.
- `ingest(snapshot)` adds a supplied website snapshot to the knowledge base.
- `answer(question, as_of=None)` answers from the appropriate snapshot.
- `conflicts()` returns detected differences in supported rate information.

## 2. Retrieval and answer strategy

During ingestion, each page is retained with its URL, title, snapshot identifier, crawl
date, text lines, token-frequency counts, and document length. Page-level token data is
precomputed so it does not need to be rebuilt for every question.

For retrieval, the system combines several lightweight techniques:

1. **Query normalization and expansion** — tokenizes the question and adds a small set
   of domain-specific alternatives, such as `CD`/`certificate`, `vehicle`/`auto`, and
   `yield`/`APY`.
2. **Page ranking** — uses a BM25-style term-frequency/inverse-document-frequency score,
   with additional weight for matching page titles and URLs.
3. **Product and page routing** — recognizes common product families and favors the
   corresponding official product or rate page.
4. **Evidence selection** — ranks relevant page lines and returns concise excerpts with
   source URLs and snapshot identifiers.

For current product-rate questions, official product and rate pages are given preference
over calculators, blog posts, and newsroom pages. Query expansion and ranking help find
relevant evidence; they do not authorize the system to invent facts that are absent from
the snapshots.

## 3. Snapshot selection and currentness

All five supplied snapshots are retained and ordered by their snapshot dates.

- **No `as_of` value:** `answer()` uses the latest available snapshot.
- **With `as_of`:** the system selects the latest snapshot whose date is on or before the
  requested date.
- **Date before the archive begins:** the system returns an unknown response rather than
  treating the oldest available snapshot as if it covered an earlier date.
- **Invalid `as_of` value:** the system returns an unknown response because it cannot
  reliably select a historical snapshot.

This keeps current answers tied to the newest supplied evidence while allowing
historical questions to be answered from the archive available at the requested time.
The system cannot establish facts for dates or events that are not represented in the
provided snapshots.

## 4. Qualifiers, uncertainty, and conflicts

The answer rules preserve distinctions that can materially change a financial answer,
including:

- APR versus APY or dividend rate;
- loan term and vehicle model year;
- balance tier, minimum balance, and loan amount;
- standard versus jumbo certificate products; and
- personal versus business products.

When the supplied evidence does not support a reliable answer, the system can abstain
instead of guessing.

The `conflicts()` method identifies selected percentage-bearing statements that change
across snapshots on recognized financial product and rate pages. It retains the
surrounding row qualifiers, records the competing values and their sources, and uses
the latest snapshot as the current resolution. It also checks a supported cross-page
auto-loan rate discrepancy, preferring the official product rate page over the
vehicle-affordability calculator for the overlapping model-year and term.

Conflict detection is a targeted audit of the supplied pages, not a guarantee that every
possible inconsistency in every page has been discovered.

## 5. Dependencies, runtime, and API keys

The retrieval and answer logic uses the Python standard library. The `contract` module
is supplied by the assignment harness and provides the required `Answer`, `Conflict`,
`Snapshot`, and `Source` types.

- **Python:** Python 3.9 or newer; Python 3.11 is a suitable runtime.
- **Third-party packages:** none are required by `solution.py`.
- **API keys:** none.
- **Network access:** none; the implementation uses the supplied snapshots only.

Keep the assignment's `harness` directory available when running the submission because
`solution.py` imports its `contract` module.

##6 Public evaluation

Run the **original public evaluation file** from the project root:

```bash
python harness/run_eval.py --submission starter --eval eval/public_eval.jsonl
```

Or evaluate the knowledge-system folder:

```bash
python harness/run_eval.py --submission knowledge-system --eval eval/public_eval.jsonl

---

**Implementation principle:** prefer evidence-backed answers from the correct snapshot;
preserve the qualifiers that define the fact; and abstain when the supplied archive
cannot support the answer.
