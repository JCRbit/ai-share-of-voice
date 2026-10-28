# Glossary

Plain definitions of the five terms used throughout the reports and the code, each tied
to where it is actually decided (not a general dictionary definition). See
[`docs/decisions.md`](decisions.md) decision 8 for the full mention-counting rules.

## Mention

A brand is mentioned when its name or one of its known variants (see
`data/competitors.csv`) appears in the answer as a whole word, or when a cited domain
matches a name or variant (for example `glovoapp.com` counts as a mention of Glovo).
Matching ignores case and accents (`normalize_brand_name`), and a cited URL that is not
a brand match is still listed separately as a source. A mention does **not** imply the
assistant is recommending the brand — see *Recommendation* below.

Example: if an assistant answers *"Puedes pedir en Glovo o en Just Eat"*, both Glovo and
Just Eat are mentioned; Uber Eats is not.

## Recommendation

Whether the answer clearly suggests, endorses or ranks the brand above alternatives —
not just mentions it in passing. `recommended` always implies `mentioned`, but a brand
can be mentioned without being recommended (`BrandMention.recommended`, in
`schemas/analysis.py`). This is the one judgement call the model makes on top of the
deterministic mention detection; code never infers a recommendation on its own.

Example: *"Glovo suele tener los mejores tiempos de entrega"* mentions and recommends
Glovo. *"En Madrid hay varias apps de delivery, como Glovo"* only mentions it.

## Position

The order in which a brand first appears in the answer, counting **only among the
tracked brands** (client plus the competitors in `data/competitors.csv`) — any other
brand or product the assistant mentions does not affect this count. Position is
computed deterministically from the raw text, independent of whether the brand is
recommended: a brand can appear first and still not be the one recommended.

Example: *"Prueba Uber Eats o Glovo"* gives Uber Eats position 1 and Glovo position 2,
even if the answer goes on to recommend Glovo.

## Tone

The sentiment of what the answer says about one already-mentioned brand: `positive`,
`neutral` or `negative` (`Tone` enum, `schemas/analysis.py`). Unlike mention and
position, tone is not decided by code — it needs judgement, so the model client
(`ModelClient.complete`) is asked for it, but only for brands already known to be
mentioned; if nothing is mentioned, no model call is made at all.

Example: *"Glovo me ha fallado varias veces este mes"* is a mention of Glovo with
negative tone, independent of whether it is also recommended.

## Confidence interval

The range the report shows next to a share (`ci_low`–`ci_high` in `BatchSummary` and in
the tracking CSV) is the **hull** of two different intervals computed over the same
batches of answers (`stats/metrics.py: brand_metrics`):

- a **Wilson score interval** on the raw proportion, which assumes every answer is an
  independent trial;
- a **cluster bootstrap** that resamples whole *questions* (not individual answers),
  because answers to the same question are not independent of each other.

The report always takes the widest low and the widest high of the two, so the interval
is never narrower than either method alone — Wilson alone would ignore the clustering,
and the bootstrap alone collapses to a single point when a brand is never or always
mentioned in the sample. This is a conservative reporting rule, not a statistically
calibrated interval, and runs with fewer than the minimum reliable number of questions
are flagged (`few_questions`) because the interval is too optimistic for them. The
report always states this is a dated sample, never "market share" (see decision 8 and
`automation/report.py`).
