# Project decisions

Task C4. Record every decision the team closes before day 1. Update the status and the date when a decision changes; do not delete old ones.

| # | Decision | Status | Decided on | Owner |
| --- | --- | --- | --- | --- |
| 1 | Roles | Decided | 2026-10-23 | All |
| 2 | Demo sector and city | Decided | 2026-10-26 | All |
| 3 | Model, provider and spending cap | Decided | 2026-11-02 | All |
| 4 | Fit as a market study | Not applicable | 2026-10-26 | All |
| 5 | Cloudista case | Decided | 2026-10-26 | All |
| 6 | Python version | Decided | 2026-10-23 | All |
| 7 | Untracked brands in answers | Decided | 2026-11-02 | E, J |
| 8 | Mention counting rules | Decided | 2026-10-26 | E, J |

## 1. Roles

| Department | Person |
| --- | --- |
| P1 Data and Statistics | E |
| P2 Agents and Orchestration | J |
| P3 Integrations and Platform | An |
| P4 Product and Documentation | Ab |

PR review rotation: E reviews J, J reviews An, An reviews E, and E reviews Ab (mentor).

## 2. Demo sector and city

- **Proposal:** Food-delivery brands in one city.
- **Constraint:** The model must know the competitors; otherwise, almost all answers measure zero.
- **Decision:** Food delivery in Madrid, with Glovo as the client brand.
- **Sector:** Food delivery.
- **City:** Madrid.
- **Client:** Glovo.
- **Competitors:** Just Eat and Uber Eats (see `data/competitors.csv`).
- Name variants in `data/competitors.csv` are separated by `|`. Mention-counting rules are defined in decision 8.
- Questions never name a brand. The measure is whether the model brings a brand up on its own.
- **Primary language:** Spanish; English is also supported.
- Questions exist in Spanish (`data/questions_es.csv`) and English (`data/questions.csv`), paired by id and intent, so visibility can be compared across languages. The question loader should select the file from `Brief.language` (not implemented yet).
- Brands mentioned by the model outside this list are handled according to decision 7.

## 3. Model, provider and spending cap

- **Decision:** Use Groq as the provider and `openai/gpt-oss-120b` for both the subject model and the agent model.
- **Subject model:** `openai/gpt-oss-120b` (the model whose brand visibility is measured).
- **Agent model:** `openai/gpt-oss-120b` (the model used by the project's agents).
- **Provider:** Groq.
- **Cap per person (USD):** `$5.00`.
- **Cap per run (USD):** `$2.00`.

The same model is used for both roles during the initial project phase to simplify configuration, reproducibility, and evaluation. The per-run cap limits spending on a single execution; the per-person cap limits cumulative individual spending during the project. These are project budget limits, not provider-enforced limits. Actual costs depend on usage and the provider's current pricing.

## 4. Fit as a market study

- **Question for the training team:** Does AI-visibility monitoring fit the “market research” brief?
- **Asked on:** N/A.
- **If it does not fit:** N/A.
- **Status:** Not applicable; no further action recorded.

## 5. Cloudista case

- **Agreed position:** Cloudista's sector (food delivery) is used for the demo, and its ethical case (not delegating allergen checks to vision) is presented as an example of a decision that is not delegated. It is not the product.
- **Confirmed with whoever proposed it:** Done.

## 6. Python version

- CrewAI 1.x requires Python below 3.14 and Pydantic below 2.13.
- **Decision:** Python 3.12 for everyone (Python 3.13 also works); Pydantic is pinned below 2.13.

## 7. Untracked brands in answers

- **Problem:** Answers can recommend brands that are not in `data/competitors.csv`. Ignoring them hides real competitors.
- **Decision:** The extractor records untracked brands per answer. The report ranks them by mention rate with an interval and suggests reviewing frequent ones for addition to the competitor list in a subsequent run.
- **Threshold to suggest a new competitor:** The brand must appear in at least 10% of all valid sampled answers. Each answer counts at most once per brand, regardless of how many times the brand is mentioned in that answer.
- **Validation condition:** The brand must appear in at least two different batches.
- **Action:** Flag the brand for human review. Do not automatically add it to `data/competitors.csv`.
- **Implementation note:** This requires a change to `AnalyzedResponse` in `schemas/`; it must be implemented in a separate PR, with the team notified first.

The threshold is intended to identify potentially relevant emerging competitors while reducing false positives from isolated mentions. Inclusion in the tracked competitor set remains a human decision.

## 8. Mention counting rules

- **Context:** The mention extractor (#46) decides mentions with deterministic code. These rules make its behaviour explicit so the metrics can be interpreted consistently.
- A brand is mentioned when its name or one of its variants appears as a whole word. Case and accents are ignored (`normalize_brand_name`), so “Glóvo” counts for Glovo. This normalization also turns “ñ” into “n”.
- A cited domain that matches a name or variant counts as a mention (for example, `glovoapp.com` counts for Glovo): a cited brand is visible to the reader. Cited URLs are also listed separately as sources.
- Variants must not be ambiguous: “Uber” alone is not a variant of Uber Eats, because it would count ride-hailing answers as mentions.
- Position is the order of first appearance among tracked brands only, independent of recommendation; read it together with `recommended`.
