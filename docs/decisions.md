# Project decisions

Task C4. Record every decision the team closes before day 1. Update the status and the date when a decision changes; do not delete old ones.

| # | Decision | Status | Decided on | Owner |
| --- | --- | --- | --- | --- |
| 1 | Roles | Decided | 2026-10-23 | All |
| 2 | Demo sector and city | Decided | 2026-10-26 | All |
| 3 | Model, provider and spending cap | Pending | | |
| 4 | Fit as a market study | Pending | | |
| 5 | Cloudista case | Pending | | |
| 6 | Python version | Decided | 2026-10-23 | All |
| 7 | Untracked brands in answers | Proposed | 2026-10-26 | E, J |

## 1. Roles

| Department | Person |
| --- | --- |
| P1 Data and Statistics | E |
| P2 Agents and Orchestration | J |
| P3 Integrations and Platform | An |
| P4 Product and Documentation | Ab |

PR review rotation: E reviews J, J reviews An, An reviews E, and E reviews Ab (mentor).

## 2. Demo sector and city

- Proposal: food-delivery brands in one city.
- Constraint: the model must know the competitors, otherwise almost all answers measure zero.
- Decision: food delivery in Madrid, with Glovo as the client brand.
- Sector: food delivery · City: Madrid · Client: Glovo · Competitors: Just Eat, Uber Eats (see `data/competitors.csv`)
- Name variants in `data/competitors.csv` are separated by `|`. "Uber" alone is not a variant: it would count ride-hailing answers as mentions.
- Questions never name a brand, so the measure is whether the model brings a brand up on its own.
- Primary language: Spanish; English is also supported.
- Questions exist in Spanish (`data/questions_es.csv`) and English (`data/questions.csv`), paired by id and intent, so visibility can be compared across languages. The question loader should pick the file from `Brief.language` (not implemented yet).
- Brands the model mentions outside this list: see decision 7.

## 3. Model, provider and spending cap

- Subject model (the one measured) and agent model (the one that runs the agents) may differ.
- Decision: _pending_
- Subject model: _pending_ · Agent model: _pending_ · Provider: _pending_
- Cap per person (USD): _pending_ · Cap per run (USD): _pending_

## 4. Fit as a market study

- Question for the training team: does AI-visibility monitoring fit the "market research" brief?
- Asked on: _pending_ · Answer: _pending_
- If it does not fit: _pending_

## 5. Cloudista case

- Agreed position: its sector (food delivery) is used for the demo, and its ethical case (not delegating allergen checks to vision) is presented as an example of a decision that is not delegated. It is not the product.
- Confirmed with whoever proposed it: _pending_

## 6. Python version

- CrewAI 1.x requires Python below 3.14 and pydantic below 2.13.
- Decision: Python 3.12 for everyone (3.13 also works); pydantic pinned below 2.13.

## 7. Untracked brands in answers

- Problem: answers can recommend brands that are not in `data/competitors.csv`. Ignoring them hides real competitors.
- Proposal: the extractor records untracked brands per answer; the report ranks them by mention rate with an interval and suggests adding frequent ones as competitors in the next run.
- Needs a change to `AnalyzedResponse` in `schemas/`, so it goes in its own PR and the team is told first.
- Decision: _pending_ · Threshold to suggest a new competitor: _pending_
