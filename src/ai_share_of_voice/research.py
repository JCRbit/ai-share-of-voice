"""Source research through the MCP sources server (T5, local integration, E).

No model is involved: code decides what to search and what to keep, and everything read is
stored as untrusted data for the writer and the Skeptic.

1. Search the sector in the city and the client brand in the city.
2. Read the first pages that robots.txt allows; a page that cannot be read is skipped and
   recorded, not fatal. Each document is saved with its injection flags and prompt block.
3. Check the most named untracked businesses (from discovery) for evidence online: a search
   result that names the business. ``found_online=False`` means "no evidence found", never
   "does not exist": the measured model may name real shops that the search misses.

A failed search is a ``ToolFailure``: the guard closes the run in a partial state.
"""

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ai_share_of_voice.discovery import BusinessKind
from ai_share_of_voice.errors import ToolFailure
from ai_share_of_voice.schemas.brief import Brief, normalize_brand_name
from ai_share_of_voice.tools.mcp_sources_client import McpSourcesClient

SOURCES_FOLDER = "sources"
RESEARCH_FILE_NAME = "research.json"
RANKING_FILE_NAME = "discovered_businesses.csv"  # written by the metrics node
HITS_PER_QUERY = 5
MAX_READS = 4
MAX_CHECKS = 3
HITS_PER_CHECK = 3


@dataclass(frozen=True)
class BusinessCheck:
    name: str
    rate: float
    found_online: bool
    evidence_url: str | None = None


@dataclass
class ResearchResult:
    queries: list[str]
    sources: list[str] = field(default_factory=list)  # URLs read and stored
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (URL, reason)
    flagged: int = 0  # documents with instruction-like lines
    checks: list[BusinessCheck] = field(default_factory=list)


def _top_businesses(run_dir: Path, limit: int) -> list[tuple[str, float]]:
    """Most named untracked businesses that can be looked up (products are not)."""
    path = run_dir / RANKING_FILE_NAME
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["kind"] != BusinessKind.PRODUCT_OR_PLAN
        ]
    rows.sort(key=lambda row: -float(row["rate"]))
    return [(row["name"], float(row["rate"])) for row in rows[:limit]]


def _place_terms(place: str) -> list[str]:
    """ "Valencia, España" -> ["valencia", "espana"]: any of them ties a result to the place."""
    return [normalize_brand_name(part.strip()) for part in place.split(",") if part.strip()]


def _check(client: McpSourcesClient, name: str, rate: float, place: str) -> BusinessCheck:
    """Evidence is a result that names the business and the city or the country.

    The name alone is not enough: a live run took Deliveroo's generic page as evidence in
    Madrid, where it does not operate. The address counts as text ("/es/valencia").
    """
    key = normalize_brand_name(name)
    terms = _place_terms(place)
    for hit in client.search_sources(f'"{name}" {place}', max_results=HITS_PER_CHECK):
        text = normalize_brand_name(f"{hit.title} {hit.snippet}")
        where = f"{text} {normalize_brand_name(str(hit.url))}"
        if key in text and any(term in where for term in terms):
            return BusinessCheck(name, rate, found_online=True, evidence_url=str(hit.url))
    return BusinessCheck(name, rate, found_online=False)


def research(
    brief: Brief,
    run_dir: Path,
    client: McpSourcesClient,
    *,
    max_reads: int = MAX_READS,
    max_checks: int = MAX_CHECKS,
    location: str | None = None,
) -> ResearchResult:
    """``location`` (e.g. "Valencia, España") replaces the bare city in every query: a city
    name alone can be ambiguous, and a live run got Valencia, California for Valencia."""
    place = location or brief.city
    queries = [
        f"{brief.sector} {place}",
        f"{brief.brand.name} {place} {brief.sector}",
    ]
    result = ResearchResult(queries=queries)

    seen: set[str] = set()
    hits = []
    for query in queries:
        for hit in client.search_sources(query, max_results=HITS_PER_QUERY):
            if str(hit.url) not in seen:
                seen.add(str(hit.url))
                hits.append(hit)

    folder = run_dir / SOURCES_FOLDER
    for hit in hits:
        if len(result.sources) == max_reads:
            break
        url = str(hit.url)
        try:
            read = client.read_source(url)
        except ToolFailure as error:
            result.skipped.append((url, error.reason))
            continue
        result.sources.append(url)
        result.flagged += bool(read.injection_flags)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{len(result.sources):02d}.json").write_text(
            json.dumps(
                {
                    "document": read.document.model_dump(mode="json"),
                    "injection_flags": list(read.injection_flags),
                    "prompt_block": read.prompt_block,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    result.checks = [
        _check(client, name, rate, place) for name, rate in _top_businesses(run_dir, max_checks)
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RESEARCH_FILE_NAME).write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
