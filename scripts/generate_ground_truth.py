# generate_ground_truth.py
# One-off calibration script: uses Claude API (claude-sonnet-4-6) to generate
# 15 tree-based model concept notes in Markdown + YAML frontmatter format.
# Outputs to projects/paper-monitoring/seeds/tree_based_ground_truth/
# Run: python scripts/generate_ground_truth.py

from __future__ import annotations

import os
import re
import sys
import time
import logging
from pathlib import Path

# Load .env before importing anything that reads env vars
from dotenv import load_dotenv

# Resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
ENV_FILE = PROJECT_DIR / ".env"

load_dotenv(ENV_FILE)

import anthropic  # noqa: E402 (after dotenv load)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = PROJECT_DIR / "seeds" / "tree_based_ground_truth"
MODEL = "claude-sonnet-4-6"

CONCEPTS = [
    "Decision Tree",
    "Information Gain (ID3/C4.5)",
    "CART",
    "Bagging",
    "Random Forest",
    "Boosting",
    "AdaBoost",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Histogram-based Gradient Boosting",
    "Oblivious Trees",
    "Extra Trees",
    "Isolation Forest",
]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an ML educator producing high-quality structured knowledge notes for a concept graph database.
Each note you produce will become a node in a knowledge graph that data scientists use to draft
LinkedIn carousels and YouTube scripts about ML lineage.

Your notes must:
1. Be deeply technically accurate — no hand-waving.
2. Include specific, actionable content_angles (not generic; these drive editorial decisions).
3. Encode the full intellectual lineage in the relationships block.
4. Cover all 7 relationship types across the 15 concepts you generate:
   BUILDS_ON, ADDRESSES, ALTERNATIVE_TO, BASELINE_OF, PREREQUISITE_OF, INTRODUCES, BELONGS_TO

Quality bar for content_angles:
  Bad:  "Explain XGBoost to beginners"
  Good: "Why XGBoost won every Kaggle competition from 2015–2017: the regularization trick \
that gradient boosting missed"

Produce ONLY the raw Markdown (frontmatter + body). No preamble, no explanation, no code fences.
"""


def build_user_prompt(concept_name: str, all_concepts: list[str]) -> str:
    other = [c for c in all_concepts if c != concept_name]
    other_str = ", ".join(other)
    return f"""\
Generate a concept note for: **{concept_name}**

The full set of 15 tree-based model concepts being documented (potential relationship targets):
{other_str}

Output EXACTLY this structure — no deviations, no extra keys:

---
name: {concept_name}
concept_type: <one of: Algorithm, Technique, Framework, Concept, Mechanism>
what_it_is: <1-2 sentence definition>
what_problem_it_solves: <1-2 sentences on the problem it addresses>
innovation_chain:
  - step: <predecessor or foundational concept>
    why: <one sentence explaining the conceptual link to {concept_name}>
  - step: {concept_name}
    why: <one sentence on its specific innovation>
limitations:
  - <limitation 1 — be specific, not generic>
  - <limitation 2>
  - <limitation 3>
introduced_year: <year as integer — the year of the defining paper or formal introduction>
domain_tags:
  - <tag1>
  - <tag2>
source_refs:
  - <key paper or book, e.g. "Breiman 2001 - Random Forests">
content_angles:
  - <specific, actionable editorial framing for a LinkedIn post or YouTube video — not generic>
  - <second angle — different format or audience than the first>
  - <third angle — focus on a surprising insight, counterintuitive result, or historical moment>
relationships:
  - type: <one of: BUILDS_ON, ADDRESSES, ALTERNATIVE_TO, BASELINE_OF, PREREQUISITE_OF, INTRODUCES, BELONGS_TO>
    target: <name of another concept in the 15-concept set, or a foundational concept if essential>
    label: <one sentence explaining exactly how {concept_name} relates to the target>
  - type: <relationship type>
    target: <concept name>
    label: <one sentence>
---

## {concept_name}

<2-3 paragraphs of body prose. Include [[wikilinks]] to related concepts by wrapping them in double \
brackets. Explain the concept's place in the tree-based model lineage. Be specific about the \
mathematical or algorithmic innovation. Do not use filler phrases like "In conclusion" or "In summary".>

Rules for relationships:
- Include at least 2 relationships per concept.
- Use targets from the 15-concept set where possible.
- The relationship label must describe the *specific* connection (not "is related to").
- Across the full 15-note set, all 7 types must appear. For THIS concept, include whichever \
  types are most accurate — aim to include at least 2 distinct types per note.
- BUILDS_ON: this concept algorithmically or theoretically extends the target.
- ADDRESSES: this concept was designed to solve a problem exhibited by the target.
- ALTERNATIVE_TO: both solve similar problems via different mechanisms.
- BASELINE_OF: the target is commonly benchmarked against or compared to this concept.
- PREREQUISITE_OF: understanding this concept is required before understanding the target.
- INTRODUCES: this concept is the first/primary vehicle through which a mechanism was introduced.
- BELONGS_TO: this concept is a member of a family or class represented by the target.
"""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def snake_case(name: str) -> str:
    """Convert concept name to snake_case filename."""
    name = name.lower()
    name = re.sub(r"[/()\-]", " ", name)
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


def generate_concept_note(client: anthropic.Anthropic, concept_name: str) -> str:
    """Call Claude API and return the raw Markdown content."""
    logger.info("Generating: %s", concept_name)
    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": build_user_prompt(concept_name, CONCEPTS),
            }
        ],
    )
    return message.content[0].text


def count_relationships(content: str) -> int:
    """Count relationship entries in a note's YAML frontmatter."""
    return len(re.findall(r"^\s+- type:", content, re.MULTILINE))


def extract_relationship_types(content: str) -> set[str]:
    """Extract unique relationship type values from a note."""
    return set(re.findall(r"type:\s+(BUILDS_ON|ADDRESSES|ALTERNATIVE_TO|BASELINE_OF|PREREQUISITE_OF|INTRODUCES|BELONGS_TO)", content))


def extract_field(content: str, field: str) -> str:
    """Extract a scalar YAML field value."""
    m = re.search(rf"^{field}:\s*(.+)$", content, re.MULTILINE)
    return m.group(1).strip() if m else "?"


def count_content_angles(content: str) -> int:
    """Count content_angles list entries."""
    in_block = False
    count = 0
    for line in content.split("\n"):
        if re.match(r"^content_angles:", line):
            in_block = True
            continue
        if in_block:
            if re.match(r"^\s{2}-", line):
                count += 1
            elif re.match(r"^\S", line) and not re.match(r"^\s", line):
                break
    return count


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

REQUIRED_REL_TYPES = {
    "BUILDS_ON",
    "ADDRESSES",
    "ALTERNATIVE_TO",
    "BASELINE_OF",
    "PREREQUISITE_OF",
    "INTRODUCES",
    "BELONGS_TO",
}


def verify_outputs(output_dir: Path) -> dict:
    """Return verification summary dict."""
    files = sorted(output_dir.glob("*.md"))
    total_relationships = 0
    all_rel_types: set[str] = set()
    results = []

    for f in files:
        content = f.read_text()
        rel_count = count_relationships(content)
        rel_types = extract_relationship_types(content)
        angles = count_content_angles(content)
        concept_type = extract_field(content, "concept_type")
        year = extract_field(content, "introduced_year")
        total_relationships += rel_count
        all_rel_types.update(rel_types)
        results.append(
            {
                "file": f.name,
                "concept_type": concept_type,
                "introduced_year": year,
                "rel_count": rel_count,
                "rel_types": rel_types,
                "content_angles": angles,
            }
        )

    missing_types = REQUIRED_REL_TYPES - all_rel_types

    return {
        "file_count": len(files),
        "total_relationships": total_relationships,
        "all_rel_types": all_rel_types,
        "missing_rel_types": missing_types,
        "results": results,
    }


def print_summary_table(verification: dict) -> None:
    """Print a summary table of generated concepts."""
    print("\n" + "=" * 78)
    print(f"{'Concept File':<42} {'Type':<15} {'Year':<6} {'Rels':<5} {'Angles'}")
    print("-" * 78)
    for r in verification["results"]:
        name = r["file"].replace(".md", "")
        print(
            f"{name:<42} {r['concept_type']:<15} {r['introduced_year']:<6} "
            f"{r['rel_count']:<5} {r['content_angles']}"
        )
    print("=" * 78)
    print(f"\nTotal files      : {verification['file_count']} / 15")
    print(f"Total relationships: {verification['total_relationships']} (minimum 30 required)")
    print(f"Relationship types : {sorted(verification['all_rel_types'])}")
    if verification["missing_rel_types"]:
        print(f"MISSING types    : {sorted(verification['missing_rel_types'])} *** ACTION REQUIRED ***")
    else:
        print("All 7 relationship types covered: YES")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error(
            "ANTHROPIC_API_KEY not found. Add it to %s or export it in the shell.", ENV_FILE
        )
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", OUTPUT_DIR)

    client = anthropic.Anthropic(api_key=api_key)

    for i, concept in enumerate(CONCEPTS, start=1):
        filename = OUTPUT_DIR / f"{snake_case(concept)}.md"
        if filename.exists():
            logger.info("[%d/%d] Skipping (already exists): %s", i, len(CONCEPTS), filename.name)
            continue

        try:
            content = generate_concept_note(client, concept)
            filename.write_text(content, encoding="utf-8")
            rels = count_relationships(content)
            angles = count_content_angles(content)
            logger.info(
                "[%d/%d] Saved %s — %d relationships, %d content_angles",
                i,
                len(CONCEPTS),
                filename.name,
                rels,
                angles,
            )
        except anthropic.APIError as e:
            logger.error("[%d/%d] API error for %s: %s", i, len(CONCEPTS), concept, e)
            sys.exit(1)

        # Polite delay between API calls
        if i < len(CONCEPTS):
            time.sleep(1.0)

    logger.info("Generation complete. Running verification...")
    verification = verify_outputs(OUTPUT_DIR)
    print_summary_table(verification)

    # Exit with error if requirements not met
    issues = []
    if verification["file_count"] != 15:
        issues.append(f"Expected 15 files, got {verification['file_count']}")
    if verification["total_relationships"] < 30:
        issues.append(
            f"Expected >= 30 relationships, got {verification['total_relationships']}"
        )
    if verification["missing_rel_types"]:
        issues.append(f"Missing relationship types: {sorted(verification['missing_rel_types'])}")

    if issues:
        for issue in issues:
            logger.error("VERIFICATION FAILED: %s", issue)
        sys.exit(1)
    else:
        logger.info("All verification checks passed.")


if __name__ == "__main__":
    main()
