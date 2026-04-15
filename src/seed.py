"""Knowledge bank seeding CLI entry point.

Usage:
    python -m src.seed                        # Full seeding run
    python -m src.seed --arxiv-id 1706.03762  # Seed a single paper
    python -m src.seed --only-papers          # Papers only (skip textbooks)
    python -m src.seed --only-textbooks       # Textbooks only (skip papers)
    python -m src.seed --dry-run              # Log plan without writing
"""
import argparse
import logging
from pathlib import Path

from src.config import settings
from src.integrations.arxiv_client import ArxivFetcher
from src.integrations.pdf_extractor import PdfExtractor
from src.services.classifier import OllamaClassifier
from src.services.seeder import Seeder
from src.store.graph_store import GraphStore
from src.utils.logging_config import setup_logging

# ---------------------------------------------------------------------------
# Landmark paper arXiv IDs (from DATA-SOURCES.md)
# ---------------------------------------------------------------------------

LANDMARK_PAPER_IDS = [
    "1706.03762",  # Attention Is All You Need (Transformer)
    "1810.04805",  # BERT
    "2005.14165",  # GPT-3
    "1512.03385",  # ResNet
    "1502.03167",  # Batch Normalization
    "1301.3781",   # Word2Vec
    "1406.2661",   # GAN
    "1312.6114",   # VAE
    "2006.11239",  # DDPM
    "2106.09685",  # LoRA
    "1707.06347",  # PPO
    "2307.09288",  # Llama 2
    "2312.00752",  # Mamba
    "2205.14135",  # FlashAttention
    "2103.00020",  # CLIP
    "2307.01852",  # RLHF Survey
    "2001.08361",  # Scaling Laws
]

SURVEY_PAPER_IDS = [
    "2303.18223",  # A Survey of Large Language Models
    "2010.11929",  # ViT (An Image is Worth 16x16 Words)
    "1701.07274",  # Deep Reinforcement Learning: An Overview
    "1901.00596",  # A Comprehensive Survey on Graph Neural Networks
    "1602.05629",  # Communication-Efficient Learning (Federated Learning)
    "2209.03430",  # Multimodal Machine Learning: A Survey and Taxonomy
    "2106.08962",  # Efficient Deep Learning: A Survey
    "1802.01528",  # A Survey of Methods for Explaining Black Box Models (XAI)
    "2107.13586",  # Pre-train, Prompt, and Predict
]

# ---------------------------------------------------------------------------
# Textbook chapter configuration
# TODO: Update page ranges after downloading PDFs to data/textbooks/.
#       Ranges are 0-indexed (start_page, end_page, source_description).
#       Current ranges are approximate placeholders.
# ---------------------------------------------------------------------------

_PDF_DIR = Path(__file__).parent.parent / "data" / "textbooks"

TEXTBOOK_CONFIGS = [
    (
        _PDF_DIR / "goodfellow_deep_learning.pdf",
        [
            (14, 55, "Goodfellow et al. Deep Learning, Ch. 2: Linear Algebra"),
            (56, 97, "Goodfellow et al. Deep Learning, Ch. 3: Probability"),
            (98, 139, "Goodfellow et al. Deep Learning, Ch. 4: Numerical Computation"),
            (140, 219, "Goodfellow et al. Deep Learning, Ch. 5: Machine Learning Basics"),
            (220, 281, "Goodfellow et al. Deep Learning, Ch. 6: Deep Feedforward Networks"),
        ],
    ),
    (
        _PDF_DIR / "murphy_pml_intro.pdf",
        [
            (49, 100, "Murphy PML Introduction, Ch. 2: Probability"),
            (101, 160, "Murphy PML Introduction, Ch. 3: Probability — Multivariate Models"),
            (400, 460, "Murphy PML Introduction, Ch. 10: Linear Discriminant Analysis"),
        ],
    ),
    (
        _PDF_DIR / "hastie_esl.pdf",
        [
            (9, 42, "Hastie et al. ESL, Ch. 2: Overview of Supervised Learning"),
            (43, 82, "Hastie et al. ESL, Ch. 3: Linear Methods for Regression"),
            (194, 228, "Hastie et al. ESL, Ch. 7: Model Assessment and Selection"),
        ],
    ),
    (
        _PDF_DIR / "sutton_barto_rl.pdf",
        [
            (47, 90, "Sutton & Barto RL, Ch. 3: Finite Markov Decision Processes"),
            (91, 130, "Sutton & Barto RL, Ch. 4: Dynamic Programming"),
            (265, 320, "Sutton & Barto RL, Ch. 13: Policy Gradient Methods"),
        ],
    ),
]


def _build_seeder(cfg=None) -> Seeder:
    """Wire up dependencies for the Seeder."""
    cfg = cfg or settings
    store = GraphStore(cfg.db_path)
    return Seeder(
        store=store,
        arxiv_fetcher=ArxivFetcher(cfg=cfg),
        pdf_extractor=PdfExtractor(),
        classifier=OllamaClassifier(cfg=cfg),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the paper monitoring knowledge bank."
    )
    parser.add_argument(
        "--arxiv-id",
        type=str,
        help="Seed a single paper by arXiv ID (e.g. 1706.03762).",
    )
    parser.add_argument(
        "--only-papers",
        action="store_true",
        help="Seed papers only; skip textbook chapters.",
    )
    parser.add_argument(
        "--only-textbooks",
        action="store_true",
        help="Seed textbook chapters only; skip papers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be seeded without writing to the database.",
    )
    args = parser.parse_args()

    setup_logging(log_dir=settings.log_dir, log_level=settings.log_level)
    logger = logging.getLogger(__name__)

    if args.dry_run:
        logger.info(
            "DRY RUN — would seed %d landmark papers, %d survey papers, %d textbook configs",
            len(LANDMARK_PAPER_IDS),
            len(SURVEY_PAPER_IDS),
            len(TEXTBOOK_CONFIGS),
        )
        return

    if args.arxiv_id:
        seeder = _build_seeder()
        seeder.seed_paper_by_id(args.arxiv_id)
        return

    seed_papers = not args.only_textbooks
    seed_textbooks = not args.only_papers

    seeder = _build_seeder()
    summary = seeder.seed_all(
        landmark_ids=LANDMARK_PAPER_IDS,
        survey_ids=SURVEY_PAPER_IDS,
        textbook_configs=TEXTBOOK_CONFIGS,
        seed_papers=seed_papers,
        seed_textbooks=seed_textbooks,
    )
    logger.info("Seeding summary: %s", summary)


if __name__ == "__main__":
    main()
