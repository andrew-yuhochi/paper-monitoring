"""Knowledge bank seeding CLI entry point.

Usage:
    python -m src.seed                                          # Full seeding run
    python -m src.seed --arxiv-id 1706.03762                   # Seed a single paper
    python -m src.seed --only-papers                            # Papers only (skip textbooks)
    python -m src.seed --only-textbooks                         # Textbooks only (skip papers)
    python -m src.seed --dry-run                                # Log plan without writing
    python -m src.seed --ground-truth seeds/tree_based_ground_truth/  # Load hand-crafted concepts
"""
import argparse
import logging
from pathlib import Path

from src.config import settings
from src.integrations.arxiv_client import ArxivFetcher
from src.integrations.pdf_extractor import PdfExtractor
from src.services.classifier import OllamaClassifier
from src.services.ground_truth_loader import load_ground_truth
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
# Page ranges are 0-indexed (start_page, end_page inclusive, source_description).
# Derived from each PDF's table of contents via fitz.get_toc() (1-indexed) minus 1.
# Skipped: prefaces, indices, bibliographies, notation appendices, pure
#   implementation/tooling sections, and non-ML sections (Sutton Psychology/Neuroscience).
# ---------------------------------------------------------------------------

_PDF_DIR = Path(__file__).parent.parent / "data" / "textbooks"

TEXTBOOK_CONFIGS = [
    # ── Murphy, Probabilistic Machine Learning: An Introduction (860 pp) ──
    (
        _PDF_DIR / "murphy_pml_intro.pdf",
        [
            (30, 61, "Murphy PML, Ch. 1: Introduction"),
            (62, 105, "Murphy PML, Ch. 2: Probability — Univariate Models"),
            (106, 135, "Murphy PML, Ch. 3: Probability — Multivariate Models"),
            (136, 195, "Murphy PML, Ch. 4: Statistics"),
            (196, 235, "Murphy PML, Ch. 5: Decision Theory"),
            (236, 257, "Murphy PML, Ch. 6: Information Theory"),
            (258, 303, "Murphy PML, Ch. 7: Linear Algebra"),
            (304, 351, "Murphy PML, Ch. 8: Optimization"),
            (352, 367, "Murphy PML, Ch. 9: Linear Discriminant Analysis"),
            (368, 399, "Murphy PML, Ch. 10: Logistic Regression"),
            (400, 443, "Murphy PML, Ch. 11: Linear Regression"),
            (444, 453, "Murphy PML, Ch. 12: Generalized Linear Models"),
            (454, 495, "Murphy PML, Ch. 13: Neural Networks for Tabular Data"),
            (496, 531, "Murphy PML, Ch. 14: Neural Networks for Images"),
            (532, 575, "Murphy PML, Ch. 15: Neural Networks for Sequences"),
            (576, 595, "Murphy PML, Ch. 16: Exemplar-based Methods"),
            (596, 631, "Murphy PML, Ch. 17: Kernel Methods"),
            (632, 655, "Murphy PML, Ch. 18: Trees, Forests, Bagging, and Boosting"),
            (656, 685, "Murphy PML, Ch. 19: Learning with Fewer Labeled Examples"),
            (686, 743, "Murphy PML, Ch. 20: Dimensionality Reduction"),
            (744, 769, "Murphy PML, Ch. 21: Clustering"),
            (770, 781, "Murphy PML, Ch. 22: Recommender Systems"),
            (782, 801, "Murphy PML, Ch. 23: Graph Embeddings"),
        ],
    ),
    # ── Hastie, Tibshirani, Friedman — Elements of Statistical Learning (764 pp) ──
    (
        _PDF_DIR / "hastie_esl.pdf",
        [
            (27, 60, "Hastie ESL, Ch. 2: Overview of Supervised Learning"),
            (61, 118, "Hastie ESL, Ch. 3: Linear Methods for Regression"),
            (119, 156, "Hastie ESL, Ch. 4: Linear Methods for Classification"),
            (157, 208, "Hastie ESL, Ch. 5: Basis Expansions and Regularization"),
            (209, 236, "Hastie ESL, Ch. 6: Kernel Smoothing Methods"),
            (237, 278, "Hastie ESL, Ch. 7: Model Assessment and Selection"),
            (279, 312, "Hastie ESL, Ch. 8: Model Inference and Averaging"),
            (313, 354, "Hastie ESL, Ch. 9: Additive Models, Trees, and Related Methods"),
            (355, 406, "Hastie ESL, Ch. 10: Boosting and Additive Trees"),
            (407, 434, "Hastie ESL, Ch. 11: Neural Networks"),
            (435, 476, "Hastie ESL, Ch. 12: Support Vector Machines and Flexible Discriminants"),
            (477, 502, "Hastie ESL, Ch. 13: Prototype Methods and Nearest-Neighbors"),
            (503, 604, "Hastie ESL, Ch. 14: Unsupervised Learning"),
            (605, 622, "Hastie ESL, Ch. 15: Random Forests"),
            (623, 642, "Hastie ESL, Ch. 16: Ensemble Learning"),
            (643, 666, "Hastie ESL, Ch. 17: Undirected Graphical Models"),
            (667, 716, "Hastie ESL, Ch. 18: High-Dimensional Problems"),
        ],
    ),
    # ── Bishop — Pattern Recognition and Machine Learning (758 pp) ──
    (
        _PDF_DIR / "bishop_prml.pdf",
        [
            (20, 85, "Bishop PRML, Ch. 1: Introduction"),
            (86, 155, "Bishop PRML, Ch. 2: Probability Distributions"),
            (156, 197, "Bishop PRML, Ch. 3: Linear Models for Regression"),
            (198, 243, "Bishop PRML, Ch. 4: Linear Models for Classification"),
            (244, 309, "Bishop PRML, Ch. 5: Neural Networks"),
            (310, 343, "Bishop PRML, Ch. 6: Kernel Methods"),
            (344, 377, "Bishop PRML, Ch. 7: Sparse Kernel Machines"),
            (378, 441, "Bishop PRML, Ch. 8: Graphical Models"),
            (442, 479, "Bishop PRML, Ch. 9: Mixture Models and EM"),
            (480, 541, "Bishop PRML, Ch. 10: Approximate Inference"),
            (542, 577, "Bishop PRML, Ch. 11: Sampling Methods"),
            (578, 623, "Bishop PRML, Ch. 12: Continuous Latent Variables"),
            (624, 671, "Bishop PRML, Ch. 13: Sequential Data"),
            (672, 695, "Bishop PRML, Ch. 14: Combining Models"),
        ],
    ),
    # ── Sutton & Barto — Reinforcement Learning: An Introduction (548 pp) ──
    (
        _PDF_DIR / "sutton_barto_rl.pdf",
        [
            (22, 45, "Sutton RL, Ch. 1: Introduction"),
            (46, 67, "Sutton RL, Ch. 2: Multi-armed Bandits"),
            (68, 93, "Sutton RL, Ch. 3: Finite Markov Decision Processes"),
            (94, 111, "Sutton RL, Ch. 4: Dynamic Programming"),
            (112, 139, "Sutton RL, Ch. 5: Monte Carlo Methods"),
            (140, 161, "Sutton RL, Ch. 6: Temporal-Difference Learning"),
            (162, 179, "Sutton RL, Ch. 7: n-step Bootstrapping"),
            (180, 217, "Sutton RL, Ch. 8: Planning and Learning with Tabular Methods"),
            (218, 263, "Sutton RL, Ch. 9: On-policy Prediction with Approximation"),
            (264, 277, "Sutton RL, Ch. 10: On-policy Control with Approximation"),
            (278, 307, "Sutton RL, Ch. 11: Off-policy Methods with Approximation"),
            (308, 341, "Sutton RL, Ch. 12: Eligibility Traces"),
            (342, 361, "Sutton RL, Ch. 13: Policy Gradient Methods"),
            (442, 479, "Sutton RL, Ch. 16: Applications and Case Studies"),
            (480, 501, "Sutton RL, Ch. 17: Frontiers"),
        ],
    ),
    # ── Zhang et al. — Dive into Deep Learning (1151 pp, Goodfellow substitute) ──
    (
        _PDF_DIR / "zhang_d2l.pdf",
        [
            (40, 68, "D2L, Ch. 1: Introduction"),
            (69, 120, "D2L, Ch. 2: Preliminaries"),
            (121, 163, "D2L, Ch. 3: Linear Neural Networks for Regression"),
            (164, 205, "D2L, Ch. 4: Linear Neural Networks for Classification"),
            (206, 245, "D2L, Ch. 5: Multilayer Perceptrons"),
            (272, 306, "D2L, Ch. 7: Convolutional Neural Networks"),
            (307, 363, "D2L, Ch. 8: Modern Convolutional Neural Networks"),
            (364, 407, "D2L, Ch. 9: Recurrent Neural Networks"),
            (408, 447, "D2L, Ch. 10: Modern Recurrent Neural Networks"),
            (448, 506, "D2L, Ch. 11: Attention Mechanisms and Transformers"),
            (507, 585, "D2L, Ch. 12: Optimization Algorithms"),
            (631, 728, "D2L, Ch. 14: Computer Vision"),
            (729, 782, "D2L, Ch. 15: NLP Pretraining"),
            (783, 819, "D2L, Ch. 16: NLP Applications"),
            (820, 835, "D2L, Ch. 17: Reinforcement Learning"),
            (836, 866, "D2L, Ch. 18: Gaussian Processes"),
            (919, 931, "D2L, Ch. 20: Generative Adversarial Networks"),
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
    parser.add_argument(
        "--ground-truth",
        type=str,
        metavar="DIR",
        help="Load hand-crafted concept Markdown files from DIR into the concept-first schema.",
    )
    args = parser.parse_args()

    setup_logging(log_dir=settings.log_dir, log_level=settings.log_level)
    logger = logging.getLogger(__name__)

    # --ground-truth short-circuits all legacy seeder logic.
    if args.ground_truth:
        gt_dir = Path(args.ground_truth)
        if not gt_dir.is_dir():
            logger.error("--ground-truth directory does not exist: %s", gt_dir)
            raise SystemExit(1)
        store = GraphStore(settings.db_path)
        concepts_n, rels_n = load_ground_truth(gt_dir, store)
        logger.info(
            "Ground truth seeding complete: %d concepts, %d relationships loaded.",
            concepts_n,
            rels_n,
        )
        return

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
