"""DigestRenderer: renders the weekly HTML digest from classified paper data."""

import logging
from pathlib import Path

import jinja2

from src.models.graph import DigestEntry

logger = logging.getLogger(__name__)


class DigestRenderer:
    """Renders a list of DigestEntry objects into a weekly HTML digest file.

    Uses Jinja2 templates from the project's ``src/templates/`` directory.
    """

    def __init__(self, template_dir: Path | str) -> None:
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=True,
        )

    def render(
        self,
        entries: list[DigestEntry],
        run_date: str,
        output_dir: Path | str,
    ) -> Path:
        """Render the digest HTML and write it to ``output_dir/YYYY-MM-DD.html``.

        Args:
            entries: Classified paper entries with linked concepts.
            run_date: ISO date string (e.g. "2026-04-15").
            output_dir: Directory where the HTML file will be written.
                        Created if it does not exist.

        Returns:
            Path to the rendered HTML file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group entries by tier; classification failures land under key None
        tier_groups: dict[int | None, list[DigestEntry]] = {}
        for entry in entries:
            key = (
                None
                if entry.classification.classification_failed
                else entry.classification.tier
            )
            tier_groups.setdefault(key, []).append(entry)

        template = self._env.get_template("digest.html.j2")
        html = template.render(
            run_date=run_date,
            entries=entries,
            tier_groups=tier_groups,
            total_count=len(entries),
        )

        output_path = output_dir / f"{run_date}.html"
        output_path.write_text(html, encoding="utf-8")
        logger.info("Digest written to %s (%d entries)", output_path, len(entries))
        return output_path
