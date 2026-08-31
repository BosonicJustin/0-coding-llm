from __future__ import annotations

import re
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs"
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)")


class DocumentationStructureTest(unittest.TestCase):
    def test_readme_is_the_only_root_markdown_file(self) -> None:
        root_markdown = sorted(path.name for path in PROJECT_ROOT.glob("*.md"))
        self.assertEqual(root_markdown, ["README.md"])

    def test_documentation_index_links_every_central_document(self) -> None:
        index = (DOCS_ROOT / "README.md").read_text(encoding="utf-8")
        documented = {
            match.group(1).split("#", 1)[0]
            for match in MARKDOWN_LINK.finditer(index)
        }
        expected = {
            path.relative_to(DOCS_ROOT).as_posix()
            for path in DOCS_ROOT.rglob("*.md")
            if path != DOCS_ROOT / "README.md"
        }
        self.assertEqual(documented & expected, expected)

    def test_all_repository_relative_markdown_links_resolve(self) -> None:
        failures: list[str] = []
        markdown_files = [PROJECT_ROOT / "README.md", *PROJECT_ROOT.rglob("*.md")]
        seen: set[Path] = set()
        for markdown in markdown_files:
            if markdown in seen or "venv" in markdown.parts:
                continue
            seen.add(markdown)
            text = markdown.read_text(encoding="utf-8")
            for match in MARKDOWN_LINK.finditer(text):
                raw_target = match.group(1).strip("<>")
                parsed = urlsplit(raw_target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                relative = unquote(parsed.path)
                if relative.startswith("/"):
                    continue
                candidate = (markdown.parent / relative).resolve(strict=False)
                if not candidate.is_relative_to(PROJECT_ROOT):
                    failures.append(
                        f"{markdown.relative_to(PROJECT_ROOT)} escapes repository: "
                        f"{raw_target}"
                    )
                elif not candidate.exists():
                    failures.append(
                        f"{markdown.relative_to(PROJECT_ROOT)} -> {raw_target}"
                    )
        self.assertEqual(failures, [], "Broken relative links:\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
