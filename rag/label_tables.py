"""Flatten openFDA SPL adverse reaction tables into caption, header, and rows."""

from dataclasses import dataclass
from html.parser import HTMLParser


@dataclass(frozen=True)
class LabelTable:
    caption: str | None
    headers: tuple[str, ...]
    rows: tuple[str, ...]


@dataclass(frozen=True)
class _Cell:
    text: str
    tag: str
    bold: bool


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[LabelTable] = []
        self._caption: str | None = None
        self._headers: list[str] = []
        self._rows: list[str] = []
        self._in_head = False
        self._cells: list[_Cell] = []
        self._text: list[str] | None = None
        self._bold = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "thead":
            self._in_head = True
        elif tag == "tr":
            self._cells = []
        elif tag in ("caption", "th", "td"):
            self._text = []
            self._bold = False
        elif self._text is not None:
            if tag in ("br", "footnote", "paragraph"):
                self._text.append(" ")
            if tag in ("b", "strong") or ("stylecode", "bold") in attrs:
                self._bold = True

    def handle_data(self, data: str) -> None:
        if self._text is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("caption", "th", "td") and self._text is not None:
            text = " ".join("".join(self._text).split())
            if tag == "caption":
                self._caption = text if text else None
            else:
                self._cells.append(_Cell(text, tag, self._bold))
            self._text = None
        elif tag in ("footnote", "paragraph") and self._text is not None:
            self._text.append(" ")
        elif tag == "tr":
            self._end_row()
        elif tag == "thead":
            self._in_head = False
        elif tag == "table":
            self.tables.append(
                LabelTable(self._caption, tuple(self._headers), tuple(self._rows))
            )
            self._caption = None
            self._headers = []
            self._rows = []

    def _end_row(self) -> None:
        populated = [cell for cell in self._cells if cell.text]
        explicit_header = bool(populated) and all(
            cell.tag == "th" for cell in populated
        )
        # Adagrasib's SPL uses bold TD cells for its two leading header rows.
        bold_header = len(populated) >= 2 and all(cell.bold for cell in populated)
        is_header = self._in_head or (
            not self._rows and (explicit_header or bold_header)
        )
        target = self._headers if is_header else self._rows
        target.append(" | ".join(cell.text for cell in self._cells))


def parse_label_tables(html: str) -> list[LabelTable]:
    parser = _TableParser()
    parser.feed(html)
    parser.close()
    return parser.tables
