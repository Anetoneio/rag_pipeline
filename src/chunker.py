"""

chunking is basically splitting the data into overlappig pieces
tries paragraph breaks first, then line breaks, then sentences, then words, then a hard char-cut(so the data can fit the chunk size)

"""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Chunk:
    text: str
    metadata: Dict = field(default_factory=dict)


class RecursiveChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Most "natural" separator first, most desperate last.
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Greedily pack `text` into pieces <= chunk_size using the first
        separator in the list; recurse with the next separator for any
        piece still too large."""
        if not separators:
            return [text]

        sep, remaining = separators[0], separators[1:]

        if sep == "":
            # Last resort: hard character-count slicing.
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size)] or [text]

        parts = text.split(sep)
        pieces, buffer = [], ""

        for part in parts:
            candidate = (buffer + sep + part) if buffer else part
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    pieces.append(buffer)
                if len(part) > self.chunk_size:
                    pieces.extend(self._split_text(part, remaining))
                    buffer = ""
                else:
                    buffer = part

        if buffer:
            pieces.append(buffer)
        return pieces

    def _add_overlap(self, pieces: List[str]) -> List[str]:
        """Prepend the tail of the previous piece onto each piece."""
        if self.chunk_overlap == 0 or len(pieces) <= 1:
            return pieces
        overlapped = [pieces[0]]
        for i in range(1, len(pieces)):
            tail = pieces[i - 1][-self.chunk_overlap:]
            # Guard against gluing two words together at the seam
            # (e.g. "...only on" + "what..." -> "...only onwhat...").
            joiner = "" if (tail.endswith((" ", "\n")) or pieces[i].startswith((" ", "\n"))) else " "
            overlapped.append(tail + joiner + pieces[i])
        return overlapped

    def chunk(self, text: str, base_metadata: Dict = None) -> List[Chunk]:
        base_metadata = base_metadata or {}
        raw = self._split_text(text.strip(), self.separators)
        pieces = self._add_overlap(raw)

        chunks = []
        for idx, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            meta = dict(base_metadata)
            meta["chunk_id"] = idx
            chunks.append(Chunk(text=piece, metadata=meta))
        return chunks