"""Rule-based correction and format validation for raw OCR plate reads.

Rules live in configs/plate_rules.yaml so a different plate format needs no
code change. Note on provenance: the confusion pairs here (0/O, 1/I, 5/S,
8/B, 2/Z) are the standard visually-confusable glyphs for this font/OCR
combination in general use, not a set mined from this project's own error
logs - there is no deployment history in this repo to mine patterns from.
Treat them as a reasonable starting point to replace once real misread data
exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "configs" / "plate_rules.yaml"


@dataclass
class PlateFormat:
    name: str
    pattern: str  # e.g. "DDDDLLL": D = digit position, L = letter position


@dataclass
class CorrectionResult:
    text: str
    valid: bool
    matched_format: Optional[str] = None


def load_rules(path: Path = DEFAULT_RULES_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class PlatePostProcessor:
    def __init__(self, rules: Optional[dict] = None, rules_path: Path = DEFAULT_RULES_PATH):
        rules = rules or load_rules(rules_path)
        self._formats = [PlateFormat(f["name"], f["format"]) for f in rules["formats"]]
        self._digit_to_letter: Dict[str, str] = rules["confusion_map"]["digit_to_letter"]
        self._letter_to_digit: Dict[str, str] = rules["confusion_map"]["letter_to_digit"]

    def _matching_formats(self, text: str) -> List[PlateFormat]:
        return [f for f in self._formats if len(f.pattern) == len(text)]

    def _correct_for_format(self, text: str, fmt: PlateFormat) -> str:
        chars = list(text)
        for i, expected in enumerate(fmt.pattern):
            c = chars[i]
            if expected == "D" and not c.isdigit():
                if c in self._letter_to_digit:
                    chars[i] = self._letter_to_digit[c]
            elif expected == "L" and not c.isalpha():
                if c in self._digit_to_letter:
                    chars[i] = self._digit_to_letter[c]
        return "".join(chars)

    def _matches_format(self, text: str, fmt: PlateFormat) -> bool:
        if len(text) != len(fmt.pattern):
            return False
        for c, expected in zip(text, fmt.pattern):
            if expected == "D" and not c.isdigit():
                return False
            if expected == "L" and not c.isalpha():
                return False
        return True

    def process(self, raw_text: str) -> CorrectionResult:
        """Try each format of matching length: attempt a class-consistent
        correction, then validate. Returns the first format that validates
        after correction; otherwise reports invalid with the uncorrected text.
        """
        text = raw_text.strip().upper()

        for fmt in self._matching_formats(text):
            corrected = self._correct_for_format(text, fmt)
            if self._matches_format(corrected, fmt):
                return CorrectionResult(text=corrected, valid=True, matched_format=fmt.name)

        return CorrectionResult(text=text, valid=False, matched_format=None)
