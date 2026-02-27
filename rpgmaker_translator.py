#!/usr/bin/env python3
"""
RPG Maker MV/MZ Text Translator
================================
Extracts, translates, and applies text translations for RPG Maker MV/MZ games.

Supported backends:
  - Google Translate (free, default) via ``deep-translator``
  - DeepL (premium, optional) via official ``deepl`` SDK
  - MarianMT (free, offline) via Hugging Face ``transformers`` — no API key, no limits

Author: Jaime
License: MIT
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEXT_LIMITS: Dict[str, int] = {
    "message_no_face": 55,
    "message_with_face": 43,
    "choice": 30,
    "description": 80,
    "name": 20,
}

# Matches RPG Maker escape codes *and* common plugin codes.
ESCAPE_CODE_PATTERN = re.compile(
    r"(\\[VNCI]\[\d+\]"        # \V[n], \N[n], \C[n], \I[n]
    r"|\\FS\[\d+\]"            # \FS[n] (MZ font size)
    r"|\\[{}.$|!><^]"          # \{, \}, \., \$, \|, \!, \>, \<, \^
    r"|\\[Gg]"                 # \G (currency)
    r"|\\SE\[\d+\]"            # \SE[n] (sound)
    r"|\\ME\[\d+\]"            # \ME[n] (music)
    r"|\\[Ww]\[\d+\]"          # \W[n] / \w[n] (wait)
    r"|\\AF\[\d+\]"            # \AF[n] (animated face)
    r"|<[^>]+>"                # <WordWrap>, <br>, etc.
    r"|\[[^\]]+\])",           # [Something] tags
    re.IGNORECASE,
)

# Detects text that starts with an escape prefix followed by translatable content.
TEXT_WITH_ESCAPE_PREFIX = re.compile(
    r"^(\\[A-Z]+\[\d+\]\s*)(.+)$", re.IGNORECASE
)

# CJK character ranges.
CJK_PATTERN = re.compile(
    r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf\uac00-\ud7af]"
)

# Script-like keywords that indicate code, not translatable text.
_CODE_KEYWORDS = (
    "function", "var ", "let ", "const ", "$game", "this.",
    "Game_", ".prototype", "eval(", "=>", "===",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TranslationEntry:
    """A single translatable text extracted from game data."""

    file: str           # JSON filename, e.g. "Map001.json"
    path: str           # JSON path, e.g. "events[3].pages[0].list[5].parameters[0]"
    original: str       # Original text
    translated: str = ""
    context: str = ""   # Human-readable context
    hash: str = ""      # Dedup key
    entry_type: str = "text"   # text | choice | escape_text | plugin_choices | plugin_text
    has_face: bool = False     # Message window shows a face graphic
    max_chars: int = 0

    def __post_init__(self) -> None:
        if not self.hash:
            raw = f"{self.file}:{self.path}:{self.original}"
            self.hash = hashlib.md5(raw.encode()).hexdigest()[:12]


@dataclass
class TranslationProject:
    """Container for a full translation project."""

    game_path: str
    source_lang: str
    target_lang: str
    entries: List[TranslationEntry] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = ""

    # -- Serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "game_path": self.game_path,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "created": self.created,
            "modified": self.modified or datetime.now().isoformat(),
            "entries": [
                {
                    "file": e.file,
                    "path": e.path,
                    "original": e.original,
                    "translated": e.translated,
                    "context": e.context,
                    "hash": e.hash,
                    "entry_type": e.entry_type,
                    "has_face": e.has_face,
                    "max_chars": e.max_chars,
                }
                for e in self.entries
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TranslationProject":
        project = cls(
            game_path=data["game_path"],
            source_lang=data["source_lang"],
            target_lang=data["target_lang"],
            created=data.get("created", ""),
            modified=data.get("modified", ""),
        )
        for e in data.get("entries", []):
            project.entries.append(
                TranslationEntry(
                    file=e["file"],
                    path=e["path"],
                    original=e["original"],
                    translated=e.get("translated", ""),
                    context=e.get("context", ""),
                    hash=e.get("hash", ""),
                    entry_type=e.get("entry_type", "text"),
                    has_face=e.get("has_face", False),
                    max_chars=e.get("max_chars", 0),
                )
            )
        return project


# ---------------------------------------------------------------------------
# Escape-code helpers
# ---------------------------------------------------------------------------

class EscapeCodeHandler:
    """Protect RPG Maker escape codes from being mangled by translation APIs."""

    @staticmethod
    def protect(text: str) -> Tuple[str, Optional[Dict[str, str]]]:
        """Replace escape codes with numbered placeholders.

        Returns ``(protected_text, placeholder_map)`` or ``(text, None)``
        when the text has nothing to protect.
        """
        if not isinstance(text, str) or not text.strip():
            return text, None

        preserved: Dict[str, str] = {}
        counter = [0]

        def _replace(match: re.Match) -> str:
            key = f"E{counter[0]}"
            preserved[key] = match.group(0)
            counter[0] += 1
            return key

        protected = ESCAPE_CODE_PATTERN.sub(_replace, text)
        return (protected, preserved) if preserved else (text, None)

    @staticmethod
    def restore(text: str, preserved: Optional[Dict[str, str]]) -> str:
        """Re-insert original escape codes into translated text."""
        if not preserved:
            return text
        for key, value in preserved.items():
            text = text.replace(key, value)
        return text


# ---------------------------------------------------------------------------
# Translation Dictionary / Glossary
# ---------------------------------------------------------------------------

class TranslationDictionary:
    """Persistent glossary for consistent translations across the project.

    Stores source→target term pairs so recurring names, places, abilities,
    and UI labels are translated identically every time.  The glossary is
    saved as a human-editable JSON file in the translation project folder.

    Features:
    - Exact-match lookup (case-sensitive for CJK, case-insensitive for Latin)
    - Auto-learn: after translation, common terms (names, items, etc.) are
      added automatically so future runs stay consistent.
    - Manual editing: users can add/override entries in the JSON file.
    - Priority system: manual entries take precedence over auto-learned ones.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = Path(path) if path else None
        # {source_text: {"translation": str, "context": str, "source": "manual"|"auto", "count": int}}
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        if self._path and self._path.exists():
            self.load()

    # -- persistence ---------------------------------------------------------

    def load(self, path: Optional[str] = None) -> None:
        """Load glossary from a JSON file."""
        p = Path(path) if path else self._path
        if not p or not p.exists():
            return
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                # Support both flat {src: tgt} and rich {src: {translation, ...}} formats.
                for key, value in data.items():
                    if isinstance(value, str):
                        self._entries[key] = {
                            "translation": value,
                            "context": "",
                            "source": "manual",
                            "count": 0,
                        }
                    elif isinstance(value, dict):
                        self._entries[key] = value
            logger.info("Glossary loaded: %d terms from %s", len(self._entries), p.name)
        except Exception as exc:
            logger.warning("Failed to load glossary %s: %s", p, exc)

    def save(self, path: Optional[str] = None) -> None:
        """Save glossary to a JSON file."""
        p = Path(path) if path else self._path
        if not p:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self._entries, fh, ensure_ascii=False, indent=2)
        self._dirty = False
        logger.debug("Glossary saved: %d terms → %s", len(self._entries), p.name)

    # -- lookup --------------------------------------------------------------

    def lookup(self, source_text: str) -> Optional[str]:
        """Return the glossary translation for *source_text*, or ``None``."""
        entry = self._entries.get(source_text)
        if entry:
            entry["count"] = entry.get("count", 0) + 1
            self._dirty = True
            return entry["translation"]
        return None

    def lookup_all(self, source_text: str) -> Dict[str, str]:
        """Return all glossary terms found inside *source_text*.

        Returns ``{original_term: translation}`` for every glossary entry
        whose key appears as a substring of *source_text*.  Useful for
        providing context to the translation backend or for post-processing.
        """
        matches: Dict[str, str] = {}
        for key, entry in self._entries.items():
            if key in source_text:
                matches[key] = entry["translation"]
        return matches

    # -- add / update --------------------------------------------------------

    def add(
        self,
        source: str,
        translation: str,
        context: str = "",
        *,
        manual: bool = False,
        overwrite: bool = False,
    ) -> bool:
        """Add a term to the glossary.

        Returns ``True`` if the entry was added/updated, ``False`` if it
        already existed and *overwrite* was ``False``.
        Manual entries always overwrite auto-learned ones.
        """
        if not source or not translation:
            return False

        existing = self._entries.get(source)
        if existing:
            # Manual entries always win over auto-learned.
            if manual or overwrite or existing.get("source") == "auto":
                pass  # proceed to overwrite
            else:
                return False

        self._entries[source] = {
            "translation": translation,
            "context": context,
            "source": "manual" if manual else "auto",
            "count": existing.get("count", 0) if existing else 0,
        }
        self._dirty = True
        return True

    def add_batch(
        self,
        pairs: List[Tuple[str, str]],
        context: str = "",
        *,
        manual: bool = False,
    ) -> int:
        """Add multiple (source, translation) pairs.  Returns count added."""
        added = 0
        for src, tgt in pairs:
            if self.add(src, tgt, context, manual=manual):
                added += 1
        return added

    def remove(self, source: str) -> bool:
        """Remove a term from the glossary."""
        if source in self._entries:
            del self._entries[source]
            self._dirty = True
            return True
        return False

    # -- auto-learn from project ---------------------------------------------

    def learn_from_project(self, project: "TranslationProject") -> int:
        """Auto-populate glossary from translated project entries.

        Focuses on short, high-value terms like character names, item names,
        skill names, etc.  Longer dialog lines are ignored.
        """
        # Entry types and contexts that are good glossary candidates.
        _GLOSSARY_CONTEXTS = (
            "name", "nickname", "currencyunit",
            "skill", "item", "weapon", "armor", "class", "enemy", "state",
            "element", "skilltype", "weapontype", "armortype", "equiptype",
            "speaker", "choice", "mapinfo", "event",
        )
        added = 0
        for entry in project.entries:
            if not entry.translated or entry.translated == entry.original:
                continue
            # Only learn short terms (names, labels).
            ctx_lower = entry.context.lower()
            is_name_like = any(kw in ctx_lower for kw in _GLOSSARY_CONTEXTS)
            is_short = len(entry.original) <= 30
            if is_name_like or (is_short and entry.max_chars and entry.max_chars <= 30):
                if self.add(entry.original, entry.translated, entry.context):
                    added += 1
        if added:
            logger.info("Glossary auto-learned %d terms from project", added)
            self._dirty = True
        return added

    # -- bulk apply to project -----------------------------------------------

    def apply_to_entries(
        self, entries: List[TranslationEntry], *, only_untranslated: bool = True
    ) -> int:
        """Apply glossary translations to project entries.

        Only exact full-text matches are applied (no partial replacement).
        Returns the number of entries filled in.
        """
        applied = 0
        for entry in entries:
            if only_untranslated and entry.translated:
                continue
            translation = self.lookup(entry.original)
            if translation:
                entry.translated = translation
                applied += 1
        if applied:
            logger.info("Glossary applied %d translations", applied)
        return applied

    # -- export / import (simple CSV) ----------------------------------------

    def export_csv(self, path: str) -> None:
        """Export glossary as a simple two-column CSV."""
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(["source", "translation", "context", "source_type", "count"])
            for src, info in sorted(self._entries.items()):
                writer.writerow([
                    src, info["translation"], info.get("context", ""),
                    info.get("source", "auto"), info.get("count", 0),
                ])
        logger.info("Glossary CSV exported: %d terms → %s", len(self._entries), path)

    def import_csv(self, path: str) -> int:
        """Import terms from a CSV (columns: source, translation, [context])."""
        imported = 0
        with open(path, "r", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                src = row.get("source", "").strip()
                tgt = row.get("translation", "").strip()
                ctx = row.get("context", "").strip()
                if src and tgt:
                    self.add(src, tgt, ctx, manual=True, overwrite=True)
                    imported += 1
        logger.info("Glossary CSV imported: %d terms from %s", imported, path)
        return imported

    # -- info ----------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def dirty(self) -> bool:
        return self._dirty

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"TranslationDictionary({len(self._entries)} terms)"


# ---------------------------------------------------------------------------
# Translation backends
# ---------------------------------------------------------------------------

class TranslationBackend(ABC):
    """Abstract base for translation backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def translate(self, text: str, source: str, target: str) -> str:
        ...

    @abstractmethod
    def translate_batch(
        self, texts: List[str], source: str, target: str
    ) -> List[str]:
        ...


class GoogleTranslateBackend(TranslationBackend):
    """Free translation using Google Translate (via *deep-translator*)."""

    def __init__(self) -> None:
        try:
            from deep_translator import GoogleTranslator  # noqa: F401
            self._translator_cls = GoogleTranslator
        except ImportError:
            raise ImportError(
                "deep-translator is required for the Google backend. "
                "Install it with: pip install deep-translator"
            )

    @property
    def name(self) -> str:
        return "Google Translate (free)"

    # -- single translation --------------------------------------------------

    def translate(self, text: str, source: str, target: str) -> str:
        if not text or not text.strip():
            return text
        protected, preserved = EscapeCodeHandler.protect(text)
        try:
            result = self._translator_cls(source=source, target=target).translate(protected)
            return EscapeCodeHandler.restore(result or text, preserved)
        except Exception as exc:
            logger.warning("Google translation failed for '%s…': %s", text[:40], exc)
            return text

    # -- batch translation with retry ----------------------------------------

    def translate_batch(
        self, texts: List[str], source: str, target: str
    ) -> List[str]:
        if not texts:
            return []

        translator = self._translator_cls(source=source, target=target)
        results: List[str] = []
        max_retries = 3

        for text in texts:
            if not text or not text.strip():
                results.append(text)
                continue

            protected, preserved = EscapeCodeHandler.protect(text)

            for attempt in range(max_retries):
                try:
                    result = translator.translate(protected)
                    results.append(EscapeCodeHandler.restore(result or text, preserved))
                    break
                except Exception as exc:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.debug("Retry %d for '%s…' in %ds", attempt + 1, text[:30], wait)
                        time.sleep(wait)
                    else:
                        logger.warning("Translation failed after retries: %s", exc)
                        results.append(text)

        return results


class DeepLTranslateBackend(TranslationBackend):
    """Premium translation using the official DeepL API."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                "DeepL API key is required. "
                "Pass --api-key or set the DEEPL_API_KEY environment variable."
            )
        try:
            import deepl
            self._client = deepl.Translator(api_key)
        except ImportError:
            raise ImportError(
                "deepl package is required. Install it with: pip install deepl"
            )

    @property
    def name(self) -> str:
        return "DeepL (premium)"

    @staticmethod
    def _map_lang(code: str, *, is_target: bool = False) -> Optional[str]:
        """Convert common language codes to DeepL format."""
        if not code:
            return None
        c = code.lower()
        if c in ("ja", "jp", "jpn"):
            return "JA"
        if c.startswith("en"):
            return "EN-US" if is_target else "EN"
        if c in ("zh", "zh-cn", "zh-hans"):
            return "ZH-HANS" if is_target else "ZH"
        if c in ("zh-tw", "zh-hant"):
            return "ZH-HANT" if is_target else "ZH"
        if c in ("pt", "pt-br"):
            return "PT-BR" if is_target else "PT"
        return c.upper()

    def translate(self, text: str, source: str, target: str) -> str:
        if not text or not text.strip():
            return text
        protected, preserved = EscapeCodeHandler.protect(text)
        try:
            result = self._client.translate_text(
                protected,
                source_lang=self._map_lang(source),
                target_lang=self._map_lang(target, is_target=True),
            )
            return EscapeCodeHandler.restore(result.text, preserved)
        except Exception as exc:
            logger.warning("DeepL translation failed: %s", exc)
            return text

    def translate_batch(
        self, texts: List[str], source: str, target: str
    ) -> List[str]:
        if not texts:
            return []

        protected_list: List[str] = []
        preserved_list: List[Optional[Dict[str, str]]] = []

        for text in texts:
            protected, preserved = EscapeCodeHandler.protect(text)
            protected_list.append(protected if preserved is not None else text)
            preserved_list.append(preserved)

        try:
            results = self._client.translate_text(
                protected_list,
                source_lang=self._map_lang(source),
                target_lang=self._map_lang(target, is_target=True),
            )
            if not isinstance(results, list):
                results = [results]

            output: List[str] = []
            for original, res, preserved in zip(texts, results, preserved_list):
                if preserved is None:
                    output.append(original)
                else:
                    output.append(EscapeCodeHandler.restore(res.text, preserved))
            return output
        except Exception as exc:
            logger.warning("DeepL batch translation failed: %s", exc)
            return texts


class MarianMTBackend(TranslationBackend):
    """Offline translation using Helsinki-NLP MarianMT models via Hugging Face.

    Runs entirely locally — no API key, no rate limits, no internet after the
    first model download.  Uses GPU automatically when available.
    """

    # Helsinki-NLP model naming convention: opus-mt-{src}-{tgt}
    _MODEL_OVERRIDES: Dict[Tuple[str, str], str] = {
        # Japanese → English uses the "big" model for better quality
        ("ja", "en"): "Helsinki-NLP/opus-mt-tc-big-ja-en",
    }

    def __init__(
        self,
        source_lang: str = "ja",
        target_lang: str = "en",
        *,
        max_length: int = 512,
    ) -> None:
        self._ensure_dependencies()

        from transformers import MarianMTModel, MarianTokenizer
        import torch

        self._src = source_lang.lower()
        self._tgt = target_lang.lower()
        self._max_length = max_length

        model_name = self._resolve_model(self._src, self._tgt)
        logger.info("Loading MarianMT model '%s' …", model_name)

        self._tokenizer = MarianTokenizer.from_pretrained(model_name)
        self._model = MarianMTModel.from_pretrained(model_name)

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model.to(self._device)
        self._model.eval()
        logger.info(
            "MarianMT ready on %s (%s → %s)",
            self._device,
            self._src,
            self._tgt,
        )

    # -- auto-install --------------------------------------------------------

    @staticmethod
    def _ensure_dependencies() -> None:
        """Auto-install transformers, sentencepiece and torch if missing."""
        required = {
            "transformers": "transformers",
            "sentencepiece": "sentencepiece",
            "torch": "torch",
        }
        missing: List[str] = []
        for module, pip_name in required.items():
            try:
                __import__(module)
            except ImportError:
                missing.append(pip_name)

        if not missing:
            return

        import subprocess
        logger.info(
            "MarianMT dependencies not found — installing: %s",
            ", ".join(missing),
        )
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", *missing],
                stdout=subprocess.DEVNULL,
            )
            logger.info("Dependencies installed successfully.")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Auto-install of MarianMT dependencies failed (exit {exc.returncode}). "
                f"Install manually: pip install {' '.join(missing)}"
            ) from exc

    # -- helpers -------------------------------------------------------------

    @classmethod
    def _resolve_model(cls, src: str, tgt: str) -> str:
        """Return the Hugging Face model ID for the given language pair."""
        key = (src, tgt)
        if key in cls._MODEL_OVERRIDES:
            return cls._MODEL_OVERRIDES[key]
        return f"Helsinki-NLP/opus-mt-{src}-{tgt}"

    # -- TranslationBackend interface ----------------------------------------

    @property
    def name(self) -> str:
        return "MarianMT (offline, free)"

    def translate(self, text: str, source: str, target: str) -> str:
        if not text or not text.strip():
            return text
        return self.translate_batch([text], source, target)[0]

    def translate_batch(
        self, texts: List[str], source: str, target: str
    ) -> List[str]:
        if not texts:
            return []

        import torch

        # Separate blank / non-blank, preserving order
        indices: List[int] = []          # positions of translatable items
        protected_texts: List[str] = []  # inputs with escape codes masked
        preserved_maps: List[Optional[Dict[str, str]]] = []
        results: List[str] = list(texts)  # start as copy

        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            protected, preserved = EscapeCodeHandler.protect(text)
            indices.append(i)
            protected_texts.append(protected)
            preserved_maps.append(preserved)

        if not protected_texts:
            return results

        try:
            batch = self._tokenizer(
                protected_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_length,
            ).to(self._device)

            with torch.no_grad():
                generated = self._model.generate(**batch, max_length=self._max_length)

            decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)

            for idx, translated, preserved in zip(indices, decoded, preserved_maps):
                results[idx] = EscapeCodeHandler.restore(translated, preserved)
        except Exception as exc:
            logger.warning("MarianMT batch translation failed: %s", exc)
            # leave results as originals

        return results


def create_backend(
    backend: str = "google",
    api_key: Optional[str] = None,
    source_lang: str = "ja",
    target_lang: str = "en",
) -> TranslationBackend:
    """Factory function to instantiate a translation backend."""
    backend = backend.lower()
    if backend == "google":
        return GoogleTranslateBackend()
    if backend == "deepl":
        api_key = api_key or os.getenv("DEEPL_API_KEY", "")
        return DeepLTranslateBackend(api_key)
    if backend in ("marian", "marianmt"):
        return MarianMTBackend(source_lang=source_lang, target_lang=target_lang)
    raise ValueError(
        f"Unknown backend '{backend}'. Supported: 'google', 'deepl', 'marian'."
    )


# ---------------------------------------------------------------------------
# Text Wrapper
# ---------------------------------------------------------------------------

class TextWrapper:
    """Word-wrap translated text to fit RPG Maker message windows."""

    def __init__(
        self,
        max_no_face: int = TEXT_LIMITS["message_no_face"],
        max_with_face: int = TEXT_LIMITS["message_with_face"],
    ) -> None:
        self.max_no_face = max_no_face
        self.max_with_face = max_with_face

    @staticmethod
    def _visible_length(text: str) -> int:
        """Character count excluding escape codes."""
        return len(ESCAPE_CODE_PATTERN.sub("", text))

    def wrap(self, text: str, has_face: bool = False) -> str:
        """Wrap *text* to fit the message window."""
        max_chars = self.max_with_face if has_face else self.max_no_face
        wrapped_lines: List[str] = []

        for line in text.split("\n"):
            if self._visible_length(line) <= max_chars:
                wrapped_lines.append(line)
                continue
            words = line.split(" ")
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip() if current else word
                if self._visible_length(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        wrapped_lines.append(current)
                    current = word
            if current:
                wrapped_lines.append(current)

        return "\n".join(wrapped_lines)


# ---------------------------------------------------------------------------
# Plugin JSON identifier keys (module-level, shared by extractor & applier)
# ---------------------------------------------------------------------------

# Keys inside JSON-encoded plugin parameters whose values are
# identifiers, config keys, or code — NOT translatable display text.
# Checked case-insensitively.
_PLUGIN_JSON_IDENTIFIER_KEYS: Set[str] = {
    # Universal identifier keys
    "key", "symbol", "tag", "code",
    # Config / internal names — often used as property lookup keys
    # (e.g. CustomizeConfigItem uses  Name  as ConfigManager[name]).
    "name", "parentname", "commandname",
    # Reference IDs (mixed-case variants of _MZ_SKIP_KEYS)
    "id", "switchid", "variableid", "commoneventid",
    "hiddenswitch", "hiddenswitchid", "disableswitchid",
    "hiddenflag", "disableflag",
    # Scripting / code
    "script", "condition", "filter", "formula", "eval",
    "method", "function", "func", "class", "classname",
    # Input / command
    "input", "command", "binding", "trigger",
    # Format strings
    "format",
    # Technical / positional
    "filename", "file", "image", "se", "missse",
    "type", "property", "param", "paramname", "meta",
    "addposition", "variable",
    # Numeric / size (catch mixed-case variants)
    "x", "y", "width", "height", "size",
    "volume", "pitch", "pan", "duration", "opacity",
    "fontsize", "fontface",
    "icon", "iconid",
    # Color (values like "silver", "white" are not translatable text)
    "color", "bordercolor", "textcolor", "textbordercolor",
    "textbordersize", "borderwidth",
    "textbackground",
    # Boolean-like
    "defaultvalue",
    # Animation / particle / effect identifiers
    "animation", "animationid", "particle", "variation",
    "blend", "blendmode", "easing", "easingtype",
    # Sound / resource references
    "bgm", "bgs", "me", "sound", "soundeffect",
    "picture", "picturename", "facename", "faceset",
    "charactername", "battlername", "tileset",
    # Plugin-specific technical keys
    "scene", "window", "handler", "callback",
    "plugin", "pluginname", "target", "action",
    "mode", "align", "anchor", "origin",
    "direction", "pattern", "priority", "layer",
    "region", "regionid", "terrain", "terrainid",
}


# ---------------------------------------------------------------------------
# Text Extractor
# ---------------------------------------------------------------------------

class TextExtractor:
    """Extracts translatable strings from RPG Maker JSON data files."""

    # Database files → list of translatable field names.
    DATABASE_FIELDS: Dict[str, List[str]] = {
        "Actors.json": ["name", "nickname", "profile"],
        "Classes.json": ["name"],
        "Skills.json": ["name", "description", "message1", "message2"],
        "Items.json": ["name", "description"],
        "Weapons.json": ["name", "description"],
        "Armors.json": ["name", "description"],
        "Enemies.json": ["name"],
        "States.json": ["name", "message1", "message2", "message3", "message4"],
        "CommonEvents.json": ["name"],
        "Troops.json": ["name"],
        "MapInfos.json": ["name"],
    }

    def __init__(self, game_path: str, source_lang: str = "ja") -> None:
        self.game_path = Path(game_path)
        self.source_lang = source_lang
        self._entries: List[TranslationEntry] = []
        self._seen: Set[str] = set()

        # Resolve data path (MV: www/data, MZ: data).
        self.data_path = self.game_path / "www" / "data"
        if not self.data_path.exists():
            self.data_path = self.game_path / "data"
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"No data folder found in {self.game_path}. "
                "Is this a valid RPG Maker game?"
            )

    # -- public API ----------------------------------------------------------

    def extract_all(self) -> TranslationProject:
        """Run full extraction and return a :class:`TranslationProject`."""
        logger.info("Extracting translatable text …")
        self._extract_database_files()
        self._extract_system()
        self._extract_maps()
        self._extract_plugins_js()
        self._extract_plugin_js_sources()
        logger.info("Extracted %d unique entries", len(self._entries))
        return TranslationProject(
            game_path=str(self.game_path),
            source_lang=self.source_lang,
            target_lang="",
            entries=self._entries,
        )

    # -- helpers -------------------------------------------------------------

    def _should_translate(self, text: Any) -> bool:
        """Return ``True`` if *text* looks like translatable content."""
        if not text or not isinstance(text, str):
            return False
        clean = ESCAPE_CODE_PATTERN.sub("", text).strip()
        if not clean:
            return False
        # Contains CJK characters → definitely translatable.
        if CJK_PATTERN.search(clean):
            return True
        # Contains at least two Latin letters (English menu text, etc.).
        if re.search(r"[a-zA-Z]{2,}", clean):
            return True
        return False

    @staticmethod
    def _looks_like_code(text: str) -> bool:
        """Return ``True`` if *text* appears to be JavaScript source code."""
        lower = text.lower()
        return any(kw in lower for kw in _CODE_KEYWORDS)

    def _add(
        self,
        file: str,
        path: str,
        text: str,
        context: str = "",
        entry_type: str = "text",
        has_face: bool = False,
        max_chars: int = 0,
    ) -> None:
        """Add an entry if it passes validation and hasn't been seen yet."""
        if not text or not isinstance(text, str):
            return
        text = text.strip()
        if not text or self._looks_like_code(text):
            return
        entry = TranslationEntry(
            file=file,
            path=path,
            original=text,
            context=context,
            entry_type=entry_type,
            has_face=has_face,
            max_chars=max_chars,
        )
        if entry.hash not in self._seen:
            self._seen.add(entry.hash)
            self._entries.append(entry)

    def _extract_escape_text(
        self, text: str, file: str, path: str, context: str, has_face: bool = False
    ) -> None:
        """Handle text that may start with escape-code prefixes."""
        if not text:
            return
        match = TEXT_WITH_ESCAPE_PREFIX.match(text)
        if match and self._should_translate(match.group(2)):
            self._add(file, path, text, f"{context} (escape)", "escape_text", has_face)
            return
        if self._should_translate(text):
            self._add(file, path, text, context, has_face=has_face)

    # -- note tag extraction --------------------------------------------------

    # Matches <Tag:Value> or <Tag> patterns common in RPG Maker note fields.
    _NOTE_TAG_PATTERN = re.compile(
        r"<([A-Za-z_][A-Za-z0-9_ ]*)(?::([^>]+))?>" 
    )

    # Tags known to contain translatable text.
    _TRANSLATABLE_TAGS = {
        "description", "text", "book text", "book", "sg description",
        "help", "help description", "name", "display name", "title",
        "message", "label", "caption", "info", "popup",
        "quest", "quest description", "objective", "reward",
        "bio", "biography", "profile",
        # Plugin-specific label tags
        "lb",           # EventLabel.js — map event labels
        "labelname",    # Alternative label plugins
        "popuptext",    # Popup text plugins
        "notify",       # Notification plugins
    }

    def _extract_note_tags(
        self, note: str, file: str, path: str, context: str
    ) -> None:
        """Extract translatable text from `<Tag:Value>` note tags."""
        if not note or not isinstance(note, str):
            return
        for match in self._NOTE_TAG_PATTERN.finditer(note):
            tag_name = match.group(1).strip().lower()
            tag_value = match.group(2)
            if not tag_value:
                continue
            tag_value = tag_value.strip()
            # Skip config-style values (booleans, pure numbers, file paths).
            if tag_value.lower() in ("true", "false", "on", "off", "yes", "no"):
                continue
            if re.match(r"^-?\d+\.?\d*$", tag_value):
                continue
            if "/" in tag_value or "\\" in tag_value:
                continue
            # Accept if tag name suggests translatable content, or value has CJK.
            if (
                tag_name in self._TRANSLATABLE_TAGS
                or self._should_translate(tag_value)
            ) and not self._looks_like_code(tag_value):
                safe_tag = tag_name.replace(" ", "_")
                self._add(
                    file, f"{path}.__note__{safe_tag}",
                    tag_value, f"{context} - Note <{match.group(1)}>",
                    "text", max_chars=80,
                )

    # -- script string extraction (conservative) ----------------------------

    # Matches quoted strings in JS code.
    _JS_STRING_PATTERN = re.compile(
        r"""(?:'([^'\\]*(?:\\.[^'\\]*)*)'|"([^"\\]*(?:\\.[^"\\]*)*)")"""  
    )

    def _extract_script_strings(
        self, lines: List[str], file: str, base_path: str, context: str
    ) -> None:
        """Extract translatable string literals from script calls (code 355/655).

        Very conservative: only extracts strings that contain CJK characters
        and don't look like file paths, variable names, or code.
        """
        full_script = " ".join(lines)
        for match in self._JS_STRING_PATTERN.finditer(full_script):
            value = match.group(1) or match.group(2)
            if not value:
                continue
            # Must contain CJK to be considered translatable (strict).
            if not CJK_PATTERN.search(value):
                continue
            # Skip file-path-like strings.
            if "/" in value or "\\" in value or "." in value.split()[-1] if value.split() else False:
                continue
            if not self._looks_like_code(value) and len(value) >= 2:
                self._add(
                    file, f"{base_path}.__script_string__",
                    value, f"{context} - Script String",
                    "text",
                )

    # -- comment extraction --------------------------------------------------

    def _extract_comments(
        self, comment_lines: List[str], file: str, base_path: str,
        first_cmd_index: int, context: str
    ) -> None:
        """Extract translatable text from comments (code 108/408).

        Some plugins use comments for configuration that includes translatable
        text (e.g., map labels, quest descriptions, NPC dialogue tags).
        """
        full_comment = "\n".join(comment_lines).strip()
        if not full_comment:
            return

        # Check for note-tag style content inside comments.
        if "<" in full_comment and ">" in full_comment:
            self._extract_note_tags(
                full_comment, file,
                f"{base_path}[{first_cmd_index}].parameters[0]",
                f"{context} - Comment",
            )
            return

        # If the entire comment looks like translatable text (not a config directive).
        if self._should_translate(full_comment) and not self._looks_like_code(full_comment):
            # Skip comments that look like plugin directives (key=value, etc.).
            if not re.match(r'^[A-Za-z_]+\s*[:=]', full_comment):
                self._add(
                    file, f"{base_path}[{first_cmd_index}].parameters[0]",
                    full_comment, f"{context} - Comment",
                    "text",
                )

    # -- plugin command parsing ----------------------------------------------

    def _parse_plugin_command_mv(
        self, cmd: str, file: str, path: str, context: str
    ) -> None:
        """Parse MV-style plugin commands (event code 356)."""
        if not cmd:
            return
        parts = cmd.split(" ", 2)
        if len(parts) < 2:
            return
        plugin, sub = parts[0], parts[1]
        args = parts[2] if len(parts) > 2 else ""

        if plugin == "LL_GalgeChoiceWindowMV":
            if sub == "setChoices" and args:
                if any(self._should_translate(c.strip().strip("\"'")) for c in args.split(",")):
                    self._add(
                        file, f"{path}.__plugin_choices__", args,
                        f"{context} - Plugin Choices", "plugin_choices",
                    )
            elif sub == "setMessageText" and self._should_translate(args):
                self._add(
                    file, f"{path}.__plugin_text__", args,
                    f"{context} - Plugin Text", "plugin_text",
                )
        elif args and self._should_translate(args) and not self._looks_like_code(args):
            if sub.lower() in ("text", "message", "name", "title", "popup"):
                self._add(
                    file, f"{path}.__plugin_generic__", args,
                    f"{context} - Plugin", "plugin_text",
                )

    def _parse_plugin_command_mz(
        self, params: list, file: str, path: str, context: str
    ) -> None:
        """Parse MZ-style plugin commands (event code 357).

        MZ plugin commands have structured parameters:
        params[0] = plugin name, params[1] = command name,
        params[2] = sub-command (optional), params[3] = JSON args dict.
        """
        if len(params) < 2:
            return
        plugin_name = params[0] if isinstance(params[0], str) else ""
        command_name = params[1] if isinstance(params[1], str) else ""

        # params[2] is the sub-command description (internal, not user-visible).
        # Translatable content lives in params[3+] (structured arg dicts).
        for pi in range(3, len(params)):
            param = params[pi]
            if isinstance(param, str) and self._should_translate(param):
                if not self._looks_like_code(param):
                    self._add(
                        file, f"{path}.parameters[{pi}]",
                        param,
                        f"{context} - MZ Plugin {plugin_name}.{command_name}",
                        "plugin_text",
                    )
            elif isinstance(param, dict):
                # MZ structured args — walk key/value pairs.
                self._extract_mz_plugin_args(
                    param, file, f"{path}.parameters[{pi}]",
                    f"{context} - MZ Plugin {plugin_name}.{command_name}",
                )

    # Keys in MZ plugin arg dicts that are NEVER translatable.
    _MZ_SKIP_KEYS = {
        "name", "volume", "pitch", "pan", "channel",  # audio params
        "id", "switchId", "variableId", "commonEventId",  # references
        "x", "y", "sx", "sy", "row", "col", "width", "height",  # positions
        "duration", "wait", "speed", "opacity", "scale",  # numeric settings
        "fontSize", "fontFace", "fontBold", "fontItalic",  # font config
        "fadeOut", "fadeIn", "filename", "fileName",  # file refs
        "switchVictory", "switchEscape", "switchAbort", "switchLose",
        "whenVictory", "whenEscape", "whenLose",  # battle conditions
        "stopBgm", "stopBgs",  # audio toggles
        "type", "listIndex",  # internal indices
        "note",  # config metadata (often contains <tags>)
        "icon",  # icon index
    }

    def _extract_mz_plugin_args(
        self, args: dict, file: str, base_path: str, context: str
    ) -> None:
        """Recursively extract translatable strings from MZ plugin arg dicts."""
        for key, value in args.items():
            if key in self._MZ_SKIP_KEYS:
                continue
            if isinstance(value, str) and self._should_translate(value):
                if not self._looks_like_code(value):
                    self._add(
                        file, f"{base_path}.{key}",
                        value, f"{context} ({key})",
                        "plugin_text",
                    )
            elif isinstance(value, dict):
                self._extract_mz_plugin_args(
                    value, file, f"{base_path}.{key}", context,
                )
            elif isinstance(value, list):
                for li, item in enumerate(value):
                    if isinstance(item, str) and self._should_translate(item):
                        self._add(
                            file, f"{base_path}.{key}[{li}]",
                            item, f"{context} ({key})",
                            "plugin_text",
                        )
                    elif isinstance(item, dict):
                        self._extract_mz_plugin_args(
                            item, file, f"{base_path}.{key}[{li}]", context,
                        )

    # -- event command list --------------------------------------------------

    def _extract_events(
        self, event_list: list, file: str, base_path: str, context: str
    ) -> None:
        """Walk an event command list and collect translatable entries."""
        if not event_list:
            return
        has_face = False
        comment_buffer: List[str] = []     # accumulates 108+408 lines
        comment_start_idx: int = 0
        script_buffer: List[str] = []      # accumulates 355+655 lines
        script_start_idx: int = 0

        def _flush_comments() -> None:
            nonlocal comment_buffer
            if comment_buffer:
                self._extract_comments(
                    comment_buffer, file, base_path,
                    comment_start_idx, context,
                )
                comment_buffer = []

        def _flush_scripts() -> None:
            nonlocal script_buffer
            if script_buffer:
                self._extract_script_strings(
                    script_buffer, file, base_path,
                    f"{context} - Script",
                )
                script_buffer = []

        for i, cmd in enumerate(event_list):
            if not isinstance(cmd, dict):
                continue
            code: int = cmd.get("code", 0)
            params: list = cmd.get("parameters", [])

            # Flush accumulated comment/script buffers when a different code appears.
            if code not in (108, 408) and comment_buffer:
                _flush_comments()
            if code not in (355, 655) and script_buffer:
                _flush_scripts()

            # Skip pure logic commands (but NOT code 122 string assignments).
            if code == 121:
                continue

            if code == 122:
                # Control Variable – type 4 means the value is a string
                # literal stored in params[4] (e.g. "'安全日'").  These
                # strings are player-visible when plugins evaluate the
                # variable at runtime.
                if len(params) >= 5 and params[3] == 4 and isinstance(params[4], str):
                    raw = params[4]
                    # Strip wrapping single-quotes that RPG Maker adds.
                    if len(raw) >= 2 and raw.startswith("'") and raw.endswith("'"):
                        raw = raw[1:-1]
                    if raw.strip() and self._should_translate(raw):
                        self._add(
                            file, f"{base_path}[{i}].parameters[4]",
                            params[4], f"{context} - Variable String",
                        )
                continue

            if code == 101:
                # Show Text header – detect face, extract speaker name.
                has_face = bool(params[0]) if params else False
                if len(params) > 4 and params[4] and self._should_translate(params[4]):
                    self._add(
                        file, f"{base_path}[{i}].parameters[4]",
                        params[4], f"{context} - Speaker",
                    )

            elif code == 401 and params and isinstance(params[0], str):
                # Text continuation line.
                self._extract_escape_text(
                    params[0], file,
                    f"{base_path}[{i}].parameters[0]",
                    f"{context} - Dialog", has_face,
                )

            elif code == 102 and params and isinstance(params[0], list):
                # Show Choices.
                for j, choice in enumerate(params[0]):
                    if choice and self._should_translate(choice):
                        self._add(
                            file, f"{base_path}[{i}].parameters[0][{j}]",
                            choice, f"{context} - Choice", "choice", max_chars=30,
                        )

            elif code == 402 and len(params) > 1 and isinstance(params[1], str):
                if self._should_translate(params[1]):
                    self._add(
                        file, f"{base_path}[{i}].parameters[1]",
                        params[1], f"{context} - Choice Label",
                    )

            elif code == 405 and params and isinstance(params[0], str):
                # Scrolling text data.
                self._extract_escape_text(
                    params[0], file,
                    f"{base_path}[{i}].parameters[0]",
                    f"{context} - Scroll",
                )

            elif code in (320, 324, 325) and len(params) > 1 and isinstance(params[1], str):
                if self._should_translate(params[1]):
                    self._add(
                        file, f"{base_path}[{i}].parameters[1]",
                        params[1], context, max_chars=20,
                    )

            elif code == 356 and params and isinstance(params[0], str):
                # MV plugin command.
                self._parse_plugin_command_mv(
                    params[0], file,
                    f"{base_path}[{i}].parameters[0]", context,
                )

            elif code == 357 and params:
                # MZ plugin command (structured args).
                self._parse_plugin_command_mz(
                    params, file,
                    f"{base_path}[{i}]", context,
                )

            elif code == 108 and params and isinstance(params[0], str):
                # Comment (first line).
                comment_buffer = [params[0]]
                comment_start_idx = i

            elif code == 408 and params and isinstance(params[0], str):
                # Comment continuation.
                comment_buffer.append(params[0])

            elif code == 355 and params and isinstance(params[0], str):
                # Script call (first line).
                script_buffer = [params[0]]
                script_start_idx = i

            elif code == 655 and params and isinstance(params[0], str):
                # Script call continuation.
                script_buffer.append(params[0])

        # Flush any remaining buffered comments/scripts at end of list.
        _flush_comments()
        _flush_scripts()

    # -- file-level extraction -----------------------------------------------

    def _load_json(self, path: Path) -> Any:
        """Load a JSON file, returning ``None`` on failure."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.debug("Failed to load %s: %s", path.name, exc)
            return None

    def _extract_database_files(self) -> None:
        """Extract translatable fields from the standard database JSON files."""
        for filename, fields in self.DATABASE_FIELDS.items():
            filepath = self.data_path / filename
            data = self._load_json(filepath)
            if not isinstance(data, list):
                continue

            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                item_name = item.get("name", "")

                for fld in fields:
                    value = item.get(fld)
                    if isinstance(value, str) and self._should_translate(value):
                        limit = 20 if fld == "name" else 80
                        self._add(
                            filename, f"[{idx}].{fld}", value,
                            f"{filename} - {item_name} - {fld}", max_chars=limit,
                        )

                # Note tags inside database objects.
                note = item.get("note")
                if note and isinstance(note, str) and ("<" in note):
                    self._extract_note_tags(
                        note, filename, f"[{idx}].note",
                        f"{filename} - {item_name}",
                    )

                # Event lists inside database objects (CommonEvents, Troops).
                if "list" in item:
                    self._extract_events(
                        item["list"], filename,
                        f"[{idx}].list", item_name,
                    )
                # Pages (Troops battle events).
                for pi, page in enumerate(item.get("pages") or []):
                    if isinstance(page, dict) and "list" in page:
                        self._extract_events(
                            page["list"], filename,
                            f"[{idx}].pages[{pi}].list",
                            f"{item_name} P{pi}",
                        )

    def _extract_system(self) -> None:
        """Extract translatable content from *System.json*."""
        data = self._load_json(self.data_path / "System.json")
        if not isinstance(data, dict):
            return

        # Game title.
        title = data.get("gameTitle", "")
        if title and self._should_translate(title):
            self._add("System.json", "gameTitle", title, "System - Game Title", max_chars=80)

        # Currency unit.
        currency = data.get("currencyUnit", "")
        if currency and self._should_translate(currency):
            self._add("System.json", "currencyUnit", currency, "System - Currency", max_chars=20)

        # Type arrays (elements, skillTypes, …).
        for key in ("elements", "skillTypes", "weaponTypes", "armorTypes", "equipTypes"):
            for i, value in enumerate(data.get(key) or []):
                if value and self._should_translate(value):
                    self._add("System.json", f"{key}[{i}]", value, f"System {key}")

        # Switch and variable names (editor labels, safe to translate).
        # These are developer labels only; translating them does NOT affect gameplay.
        for key in ("switches", "variables"):
            for i, value in enumerate(data.get(key) or []):
                if not value or not isinstance(value, str):
                    continue
                # Only extract names that contain CJK characters.
                # Skip auto-generated names like "Switch001", "Variable001".
                if CJK_PATTERN.search(value):
                    self._add(
                        "System.json", f"{key}[{i}]", value,
                        f"System {key} (editor label, safe to translate)",
                        max_chars=30,
                    )

        # Terms.
        terms = data.get("terms") or {}
        for section in ("basic", "params", "commands"):
            for i, value in enumerate(terms.get(section) or []):
                if value and self._should_translate(value):
                    self._add(
                        "System.json", f"terms.{section}[{i}]",
                        value, f"Term {section}",
                    )
        for key, value in (terms.get("messages") or {}).items():
            if value and self._should_translate(value):
                self._add(
                    "System.json", f"terms.messages.{key}",
                    value, f"Message {key}",
                )

    def _extract_maps(self) -> None:
        """Extract translatable content from all Map*.json files."""
        for map_file in sorted(self.data_path.glob("Map*.json")):
            if map_file.name == "MapInfos.json":
                continue
            data = self._load_json(map_file)
            if not isinstance(data, dict):
                continue

            fname = map_file.name

            # Map display name.
            display = data.get("displayName")
            if display and self._should_translate(display):
                self._add(fname, "displayName", display, f"{fname} Name")

            # Events.
            for ei, event in enumerate(data.get("events") or []):
                if not event:
                    continue
                ev_name = event.get("name", "")
                if ev_name and self._should_translate(ev_name):
                    self._add(fname, f"events[{ei}].name", ev_name, f"{fname} Event")

                # Note tags on events (e.g. <LB:label>, <Text:...>).
                ev_note = event.get("note", "")
                if ev_note and isinstance(ev_note, str) and "<" in ev_note:
                    self._extract_note_tags(
                        ev_note, fname, f"events[{ei}].note",
                        f"{fname} - {ev_name}",
                    )

                for pi, page in enumerate(event.get("pages") or []):
                    if isinstance(page, dict) and "list" in page:
                        self._extract_events(
                            page["list"], fname,
                            f"events[{ei}].pages[{pi}].list",
                            f"{fname} - {ev_name}",
                        )

    # -- plugins.js parameter extraction -------------------------------------

    # Regex for values that are obviously NOT translatable text.
    _PLUGIN_SKIP_VALUE_RE = re.compile(
        r'^('
        r'-?\d+\.?\d*'          # numbers
        r'|true|false'            # booleans
        r'|on|off|yes|no'         # boolean aliases
        r'|#[0-9a-fA-F]{3,8}'    # colour codes
        r'|rgba?\([^)]+\)'       # rgb/rgba colours
        r'|[A-Za-z]:\\[^\s]+'   # Windows absolute paths
        r')$',
        re.IGNORECASE,
    )

    # Detect format-code strings (date/time formats, printf-like patterns).
    _FORMAT_STRING_RE = re.compile(
        r'(?:YYYY|MM|DD|HH|MI|SS|AM|PM|DY|%[dfsYmHMS])',
    )

    # Detect identifier-like values: camelCase, PascalCase, UPPER_CASE,
    # or single words with no spaces (ASCII-only).
    _IDENTIFIER_VALUE_RE = re.compile(
        r'^[A-Za-z_][A-Za-z0-9_]*$',
    )

    # Detect comma-separated identifier lists (e.g. "yellow,green,blue,red",
    # "normal,normal:red,blue"). Translating these corrupts plugin logic that
    # splits on ',' without trimming whitespace.
    _COMMA_SEPARATED_IDS_RE = re.compile(
        r'^[A-Za-z_][\w:.-]*(?:\s*,\s*[A-Za-z_][\w:.-]*)+$',
    )

    # Detect pipe-separated or semicolon-separated value lists
    # (e.g. "normal|dark|light", "walk;dash;jump").
    _SEPARATED_IDS_RE = re.compile(
        r'^[A-Za-z_][\w:.-]*(?:\s*[|;]\s*[A-Za-z_][\w:.-]*)+$',
    )

    # Detect resource / file paths (e.g. "img/pictures/title",
    # "audio/se/click", "fonts/mplus-1m-regular").
    _RESOURCE_PATH_RE = re.compile(
        r'^[A-Za-z_][\w-]*(?:/[A-Za-z_][\w.-]*)+$',
    )

    # Detect dot-notation object references (e.g. "Scene_Map.prototype",
    # "window.innerWidth", "Graphics._defaultStretchMode").
    _DOT_NOTATION_RE = re.compile(
        r'^[\$A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+$',
    )

    # Detect $game / $ variable references (e.g. "$gameSystem",
    # "$dataActors", "$gameParty").
    _GAME_VAR_RE = re.compile(
        r'^\$[A-Za-z_][\w]*$',
    )

    @classmethod
    def _is_untranslatable_value(cls, value: str) -> bool:
        """Return ``True`` if *value* matches any non-translatable plugin parameter pattern.

        Consolidates all regex-based safety filters into a single check.
        Used in both extraction and application paths.
        """
        return bool(
            cls._IDENTIFIER_VALUE_RE.match(value)
            or cls._COMMA_SEPARATED_IDS_RE.match(value)
            or cls._SEPARATED_IDS_RE.match(value)
            or cls._RESOURCE_PATH_RE.match(value)
            or cls._DOT_NOTATION_RE.match(value)
            or cls._GAME_VAR_RE.match(value)
            or cls._PLUGIN_SKIP_VALUE_RE.match(value)
        )

    def _extract_plugins_js(self) -> None:
        """Extract translatable text from plugin parameters in ``plugins.js``.

        Plugin parameters frequently contain UI labels, quest text, menu text,
        and other player-visible strings that the standard JSON extractor misses.
        """
        js_path = self.game_path / "www" / "js" / "plugins.js"
        if not js_path.exists():
            js_path = self.game_path / "js" / "plugins.js"
        if not js_path.exists():
            return

        try:
            content = js_path.read_text(encoding="utf-8")
        except Exception:
            return

        # Extract the JSON array from  var $plugins = [...];
        match = re.search(r'\$plugins\s*=\s*(\[[\s\S]*?\])\s*;', content)
        if not match:
            return

        try:
            plugins_data = json.loads(match.group(1))
        except json.JSONDecodeError:
            return

        if not isinstance(plugins_data, list):
            return

        for pi, plugin in enumerate(plugins_data):
            if not isinstance(plugin, dict):
                continue
            # Skip disabled plugins — their text is never shown.
            if not plugin.get("status", False):
                continue
            pname = plugin.get("name", f"Plugin{pi}")
            params = plugin.get("parameters")
            if not isinstance(params, dict):
                continue

            self._extract_plugin_params_toplevel(
                params, "plugins.js", f"[{pi}].parameters",
                f"Plugin: {pname}",
            )

        logger.debug("plugins.js: scanned %d plugin parameter sets", len(plugins_data))

    def _extract_plugin_params_toplevel(
        self, params: dict, file: str, base_path: str, context: str,
    ) -> None:
        """Walk top-level plugin parameter key/value pairs."""
        for key, value in params.items():
            if not isinstance(value, str) or not value.strip():
                continue

            value_s = value.strip()

            # Skip obviously non-translatable values.
            if self._PLUGIN_SKIP_VALUE_RE.match(value_s):
                continue

            # Skip top-level param keys that are known identifiers
            # (e.g. commandName, Command — plugin command aliases).
            if key.lower() in _PLUGIN_JSON_IDENTIFIER_KEYS:
                continue

            # Some parameters store JSON-encoded arrays/objects as strings.
            if value_s.startswith('{') or value_s.startswith('['):
                try:
                    parsed = json.loads(value_s)
                    if isinstance(parsed, list):
                        for li, item in enumerate(parsed):
                            if isinstance(item, str):
                                # Double-encoded JSON: a list item that is
                                # itself a JSON string (e.g. NUUN_SaveScreen
                                # ContentsList).  Try to decode one more level.
                                item_s = item.strip()
                                if item_s.startswith('{') or item_s.startswith('['):
                                    try:
                                        inner = json.loads(item_s)
                                        if isinstance(inner, dict):
                                            self._extract_plugin_params_json(
                                                inner, file,
                                                f"{base_path}.{key}.__json__[{li}]",
                                                context,
                                            )
                                            continue
                                        elif isinstance(inner, list):
                                            for li2, item2 in enumerate(inner):
                                                if isinstance(item2, str) and self._should_translate(item2):
                                                    self._add(
                                                        file,
                                                        f"{base_path}.{key}.__json__[{li}][{li2}]",
                                                        item2, f"{context} ({key})",
                                                        "plugin_text",
                                                    )
                                                elif isinstance(item2, dict):
                                                    self._extract_plugin_params_json(
                                                        item2, file,
                                                        f"{base_path}.{key}.__json__[{li}][{li2}]",
                                                        context,
                                                    )
                                            continue
                                    except (json.JSONDecodeError, ValueError):
                                        pass
                                # Plain string item.
                                if self._should_translate(item):
                                    self._add(
                                        file, f"{base_path}.{key}.__json__[{li}]",
                                        item, f"{context} ({key})",
                                        "plugin_text",
                                    )
                            elif isinstance(item, dict):
                                self._extract_plugin_params_json(
                                    item, file, f"{base_path}.{key}.__json__[{li}]",
                                    context,
                                )
                    elif isinstance(parsed, dict):
                        self._extract_plugin_params_json(
                            parsed, file, f"{base_path}.{key}.__json__",
                            context,
                        )
                    continue
                except (json.JSONDecodeError, ValueError):
                    pass  # Not valid JSON — treat as plain string.

            # Plain string parameter.
            # Skip format-code strings (date/time patterns).
            if self._FORMAT_STRING_RE.search(value_s):
                continue
            # Skip identifier-like values (camelCase, single_word, etc.).
            if self._IDENTIFIER_VALUE_RE.match(value_s):
                continue
            # Skip comma-separated identifier lists (e.g. "yellow,green,blue").
            if self._COMMA_SEPARATED_IDS_RE.match(value_s):
                continue
            # Skip other non-translatable value patterns.
            if self._is_untranslatable_value(value_s):
                continue
            if self._should_translate(value_s) and not self._looks_like_code(value_s):
                self._add(
                    file, f"{base_path}.{key}",
                    value_s, f"{context} ({key})",
                    "plugin_text", max_chars=80,
                )

    def _extract_plugin_params_json(
        self, obj: dict, file: str, base_path: str, context: str,
    ) -> None:
        """Recursively extract translatable values from JSON-decoded plugin parameter structures."""
        for key, value in obj.items():
            # Skip keys that are identifiers / config keys (case-insensitive).
            if key.lower() in _PLUGIN_JSON_IDENTIFIER_KEYS:
                continue
            # Also check legacy MZ skip-keys (exact case).
            if key in self._MZ_SKIP_KEYS:
                continue
            if isinstance(value, str):
                vs = value.strip()
                # Recursively decode doubly-encoded JSON strings.
                if vs.startswith('{') or vs.startswith('['):
                    try:
                        inner = json.loads(vs)
                        if isinstance(inner, dict):
                            self._extract_plugin_params_json(
                                inner, file, f"{base_path}.{key}", context,
                            )
                            continue
                        elif isinstance(inner, list):
                            for li, item in enumerate(inner):
                                if isinstance(item, str):
                                    # Try to decode nested JSON strings.
                                    item_s = item.strip()
                                    if item_s.startswith('{') or item_s.startswith('['):
                                        try:
                                            nested = json.loads(item_s)
                                            if isinstance(nested, dict):
                                                self._extract_plugin_params_json(
                                                    nested, file,
                                                    f"{base_path}.{key}.__json__[{li}]",
                                                    context,
                                                )
                                                continue
                                            elif isinstance(nested, list):
                                                for li2, item2 in enumerate(nested):
                                                    if isinstance(item2, str) and self._should_translate(item2):
                                                        self._add(
                                                            file,
                                                            f"{base_path}.{key}.__json__[{li}][{li2}]",
                                                            item2, f"{context} ({key})",
                                                            "plugin_text",
                                                        )
                                                    elif isinstance(item2, dict):
                                                        self._extract_plugin_params_json(
                                                            item2, file,
                                                            f"{base_path}.{key}.__json__[{li}][{li2}]",
                                                            context,
                                                        )
                                                continue
                                        except (json.JSONDecodeError, ValueError):
                                            pass
                                    # Plain string — translate if needed.
                                    if self._should_translate(item):
                                        self._add(
                                            file, f"{base_path}.{key}[{li}]",
                                            item, f"{context} ({key})",
                                            "plugin_text",
                                        )
                                elif isinstance(item, dict):
                                    self._extract_plugin_params_json(
                                        item, file, f"{base_path}.{key}[{li}]",
                                        context,
                                    )
                            continue
                    except (json.JSONDecodeError, ValueError):
                        pass
                # Skip format-code strings (date formats, printf patterns).
                if self._FORMAT_STRING_RE.search(vs):
                    continue
                # Skip identifier-like, separated lists, paths, object refs.
                if self._is_untranslatable_value(vs):
                    continue
                if self._should_translate(value) and not self._looks_like_code(value):
                    self._add(
                        file, f"{base_path}.{key}",
                        value, f"{context} ({key})",
                        "plugin_text",
                    )
            elif isinstance(value, dict):
                self._extract_plugin_params_json(
                    value, file, f"{base_path}.{key}", context,
                )
            elif isinstance(value, list):
                for li, item in enumerate(value):
                    if isinstance(item, str) and self._should_translate(item):
                        self._add(
                            file, f"{base_path}.{key}[{li}]",
                            item, f"{context} ({key})",
                            "plugin_text",
                        )
                    elif isinstance(item, dict):
                        self._extract_plugin_params_json(
                            item, file, f"{base_path}.{key}[{li}]", context,
                        )


    # -- plugin JS source file string extraction ----------------------------

    # Matches quoted string literals containing CJK text in JS source.
    _JS_CJK_STRING_RE = re.compile(
        r"""(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\')"""
    )

    # Regexes to strip JS comments before extracting strings.
    _JS_BLOCK_COMMENT_RE = re.compile(r'/\*[\s\S]*?\*/')
    _JS_LINE_COMMENT_RE = re.compile(r'//[^\n]*')

    # Strings that look like code/config rather than UI text.
    _JS_SKIP_PATTERNS = re.compile(
        r'(^\\[A-Z]|^\s*$|^/|^https?:|\.(js|json|png|ogg|m4a|wav|mp3)$'
        r'|\.prototype|\$game|function\s|=>|===)',
        re.IGNORECASE,
    )

    # Maximum length for a JS string literal to be considered UI text.
    _JS_STRING_MAX_LEN = 500

    # Maximum line length before a file is considered minified/obfuscated.
    _JS_MINIFIED_LINE_LEN = 5000

    def _extract_plugin_js_sources(self) -> None:
        """Extract CJK string literals from plugin JS source files.

        This catches hardcoded UI text (menu labels, button labels, help text)
        that is embedded directly in plugin source code rather than in
        ``plugins.js`` parameters.

        Comment blocks (``/* ... */`` and ``// ...``) are stripped before
        scanning so that developer-facing JSDoc documentation is not
        accidentally included.
        """
        js_dir = self.game_path / "www" / "js" / "plugins"
        if not js_dir.exists():
            js_dir = self.game_path / "js" / "plugins"
        if not js_dir.exists():
            return

        count = 0
        for js_file in sorted(js_dir.glob("*.js")):
            # Skip our own generated plugin.
            if js_file.name == "TranslationWordWrap.js":
                continue
            try:
                content = js_file.read_text(encoding="utf-8")
            except Exception:
                continue

            # Quick check: does this file even contain CJK characters?
            if not CJK_PATTERN.search(content):
                continue

            # Skip minified / obfuscated files — replacements in these are
            # extremely fragile and break runtime lookups.
            if any(
                len(line) > self._JS_MINIFIED_LINE_LEN
                for line in content.splitlines()
            ):
                logger.debug(
                    "Skipping minified plugin: %s", js_file.name,
                )
                continue

            # Strip all comments so JSDoc/header documentation is ignored.
            code_only = self._JS_BLOCK_COMMENT_RE.sub('', content)
            code_only = self._JS_LINE_COMMENT_RE.sub('', code_only)

            fname = f"js:{js_file.name}"
            for match in self._JS_CJK_STRING_RE.finditer(code_only):
                value = match.group(1) or match.group(2)
                if not value:
                    continue
                # Must actually contain CJK.
                if not CJK_PATTERN.search(value):
                    continue
                # Skip code-like strings.
                if self._JS_SKIP_PATTERNS.search(value):
                    continue
                # Skip very short values (single char can be punctuation).
                stripped_val = value.strip()
                if len(stripped_val) < 2:
                    continue
                # Skip excessively long strings (documentation, not UI).
                if len(stripped_val) > self._JS_STRING_MAX_LEN:
                    continue
                self._add(
                    fname, f"__js_string__{value}",
                    value, f"Plugin JS: {js_file.name}",
                    "plugin_text", max_chars=80,
                )
                count += 1

        if count:
            logger.info("Extracted %d CJK strings from plugin JS source files", count)


# ---------------------------------------------------------------------------
# JS Plugin Patcher
# ---------------------------------------------------------------------------

class PluginPatcher:
    """Patches JS plugins for translation compatibility."""

    def __init__(self, game_path: str) -> None:
        self.game_path = Path(game_path)
        self.js_path = self.game_path / "www" / "js" / "plugins"
        if not self.js_path.exists():
            self.js_path = self.game_path / "js" / "plugins"
        self.backup_path = self.game_path / "translation_project" / "js_backup"

    def patch_all(self, crash_logger: bool = True) -> Dict[str, bool]:
        """Apply all known patches, returning ``{name: success}``."""
        result: Dict[str, bool] = {
            "Live2dFix": self._create_live2d_fix_plugin(),
            "LL_GalgeChoiceWindowMV": self._patch_ll_galge(),
            "WordWrapPlugin": self._create_wordwrap_plugin(),
        }
        if crash_logger:
            result["CrashLogger"] = self._create_crash_logger_plugin()
        return result

    def _patch_ll_galge(self) -> bool:
        """Fix comma-splitting in LL_GalgeChoiceWindowMV when choices have spaces."""
        target = self.js_path / "LL_GalgeChoiceWindowMV.js"
        if not target.exists():
            return False

        self.backup_path.mkdir(parents=True, exist_ok=True)
        backup = self.backup_path / "LL_GalgeChoiceWindowMV.js"
        if not backup.exists():
            shutil.copy2(target, backup)

        content = target.read_text(encoding="utf-8")
        if "args.slice(1).join" in content:
            return True  # Already patched.

        old = "var choices = args[1].split(',');"
        if old not in content:
            return False

        patched = content.replace(
            old,
            'var rawChoices = args.slice(1).join(" ");\n'
            '        var choices = rawChoices.split(",");',
        )
        target.write_text(patched, encoding="utf-8")
        logger.info("Patched LL_GalgeChoiceWindowMV for space-safe choice splitting")
        return True

    def _compute_text_boundary(self) -> int:
        """Compute the right text boundary in bitmap coordinates.

        Analyses WindowBackImage frame images + NRP_MessageWindow sprite offset
        to determine where text should stop so it stays within the visible
        decorative frame.  Returns ``0`` when the boundary cannot be determined
        (the JS plugin falls back to the bitmap width in that case).

        WindowBackImage positions images with:
          sprite.x = windowWidth / 2 + OffsetX
          anchor.x = 0.5  (image is centered on sprite.x)
        So the right edge of the image in window coords is:
          windowWidth / 2 + OffsetX + imageWidth / 2
        """
        try:
            plugins_js = self.js_path.parent / "plugins.js"
            if not plugins_js.exists():
                return 0
            raw = plugins_js.read_text(encoding="utf-8")
            start = raw.find("\n[")
            if start < 0:
                start = raw.find("[")
            end = raw.rfind("]")
            if start < 0 or end < 0:
                return 0
            plugins: List[Dict[str, Any]] = json.loads(raw[start : end + 1].strip())

            # -- Game resolution (window width for the message box) ---------
            window_width = 816  # RPG Maker MZ default
            try:
                sys_path = self.game_path / "data" / "System.json"
                if not sys_path.exists():
                    sys_path = self.game_path / "www" / "data" / "System.json"
                if sys_path.exists():
                    sys_data = json.loads(sys_path.read_text(encoding="utf-8"))
                    adv = sys_data.get("advanced", {})
                    ui_w = adv.get("uiAreaWidth") or adv.get("screenWidth")
                    if ui_w:
                        window_width = int(ui_w)
            except Exception:
                pass

            # -- NRP_MessageWindow ------------------------------------------
            adjust_x = 0
            padding = 12  # MZ default
            nrp_width: Optional[int] = None
            for p in plugins:
                if not p.get("status"):
                    continue
                name = p.get("name", "")
                if "NRP_MessageWindow" in name:
                    params = p.get("parameters", {})
                    try:
                        adjust_x = int(params.get("AdjustMessageX", "0"))
                    except (ValueError, TypeError):
                        pass
                    # WindowWidth may be an expression like "Graphics.boxWidth"
                    ww_str = params.get("WindowWidth", "")
                    if ww_str:
                        try:
                            nrp_width = int(ww_str)
                        except ValueError:
                            # Common expression: Graphics.boxWidth → use screen width
                            if "boxWidth" in ww_str or "width" in ww_str.lower():
                                nrp_width = window_width

            if nrp_width is not None:
                window_width = nrp_width

            sprite_x = padding + adjust_x  # contentsSprite.x in window coords

            # -- WindowBackImage: find frame images for Window_Message ------
            # Positioning: sprite.x = windowWidth/2 + OffsetX; anchor.x = 0.5
            boundaries: List[int] = []
            for p in plugins:
                if not p.get("status"):
                    continue
                if "BackImage" not in p.get("name", ""):
                    continue
                params = p.get("parameters", {})
                for key in ("windowImageInfo", "WindowBackImageList", "WindowList"):
                    raw_list = params.get(key, "")
                    if not raw_list:
                        continue
                    try:
                        items = json.loads(raw_list)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    for item_raw in items:
                        try:
                            item = json.loads(item_raw) if isinstance(item_raw, str) else item_raw
                        except (json.JSONDecodeError, TypeError):
                            continue
                        if item.get("WindowClass") != "Window_Message":
                            continue
                        try:
                            offset_x = int(item.get("OffsetX", "0"))
                        except (ValueError, TypeError):
                            offset_x = 0
                        image_name = item.get("ImageFile", "")
                        if not image_name:
                            continue
                        img_w = self._get_image_width(image_name)
                        if img_w <= 0:
                            continue
                        # Image center in window coords:
                        center_x = window_width / 2 + offset_x
                        # Right edge (anchor 0.5 → half width on each side):
                        frame_right = center_x + img_w / 2
                        # Margin: gold border (~8px) + aesthetic padding (20px):
                        frame_inner_right = frame_right - 28
                        # Convert to bitmap coordinates:
                        bitmap_x = int(frame_inner_right - sprite_x)
                        boundaries.append(bitmap_x)

            if not boundaries:
                return 0

            result = max(0, min(boundaries))
            logger.info("Computed text right boundary: %d bitmap px", result)
            return result
        except Exception:
            logger.exception("Failed to compute text boundary")
            return 0

    def _get_image_width(self, name: str) -> int:
        """Return the pixel width of a game image (pictures / system folder)."""
        try:
            from PIL import Image as PILImage
        except ImportError:
            return 0

        search_dirs = [
            self.game_path / "img" / "pictures",
            self.game_path / "img" / "system",
        ]
        if (self.game_path / "www").is_dir():
            search_dirs.extend([
                self.game_path / "www" / "img" / "pictures",
                self.game_path / "www" / "img" / "system",
            ])
        for d in search_dirs:
            for ext in (".png", ".png_"):
                path = d / f"{name}{ext}"
                if not path.exists():
                    continue
                try:
                    im = PILImage.open(path)
                    return im.size[0]
                except Exception:
                    # Encrypted .png_ — try reading raw PNG IHDR
                    try:
                        return self._read_png_width_raw(path)
                    except Exception:
                        continue
        return 0

    @staticmethod
    def _read_png_width_raw(path: Path) -> int:
        """Read width from a PNG IHDR chunk (handles encrypted RPG Maker files)."""
        import struct

        with open(path, "rb") as fh:
            data = fh.read(128)
        idx = data.find(b"IHDR")
        if idx >= 0 and idx + 8 <= len(data):
            width = struct.unpack(">I", data[idx + 4 : idx + 8])[0]
            if 0 < width < 10000:
                return width
        return 0

    def _create_wordwrap_plugin(self) -> bool:
        """Create a per-line font-scaling JS plugin for translated text.

        Works on both MV and MZ.  At ``startMessage`` time the plugin
        measures each line at the default font size and computes the
        minimum font size that makes it fit.  The sizes are stored in
        an array on the window and applied via ``processNewLine`` /
        ``processNewPage`` hooks — no engine-specific escape codes needed.
        """
        max_right = self._compute_text_boundary()
        plugin_code = (
            "//=============================================================================\n"
            "// TranslationWordWrap.js — Per-line font scaling (MV + MZ)\n"
            "//=============================================================================\n"
            "/*:\n"
            " * @plugindesc Per-line dynamic font scaling for translated text.\n"
            " * @author JaimeDevCode — RPG Maker Translator\n"
            " * @param Enable\n"
            " * @type boolean\n"
            " * @default true\n"
            " */\n"
            "(function () {\n"
            "  'use strict';\n"
            "  var params = PluginManager.parameters('TranslationWordWrap');\n"
            "  if (params['Enable'] === 'false') return;\n"
            "\n"
            f"  var MAX_RIGHT = {max_right};  // 0 = auto from bitmap width\n"
            "  var MIN_FONT  = 14;\n"
            "\n"
            "  /* ---- helpers ---- */\n"
            "\n"
            "  function stripEsc(t) {\n"
            "    return t.replace(/\\x1b[A-Z]+\\[\\d+\\]/gi, '')\n"
            "            .replace(/\\x1b.(\\[\\d+\\])?/g, '');\n"
            "  }\n"
            "\n"
            "  function safeNewLineX(win, ts) {\n"
            "    try { return win.newLineX(ts); } catch(e) {}\n"
            "    try { return win.newLineX(); } catch(e) {}\n"
            "    return $gameMessage.faceName() !== '' ? 168 : 0;\n"
            "  }\n"
            "\n"
            "  function textRight(win) {\n"
            "    if (MAX_RIGHT > 0) return MAX_RIGHT;\n"
            "    var w = win.contents ? win.contents.width : (win.innerWidth || 0);\n"
            "    return Math.max(0, w);\n"
            "  }\n"
            "\n"
            "  function computeLineFontSize(win, cleanText, avail, defSize) {\n"
            "    if (!cleanText || cleanText.length === 0) return defSize;\n"
            "    win.contents.fontSize = defSize;\n"
            "    var w = win.textWidth(cleanText);\n"
            "    if (w <= avail) return defSize;\n"
            "    var needed = Math.max(MIN_FONT, Math.floor(defSize * avail / w));\n"
            "    win.contents.fontSize = needed;\n"
            "    while (win.textWidth(cleanText) > avail && needed > MIN_FONT) {\n"
            "      needed--;\n"
            "      win.contents.fontSize = needed;\n"
            "    }\n"
            "    return needed;\n"
            "  }\n"
            "\n"
            "  /* ---- startMessage: compute per-line sizes ---- */\n"
            "\n"
            "  var _startMsg = Window_Message.prototype.startMessage;\n"
            "  Window_Message.prototype.startMessage = function () {\n"
            "    this._twLineSizes = null;\n"
            "    this._twLineIdx  = 0;\n"
            "    _startMsg.call(this);\n"
            "    var ts = this._textState;\n"
            "    if (!ts || !this.contents) return;\n"
            "    var right = textRight(this);\n"
            "    var left  = safeNewLineX(this, ts);\n"
            "    var avail = right - left;\n"
            "    if (avail <= 0) return;\n"
            "    var defSize = this.contents.fontSize;\n"
            "    var lines = ts.text.split('\\n');\n"
            "    var sizes = [];\n"
            "    var anyScaled = false;\n"
            "    for (var i = 0; i < lines.length; i++) {\n"
            "      var s = computeLineFontSize(this, stripEsc(lines[i]), avail, defSize);\n"
            "      sizes.push(s);\n"
            "      if (s < defSize) anyScaled = true;\n"
            "    }\n"
            "    this.contents.fontSize = defSize;\n"
            "    if (anyScaled) {\n"
            "      this._twLineSizes = sizes;\n"
            "      this._twLineIdx  = 0;\n"
            "      // Apply first line size immediately\n"
            "      this.contents.fontSize = sizes[0];\n"
            "    }\n"
            "  };\n"
            "\n"
            "  /* ---- processNewLine: switch font for next line ---- */\n"
            "\n"
            "  var _processNewLine = Window_Message.prototype.processNewLine;\n"
            "  Window_Message.prototype.processNewLine = function (textState) {\n"
            "    _processNewLine.call(this, textState);\n"
            "    if (this._twLineSizes) {\n"
            "      this._twLineIdx++;\n"
            "      var idx = this._twLineIdx;\n"
            "      if (idx < this._twLineSizes.length) {\n"
            "        this.contents.fontSize = this._twLineSizes[idx];\n"
            "      }\n"
            "    }\n"
            "  };\n"
            "\n"
            "  /* ---- processNewPage: reset to first line ---- */\n"
            "\n"
            "  var _processNewPage = Window_Message.prototype.processNewPage;\n"
            "  Window_Message.prototype.processNewPage = function (textState) {\n"
            "    _processNewPage.call(this, textState);\n"
            "    if (this._twLineSizes) {\n"
            "      this._twLineIdx = 0;\n"
            "      this.contents.fontSize = this._twLineSizes[0];\n"
            "    }\n"
            "  };\n"
            "\n"
            "  /* ---- terminateMessage: clean up ---- */\n"
            "\n"
            "  var _termMsg = Window_Message.prototype.terminateMessage;\n"
            "  Window_Message.prototype.terminateMessage = function () {\n"
            "    this._twLineSizes = null;\n"
            "    this._twLineIdx  = 0;\n"
            "    _termMsg.call(this);\n"
            "  };\n"
            "\n"
            "  /* ---- Fallback: character-level safety clamp ---- */\n"
            "\n"
            "  var _procChar = Window_Message.prototype.processNormalCharacter;\n"
            "  Window_Message.prototype.processNormalCharacter = function (ts) {\n"
            "    var c = ts.text[ts.index];\n"
            "    var w = this.textWidth(c);\n"
            "    var right = textRight(this);\n"
            "    if (ts.x + w > right) {\n"
            "      ts.text = ts.text.slice(0, ts.index) + '\\n' + ts.text.slice(ts.index);\n"
            "      return;\n"
            "    }\n"
            "    _procChar.call(this, ts);\n"
            "  };\n"
            "})();\n"
        )
        try:
            self.js_path.mkdir(parents=True, exist_ok=True)
            (self.js_path / "TranslationWordWrap.js").write_text(plugin_code, encoding="utf-8")
            self._register_plugin_in_js("TranslationWordWrap")
            return True
        except Exception:
            logger.exception("Failed to create word-wrap plugin")
            return False

    def _create_live2d_fix_plugin(self) -> bool:
        """Deploy TranslationLive2dFix.js to null-guard enc_lv2d.js hooks.

        The fix plugin MUST load before enc_lv2d.js so it can save
        original RPG Maker methods in Phase 1.  It is registered at
        position "first" in plugins.js.  If enc_lv2d.js is not present
        in the game, the plugin is not deployed.
        """
        # Only deploy if the game actually uses enc_lv2d.js
        enc_lv2d_found = False
        for js_dir in (self.game_path / "www" / "js" / "plugins",
                       self.game_path / "js" / "plugins"):
            if (js_dir / "enc_lv2d.js").exists():
                enc_lv2d_found = True
                break
        if not enc_lv2d_found:
            return False

        # Also restore enc_lv2d.js from backup in case a previous run
        # applied the (now removed) Proxy stub.
        js_backup = self.game_path / "translation_project" / "js_backup" / "enc_lv2d.js"
        lv2d = js_dir / "enc_lv2d.js"
        if js_backup.exists():
            shutil.copy2(js_backup, lv2d)
            logger.debug("Restored enc_lv2d.js from js_backup")

        src = Path(__file__).resolve().parent / "plugins" / "TranslationLive2dFix.js"
        if not src.exists():
            logger.debug("TranslationLive2dFix.js not found in plugins/ folder")
            return False
        try:
            self.js_path.mkdir(parents=True, exist_ok=True)
            dest = self.js_path / "TranslationLive2dFix.js"
            shutil.copy2(src, dest)
            self._register_plugin_in_js(
                "TranslationLive2dFix",
                params={"Enable": "true"},
                position="first",
            )
            return True
        except Exception:
            logger.exception("Failed to create Live2D fix plugin")
            return False

    def _create_crash_logger_plugin(self) -> bool:
        """Deploy TranslationCrashLogger.js to capture errors to crash_log.txt."""
        src = Path(__file__).resolve().parent / "plugins" / "TranslationCrashLogger.js"
        if not src.exists():
            logger.debug("TranslationCrashLogger.js not found in plugins/ folder")
            return False
        try:
            self.js_path.mkdir(parents=True, exist_ok=True)
            dest = self.js_path / "TranslationCrashLogger.js"
            shutil.copy2(src, dest)
            self._register_plugin_in_js(
                "TranslationCrashLogger",
                params={"ShowAlert": "true", "PreventClose": "true"},
                position="first",
            )
            return True
        except Exception:
            logger.exception("Failed to create crash-logger plugin")
            return False

    def _register_plugin_in_js(self, name: str, params: Optional[Dict[str, str]] = None, position: str = "last") -> None:
        """Ensure *name* appears in ``plugins.js`` so the engine loads it."""
        plugins_js = self.js_path.parent / "plugins.js"
        if not plugins_js.exists():
            return
        raw = plugins_js.read_text(encoding="utf-8")
        if f'"name":"{name}"' in raw or f'"name": "{name}"' in raw:
            return  # Already registered.
        plugin_params = params if params else {"Enable": "true"}
        entry = json.dumps(
            {"name": name, "status": True, "description": "", "parameters": plugin_params},
            ensure_ascii=False,
        )
        raw = raw.rstrip()
        if raw.endswith("];"):
            if position == "first":
                # Insert right after the opening [
                idx = raw.index("[")
                raw = raw[: idx + 1] + "\n  " + entry + "," + raw[idx + 1 :]
            else:
                # Insert before the closing ];
                raw = raw[:-2].rstrip() + ",\n  " + entry + "\n];\n"
        plugins_js.write_text(raw, encoding="utf-8")


# ---------------------------------------------------------------------------
# Translation Applier
# ---------------------------------------------------------------------------

class TranslationApplier:
    """Writes translated text back into the game's JSON data files."""

    def __init__(self, game_path: str, project: TranslationProject) -> None:
        self.game_path = Path(game_path)
        self.project = project
        self.wrapper = TextWrapper()

        self.data_path = self.game_path / "www" / "data"
        if not self.data_path.exists():
            self.data_path = self.game_path / "data"

        self.backup_path = self.game_path / "translation_project" / "backup"

        # Group translated entries by file.
        self._by_file: Dict[str, Dict[str, TranslationEntry]] = {}
        for entry in project.entries:
            if entry.translated and entry.translated != entry.original:
                self._by_file.setdefault(entry.file, {})[entry.path] = entry

    # -- JSON path resolution ------------------------------------------------

    @staticmethod
    def _parse_path(path: str) -> List[Any]:
        """Convert a string path like ``[3].pages[0].list[5]`` to a list of keys."""
        parts: List[Any] = []
        current = ""
        i = 0
        while i < len(path):
            ch = path[i]
            if ch == "[":
                if current:
                    parts.append(current)
                    current = ""
                j = path.index("]", i)
                token = path[i + 1 : j]
                try:
                    parts.append(int(token))
                except ValueError:
                    parts.append(token)
                i = j + 1
            elif ch == ".":
                if current:
                    parts.append(current)
                    current = ""
                i += 1
            else:
                current += ch
                i += 1
        if current:
            parts.append(current)
        return parts

    def _set_value(self, data: Any, path: str, value: str) -> bool:
        """Set a value inside a nested structure following *path*."""
        parts = self._parse_path(path)
        obj = data
        try:
            for part in parts[:-1]:
                obj = obj[part]
            obj[parts[-1]] = value
            return True
        except (KeyError, IndexError, TypeError):
            return False

    @staticmethod
    def _escape_variable_string(value: str) -> str:
        """Escape inner single-quotes in a code-122 variable string value.

        RPG Maker stores these as ``'text'`` and ``eval()``s them at runtime,
        so any unescaped ``'`` inside the content causes a SyntaxError.
        Existing backslash-escaped quotes are left alone.
        """
        if len(value) >= 2 and value.startswith("'") and value.endswith("'"):
            inner = value[1:-1]
            # Escape unescaped single-quotes (don't double-escape).
            inner = inner.replace("\\'", "\x00")
            inner = inner.replace("'", "\\'")
            inner = inner.replace("\x00", "\\'")
            return f"'{inner}'"
        return value

    _QUOTE_NORMALIZE_RE = re.compile(r"'{2,}|`{2,}")

    @classmethod
    def _normalize_quotes(cls, text: str) -> str:
        """Replace Google Translate quote artifacts ('' and ``) with double-quotes."""
        return cls._QUOTE_NORMALIZE_RE.sub('"', text)

    # -- apply ---------------------------------------------------------------

    def apply_all(self, backup: bool = True, wordwrap: bool = True) -> None:
        """Apply all translations to the game's data files."""
        if backup:
            self.backup_path.mkdir(parents=True, exist_ok=True)

        for filename, entries in self._by_file.items():
            # plugins.js lives outside the data directory — handle separately.
            if filename == "plugins.js":
                self._apply_plugins_js(entries, backup)
                continue

            # Plugin JS source files use "js:PluginName.js" as their filename.
            if filename.startswith("js:"):
                self._apply_plugin_js_source(filename, entries, backup)
                continue

            filepath = self.data_path / filename
            if not filepath.exists():
                continue

            data = self._load_json(filepath)
            if data is None:
                continue

            if backup and not (self.backup_path / filename).exists():
                shutil.copy2(filepath, self.backup_path / filename)

            modified = False
            for path, entry in entries.items():
                translated = entry.translated
                # Normalize quote artifacts from machine translation.
                translated = self._normalize_quotes(translated)
                # NOTE: Static newline insertion is disabled because the
                # runtime TranslationWordWrap.js plugin handles font scaling
                # and fallback wrapping at the pixel level.
                # if wordwrap and entry.entry_type in ("text", "escape_text"):
                #     translated = self.wrapper.wrap(translated, entry.has_face)

                if "__plugin_choices__" in path:
                    modified |= self._apply_plugin_choices(data, path, translated)
                elif "__plugin_text__" in path or "__plugin_generic__" in path:
                    modified |= self._apply_plugin_text(data, path, translated)
                elif "__note__" in path:
                    modified |= self._apply_note_tag(data, path, entry.original, translated)
                elif "__script_string__" in path:
                    modified |= self._apply_script_string(data, path, entry.original, translated)
                else:
                    # Code 122 (Control Variable) string values are eval'd
                    # by the engine, so inner single-quotes must be escaped.
                    if "Variable String" in entry.context:
                        translated = self._escape_variable_string(translated)
                    modified |= self._set_value(data, path, translated)

            if modified:
                self._save_json(filepath, data)
                logger.info("Updated %s", filename)

    # -- plugins.js apply ----------------------------------------------------

    def _apply_plugins_js(
        self, entries: Dict[str, TranslationEntry], backup: bool
    ) -> None:
        """Apply translations to plugin parameters in ``plugins.js``."""
        js_path = self.game_path / "www" / "js" / "plugins.js"
        if not js_path.exists():
            js_path = self.game_path / "js" / "plugins.js"
        if not js_path.exists():
            logger.warning("plugins.js not found — skipping plugin parameter translations")
            return

        try:
            content = js_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not read plugins.js: %s", exc)
            return

        match = re.search(r'(\$plugins\s*=\s*)(\[[\s\S]*?\])(\s*;)', content)
        if not match:
            logger.warning("plugins.js does not contain $plugins array")
            return

        try:
            plugins_data = json.loads(match.group(2))
        except json.JSONDecodeError:
            logger.warning("Failed to parse $plugins JSON array")
            return

        if backup:
            self.backup_path.mkdir(parents=True, exist_ok=True)
            backup_file = self.backup_path / "plugins.js"
            if not backup_file.exists():
                shutil.copy2(js_path, backup_file)

        modified = False
        for path, entry in entries.items():
            translated = entry.translated
            if "__json__" in path:
                modified |= self._apply_plugin_json_value(
                    plugins_data, path, translated,
                )
            else:
                # Safety: skip top-level params whose key is an identifier.
                last_dot = path.rfind(".")
                if last_dot >= 0:
                    param_key = path[last_dot + 1 :]
                    if param_key.lower() in _PLUGIN_JSON_IDENTIFIER_KEYS:
                        continue
                # Safety: skip format strings and non-translatable value patterns.
                orig = entry.original
                if orig and TextExtractor._FORMAT_STRING_RE.search(orig):
                    continue
                if orig and TextExtractor._is_untranslatable_value(orig.strip()):
                    continue
                # Safety: skip values that are JSON strings (should have been
                # extracted as __json__ paths, not plain strings).
                if orig:
                    os = orig.strip()
                    if (os.startswith("{") or os.startswith("[")) and len(os) > 2:
                        continue
                modified |= self._set_value(plugins_data, path, translated)

        if modified:
            new_json = json.dumps(plugins_data, ensure_ascii=False, indent=2)
            new_content = (
                content[: match.start(2)] + new_json + content[match.end(2) :]
            )
            js_path.write_text(new_content, encoding="utf-8")
            logger.info("Updated plugins.js (%d parameter translations)", len(entries))

    def _apply_plugin_json_value(
        self, data: Any, path: str, translated: str
    ) -> bool:
        """Apply a translation inside a JSON-encoded plugin parameter string.

        Path format: ``[i].parameters.Key.__json__[0].field``
        Strategy: split at ``.__json__``, navigate to the JSON string,
        parse it, apply the translation to the inner path, re-encode.
        Handles doubly-encoded JSON (e.g. NUUN_SaveScreen ``ContentsList``
        where each list item is itself a JSON string containing the real
        object).
        """
        parts = path.split(".__json__", 1)
        if len(parts) != 2:
            return False

        outer_path = parts[0]   # e.g. "[0].parameters.Key"
        inner_path = parts[1]   # e.g. "[0].field" or ".field"

        outer_parts = self._parse_path(outer_path)
        inner_parts = self._parse_path(inner_path)
        if not outer_parts or not inner_parts:
            return False

        # Safety: refuse to overwrite identifier keys inside plugin JSON.
        final_key = inner_parts[-1]
        if isinstance(final_key, str) and final_key.lower() in _PLUGIN_JSON_IDENTIFIER_KEYS:
            return False

        # Safety: refuse to apply translated JSON strings — these are
        # JSON-encoded objects that got sent to translation as plain text.
        ts = translated.strip()
        if (ts.startswith("{") or ts.startswith("[")) and len(ts) > 2:
            try:
                json.loads(ts)
                return False  # Valid JSON — skip
            except (json.JSONDecodeError, ValueError):
                pass  # Not valid JSON — allow

        try:
            # Navigate to the JSON-encoded string parameter.
            obj = data
            for p in outer_parts[:-1]:
                obj = obj[p]
            key = outer_parts[-1]
            json_str = obj[key] if isinstance(key, int) else obj.get(key, "")
            if not isinstance(json_str, str):
                return False

            parsed = json.loads(json_str)

            # Navigate inside the parsed JSON, decoding any doubly-encoded
            # strings along the way.
            inner_obj = parsed
            re_encode_stack: list = []  # (container, key, was_string) for re-encoding
            for p in inner_parts[:-1]:
                child = inner_obj[p]
                if isinstance(child, str):
                    # Doubly-encoded JSON string — decode one more level.
                    try:
                        decoded = json.loads(child)
                    except (json.JSONDecodeError, ValueError):
                        return False
                    re_encode_stack.append((inner_obj, p))
                    inner_obj = decoded
                else:
                    inner_obj = child

            inner_obj[inner_parts[-1]] = translated

            # Re-encode inner objects back to JSON strings (reverse order).
            for container, idx in reversed(re_encode_stack):
                container[idx] = json.dumps(
                    inner_obj, ensure_ascii=False, separators=(",", ":"),
                )
                inner_obj = container

            # Re-encode the outer JSON string.
            new_json_str = json.dumps(
                parsed, ensure_ascii=False, separators=(",", ":"),
            )
            if isinstance(key, int):
                obj[key] = new_json_str
            else:
                obj[key] = new_json_str
            return True
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return False

    # -- plugin JS source apply ----------------------------------------------

    def _collect_plugin_param_keys(self, js_name: str, content: str) -> Set[str]:
        """Return the set of parameter lookup keys for a plugin.

        Collects from multiple sources:
        1. ``@param`` names in the plugin header comment block.
        2. Parameter keys stored in ``plugins.js`` for **all** plugins
           (cross-plugin parameter access is common).
        3. Bracket-access property keys (``obj['key']`` patterns) found in
           the executable code — these are runtime lookups that must stay
           in sync with whatever object they reference.
        4. Plugin command strings (``command === 'xxx'`` patterns).

        These strings must **never** be translated in executable code.
        """
        keys: Set[str] = set()

        # 1) @param names from all comment header blocks (/*: ... */ or /*:ja ... */)
        for header_m in re.finditer(r'/\*:[\s\S]*?\*/', content):
            for pm in re.finditer(r'@param\s+(.+)', header_m.group()):
                keys.add(pm.group(1).strip())

        # 2) Parameter keys from plugins.js — ALL plugins, not just this one.
        #    Cross-plugin access (e.g. AddFunctionByVillaA reading
        #    TsumioGatheringSystem params) is common.
        for prefix in (self.game_path / "www" / "js", self.game_path / "js"):
            pjs_path = prefix / "plugins.js"
            if not pjs_path.exists():
                continue
            try:
                pjs_text = pjs_path.read_text(encoding="utf-8")
                pjs_start = pjs_text.index("[")
                pjs_data = json.loads(pjs_text[pjs_start:].rstrip().rstrip(";"))
                for p in pjs_data:
                    for key in p.get("parameters", {}).keys():
                        keys.add(key)
            except Exception:
                pass
            break  # only use first found plugins.js

        # 3) Bracket-access property keys: obj['key'] or obj["key"]
        #    These are runtime lookups that must match their source data.
        for bm in re.finditer(r"""\[['"]([^'"]{2,})['"]\]""", content):
            keys.add(bm.group(1))

        # 4) Plugin command strings: command === 'xxx'
        for cmd_m in re.finditer(
            r"""(?:command\s*[!=]==?\s*['"]([^'"]+)['"]|['"]([^'"]+)['"]\s*[!=]==?\s*command)""",
            content,
        ):
            val = cmd_m.group(1) or cmd_m.group(2)
            if val:
                keys.add(val)

        return keys

    def _apply_plugin_js_source(
        self, filename: str, entries: Dict[str, TranslationEntry], backup: bool,
    ) -> None:
        """Apply translations to hardcoded strings in a plugin JS source file.

        *filename* has the format ``js:PluginName.js``.  Each entry path is
        ``__js_string__<original_text>``.  We do a literal string replacement
        inside the JS source, matching both single- and double-quoted forms.

        Safety guards (prevent breaking executable code):
        - Strings that match ``@param`` names or ``plugins.js`` parameter keys
          are skipped (they are configuration lookup keys).
        - Quotes inside translated text are escaped for the enclosing quote
          style so JS string syntax is preserved.
        """
        js_name = filename[3:]  # strip "js:" prefix
        js_path = self.game_path / "www" / "js" / "plugins" / js_name
        if not js_path.exists():
            js_path = self.game_path / "js" / "plugins" / js_name
        if not js_path.exists():
            logger.warning("Plugin JS file not found: %s", js_name)
            return

        try:
            content = js_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not read %s: %s", js_name, exc)
            return

        # Skip minified / obfuscated files.
        if any(len(ln) > 5000 for ln in content.splitlines()):
            logger.info(
                "Skipping minified plugin (unsafe to modify): %s", js_name,
            )
            return

        if backup:
            self.backup_path.mkdir(parents=True, exist_ok=True)
            backup_file = self.backup_path / js_name
            if not backup_file.exists():
                shutil.copy2(js_path, backup_file)

        # Build set of parameter keys that must NOT be translated.
        param_keys = self._collect_plugin_param_keys(js_name, content)

        modified = False
        for path, entry in entries.items():
            original = entry.original
            translated = entry.translated
            if not translated or translated == original:
                continue

            # Skip strings that are plugin parameter lookup keys.
            if original in param_keys:
                logger.debug(
                    "Skipping param key in %s: %s", js_name, original[:60],
                )
                continue

            # Replace with proper quote escaping to avoid breaking JS syntax.
            for quote in ('"', "'"):
                old = f"{quote}{original}{quote}"
                if old in content:
                    # Escape backslashes first, then the enclosing quote char.
                    safe = translated.replace("\\", "\\\\").replace(
                        quote, f"\\{quote}"
                    )
                    new = f"{quote}{safe}{quote}"
                    content = content.replace(old, new)
                    modified = True

        if modified:
            js_path.write_text(content, encoding="utf-8")
            logger.info("Updated plugin JS: %s", js_name)

    # -- plugin helpers ------------------------------------------------------

    def _apply_plugin_choices(self, data: Any, path: str, translated: str) -> bool:
        clean_path = path.replace(".__plugin_choices__", "")
        parts = self._parse_path(clean_path)
        try:
            obj = data
            for p in parts[:-1]:
                obj = obj[p]
            key = parts[-1]
            original_cmd: str = obj[key] if isinstance(key, int) else obj.get(key, "")
            if "LL_GalgeChoiceWindowMV setChoices" in original_cmd:
                obj[key] = f"LL_GalgeChoiceWindowMV setChoices {translated}"
                return True
        except (KeyError, IndexError, TypeError):
            pass
        return False

    def _apply_plugin_text(self, data: Any, path: str, translated: str) -> bool:
        clean_path = path.replace(".__plugin_text__", "").replace(".__plugin_generic__", "")
        parts = self._parse_path(clean_path)
        try:
            obj = data
            for p in parts[:-1]:
                obj = obj[p]
            key = parts[-1]
            original_cmd: str = obj[key] if isinstance(key, int) else obj.get(key, "")
            cmd_parts = original_cmd.split(" ", 2)
            if len(cmd_parts) >= 3:
                obj[key] = f"{cmd_parts[0]} {cmd_parts[1]} {translated}"
                return True
        except (KeyError, IndexError, TypeError):
            pass
        return False

    def _apply_note_tag(
        self, data: Any, path: str, original: str, translated: str
    ) -> bool:
        """Replace a translated note-tag value inside the ``note`` field.

        Path format: ``[idx].note.__note__tag_name``
        Strategy: find ``<Tag:original>`` in the note string and replace
        the value portion with the translated text.
        """
        # Strip __note__xxx suffix to get the note field path.
        note_suffix = re.search(r"\.__note__(.+)$", path)
        if not note_suffix:
            return False
        clean_path = path[: note_suffix.start()]
        parts = self._parse_path(clean_path)
        try:
            obj = data
            for p in parts[:-1]:
                obj = obj[p]
            key = parts[-1]
            note_str: str = obj[key] if isinstance(key, int) else obj.get(key, "")
            if not note_str or original not in note_str:
                return False
            obj[key] = note_str.replace(original, translated, 1)
            return True
        except (KeyError, IndexError, TypeError):
            return False

    def _apply_script_string(
        self, data: Any, path: str, original: str, translated: str
    ) -> bool:
        """Replace a translated string literal inside a script call.

        Path format: ``events[n].pages[p].list[i].parameters[0].__script_string__``
        Strategy: find the original string (in quotes) in the script text
        and replace it with the translated text.
        """
        clean_path = path.replace(".__script_string__", "")
        parts = self._parse_path(clean_path)
        try:
            obj = data
            for p in parts[:-1]:
                obj = obj[p]
            key = parts[-1]
            script_str: str = obj[key] if isinstance(key, int) else obj.get(key, "")
            if not script_str:
                return False
            # Try both quote styles.
            for quote in ('"', "'"):
                target = f"{quote}{original}{quote}"
                if target in script_str:
                    obj[key] = script_str.replace(
                        target, f"{quote}{translated}{quote}", 1
                    )
                    return True
            # Fallback: raw replacement (less safe).
            if original in script_str:
                obj[key] = script_str.replace(original, translated, 1)
                return True
        except (KeyError, IndexError, TypeError):
            pass
        return False

    # -- I/O -----------------------------------------------------------------

    @staticmethod
    def _load_json(path: Path) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Main Translator (orchestrator)
# ---------------------------------------------------------------------------

class RPGMakerTranslator:
    """High-level orchestrator: extract → translate → apply."""

    def __init__(
        self,
        game_path: str,
        source_lang: str = "ja",
        target_lang: str = "en",
        backend: str = "google",
        api_key: Optional[str] = None,
        glossary_path: Optional[str] = None,
    ) -> None:
        self.game_path = Path(game_path)
        self.source_lang = source_lang
        self.target_lang = target_lang

        self.project_dir = self.game_path / "translation_project"
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.project_file = self.project_dir / "project_v2.json"
        self.csv_file = self.project_dir / f"translations_{target_lang}_v2.csv"

        self.backend = create_backend(backend, api_key, source_lang, target_lang)
        self.wrapper = TextWrapper()
        self.patcher = PluginPatcher(str(game_path))
        self.project: Optional[TranslationProject] = None

        # -- Glossary / Dictionary -------------------------------------------
        gpath = Path(glossary_path) if glossary_path else self.project_dir / "glossary.json"
        self.glossary = TranslationDictionary(str(gpath))
        self.glossary_file = gpath

        # Load existing project if available (resumes interrupted work).
        if self.project_file.exists():
            try:
                with open(self.project_file, "r", encoding="utf-8") as fh:
                    self.project = TranslationProject.from_dict(json.load(fh))
                logger.info(
                    "Resumed project with %d entries (%s)",
                    len(self.project.entries), self.backend.name,
                )
            except Exception:
                pass

    # -- persistence ---------------------------------------------------------

    def _save(self) -> None:
        if not self.project:
            return
        self.project.modified = datetime.now().isoformat()
        with open(self.project_file, "w", encoding="utf-8") as fh:
            json.dump(self.project.to_dict(), fh, ensure_ascii=False, indent=2)

    # -- pipeline steps ------------------------------------------------------

    def _restore_backups(self) -> int:
        """Restore original data files from backup so extraction reads untranslated text.

        Returns the number of files restored.
        """
        backup_dir = self.game_path / "translation_project" / "backup"
        if not backup_dir.exists():
            return 0

        data_path = self.game_path / "www" / "data"
        if not data_path.exists():
            data_path = self.game_path / "data"

        restored = 0
        for backup_file in backup_dir.iterdir():
            if backup_file.suffix == ".json":
                target = data_path / backup_file.name
                if target.exists():
                    shutil.copy2(backup_file, target)
                    restored += 1
        return restored

    def _backup_project_file(self) -> None:
        """Create a rotating backup of the project file before extraction.

        Keeps up to 3 backups so the user can recover if something goes wrong.
        """
        if not self.project_file.exists():
            return
        for i in range(2, 0, -1):
            src = self.project_file.with_suffix(f".json.bak{i}")
            dst = self.project_file.with_suffix(f".json.bak{i + 1}")
            if src.exists():
                shutil.copy2(src, dst)
        dst = self.project_file.with_suffix(".json.bak1")
        shutil.copy2(self.project_file, dst)

    def extract(self) -> TranslationProject:
        """Step 1 – Extract all translatable text from the game.

        If a previous run already applied translations to the JSON files,
        we restore from backup first so that extraction reads the original
        (untranslated) text.  This prevents hash mismatches during merge.
        """
        old_project = self.project  # may contain translations from a previous run

        # Safety: back up the project file before overwriting.
        self._backup_project_file()

        # Restore backups before extraction so we read original text.
        restored = self._restore_backups()
        if restored:
            logger.info(
                "Restored %d data files from backup before re-extraction", restored,
            )

        self.project = TextExtractor(str(self.game_path), self.source_lang).extract_all()
        self.project.target_lang = self.target_lang

        # Merge previously saved translations so we don't lose progress.
        if old_project and old_project.entries:
            old_map: Dict[str, TranslationEntry] = {
                e.hash: e for e in old_project.entries if e.translated
            }
            # Secondary lookup by (file, path) for entries whose hash changed
            # (e.g. extraction logic was updated but the entry is the same).
            old_by_loc: Dict[str, TranslationEntry] = {
                f"{e.file}:{e.path}": e for e in old_project.entries if e.translated
            }
            if old_map or old_by_loc:
                merged = 0
                for entry in self.project.entries:
                    if entry.translated:
                        continue
                    prev = old_map.get(entry.hash)
                    if not prev:
                        # Fallback: match by file + path (same location, text may differ).
                        prev = old_by_loc.get(f"{entry.file}:{entry.path}")
                    if prev:
                        entry.translated = prev.translated
                        merged += 1
                if merged:
                    logger.info(
                        "Merged %d existing translations from previous run", merged,
                    )

        self._save()
        return self.project

    def translate(
        self,
        batch_size: int = 50,
        delay: float = 0.5,
        skip_translated: bool = True,
    ) -> None:
        """Step 2 – Translate all entries using the configured backend.

        The glossary is consulted first: any entry whose original text
        has an exact match in the glossary is filled in immediately,
        skipping the API call.  After translation, newly translated
        short terms (names, items, etc.) are auto-learned into the
        glossary for future consistency.
        """
        if not self.project:
            self.extract()

        assert self.project is not None

        # --- Phase 1: Apply glossary matches before calling the API ---------
        glossary_applied = self.glossary.apply_to_entries(
            self.project.entries, only_untranslated=True
        )
        if glossary_applied:
            self._save()

        # --- Phase 2: Translate remaining entries via API -------------------
        todo = [
            e for e in self.project.entries
            if not (skip_translated and e.translated)
        ]
        if not todo:
            logger.info("All entries already translated — nothing to do")
            self._post_translate_glossary()
            return

        total = len(todo)
        logger.info(
            "Translating %d entries with %s (batch=%d) …",
            total, self.backend.name, batch_size,
        )

        for i in range(0, total, batch_size):
            batch = todo[i : i + batch_size]
            texts = [e.original for e in batch]

            try:
                translations = self.backend.translate_batch(
                    texts, self.source_lang, self.target_lang,
                )
                for entry, translated in zip(batch, translations):
                    entry.translated = translated

                self._save()
                done = min(i + batch_size, total)
                pct = round(done / total * 100, 1)
                logger.info("  [%d/%d] %s%% complete", done, total, pct)

                if i + batch_size < total:
                    time.sleep(delay)
            except Exception as exc:
                logger.error("Translation batch failed: %s", exc)
                self._save()  # Save partial progress.
                raise

        # --- Phase 3: Auto-learn new terms into glossary --------------------
        self._post_translate_glossary()

    def _post_translate_glossary(self) -> None:
        """Auto-learn terms from the translated project and save glossary."""
        if self.project:
            self.glossary.learn_from_project(self.project)
        if self.glossary.dirty:
            self.glossary.save()
            logger.info(
                "Glossary saved: %d terms → %s",
                self.glossary.size, self.glossary_file.name,
            )

    def apply(self, backup: bool = True, wordwrap: bool = True) -> None:
        """Step 3 – Write translations back to game files."""
        if not self.project:
            return
        TranslationApplier(str(self.game_path), self.project).apply_all(backup, wordwrap)

    def patch_plugins(self, crash_logger: bool = True) -> Dict[str, bool]:
        """Patch JS plugins for translation compatibility."""
        return self.patcher.patch_all(crash_logger=crash_logger)

    # -- CSV import / export -------------------------------------------------

    def export_csv(self, path: Optional[str] = None) -> Path:
        """Export the project to an editable CSV (UTF-8 with BOM)."""
        if not self.project:
            raise RuntimeError("No project loaded; run extract() first.")

        out = Path(path) if path else self.csv_file
        with open(out, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "hash", "file", "path", "context", "type",
                "has_face", "max_chars", "original", "translated",
            ])
            for e in self.project.entries:
                writer.writerow([
                    e.hash, e.file, e.path, e.context, e.entry_type,
                    e.has_face, e.max_chars,
                    e.original.replace("\n", "\\n"),
                    e.translated.replace("\n", "\\n") if e.translated else "",
                ])
        logger.info("CSV exported → %s", out)
        return out

    def import_csv(self, path: Optional[str] = None) -> None:
        """Import translations from an edited CSV back into the project."""
        if not self.project:
            raise RuntimeError("No project loaded; run extract() first.")

        src = Path(path) if path else self.csv_file
        lookup = {e.hash: e for e in self.project.entries}
        imported = 0

        with open(src, "r", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                h = row.get("hash")
                t = row.get("translated", "")
                if h in lookup and t:
                    lookup[h].translated = t.replace("\\n", "\n")
                    imported += 1

        self._save()
        logger.info("Imported %d translations from CSV", imported)

    # -- stats ---------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return translation progress statistics."""
        if not self.project:
            return {}
        total = len(self.project.entries)
        done = sum(1 for e in self.project.entries if e.translated)
        by_type: Dict[str, Dict[str, int]] = {}
        for e in self.project.entries:
            bucket = by_type.setdefault(e.entry_type, {"total": 0, "translated": 0})
            bucket["total"] += 1
            if e.translated:
                bucket["translated"] += 1
        return {
            "total_entries": total,
            "translated_entries": done,
            "untranslated_entries": total - done,
            "progress_percent": round(done / total * 100, 1) if total else 0,
            "by_type": by_type,
            "glossary_terms": self.glossary.size,
        }

    # -- glossary management -------------------------------------------------

    def export_glossary_csv(self, path: Optional[str] = None) -> str:
        """Export the glossary to a CSV file."""
        out = path or str(self.project_dir / "glossary.csv")
        self.glossary.export_csv(out)
        return out

    def import_glossary_csv(self, path: str) -> int:
        """Import glossary terms from a CSV file."""
        count = self.glossary.import_csv(path)
        if self.glossary.dirty:
            self.glossary.save()
        return count

    def add_glossary_term(
        self, source: str, translation: str, context: str = ""
    ) -> bool:
        """Manually add a term to the glossary."""
        ok = self.glossary.add(source, translation, context, manual=True, overwrite=True)
        if ok and self.glossary.dirty:
            self.glossary.save()
        return ok

    def build_glossary_from_project(self) -> int:
        """Build/update glossary from already-translated entries."""
        if not self.project:
            return 0
        count = self.glossary.learn_from_project(self.project)
        if self.glossary.dirty:
            self.glossary.save()
        return count


# ---------------------------------------------------------------------------
# Backwards-compat alias (used by main.py)
# ---------------------------------------------------------------------------
RPGMakerTranslatorV2 = RPGMakerTranslator
