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

    # -- event command list --------------------------------------------------

    def _extract_events(
        self, event_list: list, file: str, base_path: str, context: str
    ) -> None:
        """Walk an event command list and collect translatable entries."""
        if not event_list:
            return
        has_face = False

        for i, cmd in enumerate(event_list):
            if not isinstance(cmd, dict):
                continue
            code: int = cmd.get("code", 0)
            params: list = cmd.get("parameters", [])

            # Skip script / logic commands.
            if code in (355, 655, 122, 121):
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

                for pi, page in enumerate(event.get("pages") or []):
                    if isinstance(page, dict) and "list" in page:
                        self._extract_events(
                            page["list"], fname,
                            f"events[{ei}].pages[{pi}].list",
                            f"{fname} - {ev_name}",
                        )


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

    def patch_all(self) -> Dict[str, bool]:
        """Apply all known patches, returning ``{name: success}``."""
        return {
            "LL_GalgeChoiceWindowMV": self._patch_ll_galge(),
            "WordWrapPlugin": self._create_wordwrap_plugin(),
        }

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

    def _create_wordwrap_plugin(self) -> bool:
        """Create a word-wrap JS plugin for Western-language translations."""
        plugin_code = (
            "//=============================================================================\n"
            "// TranslationWordWrap.js — Automatic word wrap for translated text\n"
            "//=============================================================================\n"
            "/*:\n"
            " * @plugindesc Automatic word wrap for translated Western-language text.\n"
            " * @author JaimeDevCode — RPG Maker Translator\n"
            " * @param Enable\n"
            " * @type boolean\n"
            " * @default true\n"
            " */\n"
            "(function () {\n"
            "  var params = PluginManager.parameters('TranslationWordWrap');\n"
            "  if (params['Enable'] === 'false') return;\n"
            "\n"
            "  var _orig = Window_Message.prototype.processNormalCharacter;\n"
            "  Window_Message.prototype.processNormalCharacter = function (textState) {\n"
            "    var c = textState.text[textState.index++];\n"
            "    var w = this.textWidth(c);\n"
            "    if (textState.x + w > this.contents.width - this.newLineX()) {\n"
            "      this.processNewLine(textState);\n"
            "      textState.height = this.calcTextHeight(textState, false);\n"
            "    }\n"
            "    this.contents.drawText(c, textState.x, textState.y, w * 2, textState.height);\n"
            "    textState.x += w;\n"
            "  };\n"
            "})();\n"
        )
        try:
            self.js_path.mkdir(parents=True, exist_ok=True)
            (self.js_path / "TranslationWordWrap.js").write_text(plugin_code, encoding="utf-8")
            return True
        except Exception:
            return False


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

    # -- apply ---------------------------------------------------------------

    def apply_all(self, backup: bool = True, wordwrap: bool = True) -> None:
        """Apply all translations to the game's data files."""
        if backup:
            self.backup_path.mkdir(parents=True, exist_ok=True)

        for filename, entries in self._by_file.items():
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
                if wordwrap and entry.entry_type in ("text", "escape_text"):
                    translated = self.wrapper.wrap(translated, entry.has_face)

                if "__plugin_choices__" in path:
                    modified |= self._apply_plugin_choices(data, path, translated)
                elif "__plugin_text__" in path or "__plugin_generic__" in path:
                    modified |= self._apply_plugin_text(data, path, translated)
                else:
                    modified |= self._set_value(data, path, translated)

            if modified:
                self._save_json(filepath, data)
                logger.info("Updated %s", filename)

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

    def extract(self) -> TranslationProject:
        """Step 1 – Extract all translatable text from the game."""
        old_project = self.project  # may contain translations from a previous run
        self.project = TextExtractor(str(self.game_path), self.source_lang).extract_all()
        self.project.target_lang = self.target_lang

        # Merge previously saved translations so we don't lose progress.
        if old_project and old_project.entries:
            old_map: Dict[str, TranslationEntry] = {
                e.hash: e for e in old_project.entries if e.translated
            }
            if old_map:
                merged = 0
                for entry in self.project.entries:
                    prev = old_map.get(entry.hash)
                    if prev and not entry.translated:
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
        """Step 2 – Translate all entries using the configured backend."""
        if not self.project:
            self.extract()

        assert self.project is not None
        todo = [
            e for e in self.project.entries
            if not (skip_translated and e.translated)
        ]
        if not todo:
            logger.info("All entries already translated — nothing to do")
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

    def apply(self, backup: bool = True, wordwrap: bool = True) -> None:
        """Step 3 – Write translations back to game files."""
        if not self.project:
            return
        TranslationApplier(str(self.game_path), self.project).apply_all(backup, wordwrap)

    def patch_plugins(self) -> Dict[str, bool]:
        """Patch JS plugins for translation compatibility."""
        return self.patcher.patch_all()

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
        }


# ---------------------------------------------------------------------------
# Backwards-compat alias (used by main.py)
# ---------------------------------------------------------------------------
RPGMakerTranslatorV2 = RPGMakerTranslator
