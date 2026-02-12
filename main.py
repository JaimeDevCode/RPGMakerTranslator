#!/usr/bin/env python3
"""
RPG Maker MV/MZ Game Translator
=================================
One-command automatic translation of text **and** images in RPG Maker games.

Usage
-----
::

    # Translate everything (text + images), auto-detect GPU
    python main.py /path/to/game

    # Specify languages and backend
    python main.py /path/to/game -s ja -t en --backend google

    # Text only (skip images)
    python main.py /path/to/game --no-images

    # Export CSV for manual editing
    python main.py /path/to/game --export-only

Author: JaimeDevCode
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TranslationConfig:
    """All settings for a translation run."""

    game_path: str
    source_lang: str = "ja"
    target_lang: str = "en"
    backend: str = "google"
    api_key: Optional[str] = None
    include_images: bool = True
    export_only: bool = False
    skip_translated: bool = True
    batch_size: int = 50
    delay: float = 0.5
    create_backup: bool = True
    output_dir: Optional[str] = None
    verbose: bool = False
    patch_plugins: bool = True
    enable_word_wrap: bool = True
    replace_images: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# Game Detection
# ═══════════════════════════════════════════════════════════════════════════


class GameInfo:
    """Auto-detect RPG Maker game type, paths, plugins, and encryption."""

    def __init__(self, game_path: str) -> None:
        self.game_path = Path(game_path)
        if not self.game_path.exists():
            raise FileNotFoundError(f"Game folder not found: {game_path}")

        self.game_type = self._detect_type()
        self.data_path = self._find_data()
        self.js_path = self._find_js()
        self.img_path = self._find_img()
        self.game_title = self._read_title()
        self.detected_plugins = self._detect_plugins()
        self.has_encrypted_images = self._check_encrypted()

    # -- detection helpers ---------------------------------------------------

    def _detect_type(self) -> str:
        if (self.game_path / "game.rmmzproject").exists():
            return "MZ"
        if (self.game_path / "game.rmmvproject").exists():
            return "MV"
        if (self.game_path / "Game.rpgproject").exists():
            return "MV"
        if (self.game_path / "www" / "data").exists():
            return "MV"
        if (self.game_path / "data").exists():
            return "MZ"
        return "MV"

    def _find_data(self) -> Path:
        if self.game_type == "MV":
            candidate = self.game_path / "www" / "data"
            if candidate.exists():
                return candidate
        candidate = self.game_path / "data"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Data folder not found in {self.game_path}")

    def _find_js(self) -> Path:
        if self.game_type == "MV":
            candidate = self.game_path / "www" / "js" / "plugins"
            if candidate.exists():
                return candidate
        return self.game_path / "js" / "plugins"

    def _find_img(self) -> Path:
        if self.game_type == "MV":
            candidate = self.game_path / "www" / "img"
            if candidate.exists():
                return candidate
        return self.game_path / "img"

    def _read_title(self) -> str:
        system = self.data_path / "System.json"
        if system.exists():
            try:
                with open(system, "r", encoding="utf-8") as fh:
                    return json.load(fh).get("gameTitle", "Unknown Game")
            except Exception:
                pass
        return "Unknown Game"

    def _detect_plugins(self) -> List[str]:
        if not self.js_path.exists():
            return []
        known = [
            "LL_GalgeChoiceWindowMV.js",
            "YEP_MessageCore.js",
            "SRD_TranslationEngine.js",
            "GALV_MessageBusts.js",
        ]
        return [
            p.replace(".js", "") for p in known
            if (self.js_path / p).exists()
        ]

    def _check_encrypted(self) -> bool:
        if self.img_path.exists():
            return bool(list(self.img_path.rglob("*.rpgmvp"))[:1])
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Translation Pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TranslationPipeline:
    """Orchestrates the 5-step translation pipeline."""

    def __init__(self, config: TranslationConfig) -> None:
        self.config = config
        self.game = GameInfo(config.game_path)
        self.output = (
            Path(config.output_dir) if config.output_dir
            else self.game.game_path / "translation_project"
        )
        self.output.mkdir(parents=True, exist_ok=True)

        self._text = None
        self._image = None
        self._init_modules()

    def _init_modules(self) -> None:
        # -- text translator ---------------------------------------------------
        try:
            from rpgmaker_translator import RPGMakerTranslator

            self._text = RPGMakerTranslator(
                game_path=self.config.game_path,
                source_lang=self.config.source_lang,
                target_lang=self.config.target_lang,
                backend=self.config.backend,
                api_key=self.config.api_key,
            )
            logger.info("Text translator ready  (%s)", self._text.backend.name)
        except ImportError as exc:
            logger.error("Failed to import text translator: %s", exc)

        # -- image translator --------------------------------------------------
        if self.config.include_images:
            try:
                from rpgmaker_image_translator import RPGMakerImageTranslator

                self._image = RPGMakerImageTranslator(
                    game_path=self.config.game_path,
                    source_lang=self.config.source_lang,
                    target_lang=self.config.target_lang,
                    backend=self.config.backend,
                    api_key=self.config.api_key,
                    # GPU auto-detected inside EasyOCREngine
                )
                logger.info("Image translator ready")
                if self.game.has_encrypted_images:
                    logger.info("  Encrypted .rpgmvp files detected")
            except ImportError as exc:
                logger.warning(
                    "Image translator unavailable: %s\n"
                    "  Install: pip install easyocr opencv-python Pillow",
                    exc,
                )

    # -- pipeline execution --------------------------------------------------

    def run(self) -> None:
        """Execute the full translation pipeline."""
        self._banner()

        if not self._text:
            logger.error("Text translator not available — cannot continue")
            return

        # Step 1: Extract
        logger.info("")
        logger.info("STEP 1 / 5 — Extracting translatable text …")
        project = self._text.extract()
        logger.info("  Found %d text entries", len(project.entries))
        self._text.export_csv()

        if self.config.export_only:
            logger.info("")
            logger.info("Export complete!  Edit the CSV at:")
            logger.info("  %s", self._text.csv_file)
            return

        # Step 2: Patch plugins
        if self.config.patch_plugins:
            logger.info("")
            logger.info("STEP 2 / 5 — Patching plugins …")
            for name, ok in self._text.patch_plugins().items():
                logger.info("  %s %s", "OK" if ok else "--", name)

        # Step 3: Translate text
        logger.info("")
        logger.info("STEP 3 / 5 — Translating text …")
        self._text.translate(
            batch_size=self.config.batch_size,
            delay=self.config.delay,
            skip_translated=self.config.skip_translated,
        )
        stats = self._text.get_stats()
        logger.info(
            "  Progress: %s%%  (%d / %d)",
            stats.get("progress_percent", 0),
            stats.get("translated_entries", 0),
            stats.get("total_entries", 0),
        )

        # Step 4: Apply text translations
        logger.info("")
        logger.info("STEP 4 / 5 — Applying text translations …")
        self._text.apply(
            backup=self.config.create_backup,
            wordwrap=self.config.enable_word_wrap,
        )

        # Step 5: Images
        if self.config.include_images and self._image:
            logger.info("")
            logger.info("STEP 5 / 5 — Translating images …")
            self._translate_images()

        self._summary()

    # -- image translation ---------------------------------------------------

    def _translate_images(self) -> None:
        if not self._image:
            return

        try:
            if self.game.has_encrypted_images:
                logger.info("  Processing encrypted images …")
                res = self._image.translate_all_rpgmvp_files(
                    replace_originals=self.config.replace_images,
                    backup=self.config.create_backup,
                    folders=["pictures", "titles1", "titles2"],
                )
                logger.info(
                    "  Scanned %d | Translated %d | Skipped %d",
                    res.get("total_files", 0),
                    res.get("files_translated", 0),
                    res.get("files_skipped", 0),
                )
            else:
                logger.info("  Processing PNG images …")
                img_base = self.game.img_path
                if not img_base or not img_base.exists():
                    return
                processed = translated = 0
                for folder_name in ("pictures", "titles1", "titles2"):
                    folder = img_base / folder_name
                    if not folder.exists():
                        continue
                    for img_file in folder.rglob("*.png"):
                        out_path = self.output / "img_translated" / img_file.relative_to(img_base)
                        result = self._image.process_image(str(img_file), str(out_path), force=True)
                        processed += 1
                        if result.modified:
                            translated += 1
                            if self.config.replace_images:
                                shutil.copy2(str(out_path), str(img_file))
                logger.info("  Processed %d images, %d had text", processed, translated)
        except Exception as exc:
            logger.error("  Image translation error: %s", exc)
            if self.config.verbose:
                import traceback
                traceback.print_exc()

    # -- display helpers -----------------------------------------------------

    def _banner(self) -> None:
        logger.info("=" * 60)
        logger.info("  RPG Maker Game Translator")
        logger.info("=" * 60)
        logger.info("  Game:     %s", self.game.game_title)
        logger.info("  Engine:   RPG Maker %s", self.game.game_type)
        logger.info("  Langs:    %s → %s", self.config.source_lang, self.config.target_lang)
        logger.info("  Backend:  %s", self.config.backend)
        logger.info("  Images:   %s", "yes" if self.config.include_images else "no")
        if self.game.has_encrypted_images:
            logger.info("  Encrypted: yes (.rpgmvp)")
        if self.game.detected_plugins:
            logger.info("  Plugins: %s", ", ".join(self.game.detected_plugins))
        logger.info("=" * 60)

    def _summary(self) -> None:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  Translation complete!")
        logger.info("=" * 60)
        if self._text:
            s = self._text.get_stats()
            logger.info("  Text:  %d / %d  (%s%%)",
                        s.get("translated_entries", 0),
                        s.get("total_entries", 0),
                        s.get("progress_percent", 0))
        logger.info("  Output: %s", self.output)
        logger.info("")
        logger.info("  Test your game thoroughly after translation!")
        logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rpgmaker-translator",
        description="Automatic RPG Maker MV/MZ game translator (text + images)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py /path/to/game                    # translate everything
  python main.py /path/to/game -s ja -t es        # Japanese → Spanish
  python main.py /path/to/game --no-images        # text only
  python main.py /path/to/game --export-only      # CSV for manual editing
  python main.py /path/to/game --import-csv       # apply edited CSV
  python main.py /path/to/game --backend deepl --api-key KEY
  python main.py /path/to/game --backend marian   # offline, unlimited

supported languages:
  ja (Japanese), en (English), zh (Chinese), ko (Korean),
  es (Spanish), fr (French), de (German), pt (Portuguese),
  ru (Russian), it (Italian), and many more.
""",
    )

    parser.add_argument("game_path", help="Path to RPG Maker game folder")
    parser.add_argument("-s", "--source", default="ja",
                        help="Source language (default: ja)")
    parser.add_argument("-t", "--target", default="en",
                        help="Target language (default: en)")
    parser.add_argument("-b", "--backend", choices=["google", "deepl", "marian"],
                        default="google",
                        help="Translation backend: google (free), deepl (API key), "
                             "marian (offline, unlimited)")
    parser.add_argument("--api-key",
                        help="API key for DeepL (or set DEEPL_API_KEY env)")
    parser.add_argument("--export-only", action="store_true",
                        help="Extract text to CSV without translating")
    parser.add_argument("--import-csv", action="store_true",
                        help="Import translations from edited CSV")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip image translation")
    parser.add_argument("--no-backup", action="store_true",
                        help="Don't create backups")
    parser.add_argument("--no-patch", action="store_true",
                        help="Don't patch JS plugins")
    parser.add_argument("--no-wordwrap", action="store_true",
                        help="Don't inject word-wrap plugin")
    parser.add_argument("--no-replace-images", action="store_true",
                        help="Keep translated images in output folder only")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Strings per translation batch (default: 50)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between batches (default: 0.5)")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Debug-level logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = TranslationConfig(
        game_path=args.game_path,
        source_lang=args.source,
        target_lang=args.target,
        backend=args.backend,
        api_key=args.api_key,
        export_only=args.export_only,
        include_images=not args.no_images,
        batch_size=args.batch_size,
        delay=args.delay,
        create_backup=not args.no_backup,
        output_dir=args.output,
        verbose=args.verbose,
        patch_plugins=not args.no_patch,
        enable_word_wrap=not args.no_wordwrap,
        replace_images=not args.no_replace_images,
    )

    # -- CSV import mode -----------------------------------------------------
    if args.import_csv:
        try:
            from rpgmaker_translator import RPGMakerTranslator

            translator = RPGMakerTranslator(
                game_path=args.game_path,
                source_lang=args.source,
                target_lang=args.target,
                backend=args.backend,
                api_key=args.api_key,
            )
            translator.import_csv()
            translator.apply(
                backup=not args.no_backup,
                wordwrap=not args.no_wordwrap,
            )
            logger.info("Translations imported and applied")
            return 0
        except Exception as exc:
            logger.error("CSV import failed: %s", exc)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    # -- normal pipeline mode ------------------------------------------------
    try:
        pipeline = TranslationPipeline(config)
        pipeline.run()
        return 0
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("\nInterrupted — progress has been saved. Run again to resume.")
        return 0
    except Exception as exc:
        logger.error("Translation failed: %s", exc)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
