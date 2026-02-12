#!/usr/bin/env python3
"""
Apply Translated Images to RPG Maker Game
===========================================
Standalone utility that copies only *actually translated* images back into the
game folder.  Provides backup and restore functionality.

Usage::

    python apply_translated_images.py /path/to/game
    python apply_translated_images.py /path/to/game --restore

Author: JaimeDevCode
License: MIT
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_img_path(game_path: Path) -> Path:
    """Return the game's ``img/`` folder (MV or MZ)."""
    for candidate in (game_path / "www" / "img", game_path / "img"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find img/ folder in game")


def _get_translated_files(project_path: Path) -> List[Path]:
    """Return encrypted files that were actually modified (contain translated text)."""
    encrypted = project_path / "img_encrypted"
    translated = project_path / "img_translated"
    decrypted = project_path / "img_decrypted"

    result: List[Path] = []

    if not encrypted.exists():
        return result

    for rpgmvp in encrypted.rglob("*.rpgmvp"):
        rel = rpgmvp.relative_to(encrypted)
        png_rel = rel.with_suffix(".png")

        tr_png = translated / png_rel
        dec_png = decrypted / png_rel

        if tr_png.exists():
            # Quick size comparison — translated will differ from original.
            if dec_png.exists() and tr_png.stat().st_size != dec_png.stat().st_size:
                result.append(rpgmvp)
            elif not dec_png.exists():
                result.append(rpgmvp)

    return result


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def apply_translated_images(
    game_path: str,
    create_backup: bool = True,
    check_modified: bool = True,
) -> bool:
    """Copy translated images into the game folder."""
    root = Path(game_path)
    project = root / "translation_project"
    encrypted = project / "img_encrypted"

    if not encrypted.exists():
        logger.error("No encrypted images folder: %s", encrypted)
        return False

    try:
        img_path = _find_img_path(root)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return False

    if check_modified:
        files = _get_translated_files(project)
        logger.info("Found %d translated images", len(files))
    else:
        files = list(encrypted.rglob("*.rpgmvp"))
        logger.info("Found %d encrypted images (force-all)", len(files))

    if not files:
        logger.warning("No translated images to apply")
        return False

    backup_dir: Optional[Path] = None
    if create_backup:
        backup_dir = project / "backup" / "img_before_apply"
        backup_dir.mkdir(parents=True, exist_ok=True)

    replaced = skipped = 0

    for src in files:
        rel = src.relative_to(encrypted)
        dest = img_path / rel

        if not dest.exists():
            logger.warning("Original not found, skipping: %s", rel)
            skipped += 1
            continue

        try:
            if backup_dir:
                bak = backup_dir / rel
                bak.parent.mkdir(parents=True, exist_ok=True)
                if not bak.exists():
                    shutil.copy2(dest, bak)
            shutil.copy2(src, dest)
            logger.info("Replaced: %s", rel)
            replaced += 1
        except Exception as exc:
            logger.error("Failed: %s — %s", rel, exc)
            skipped += 1

    logger.info("")
    logger.info("Applied %d translated images", replaced)
    if skipped:
        logger.info("Skipped: %d", skipped)
    if backup_dir:
        logger.info("Backup:  %s", backup_dir)

    return True


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


def restore_from_backup(game_path: str) -> bool:
    """Restore original images from backup."""
    root = Path(game_path)
    project = root / "translation_project"

    for candidate in (
        project / "backup" / "img_before_apply",
        project / "backup" / "img",
    ):
        if candidate.exists() and list(candidate.rglob("*.rpgmvp")):
            backup = candidate
            break
    else:
        logger.error("No backup found")
        return False

    try:
        img_path = _find_img_path(root)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return False

    restored = 0
    for bak_file in backup.rglob("*.rpgmvp"):
        rel = bak_file.relative_to(backup)
        dest = img_path / rel
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bak_file, dest)
            restored += 1
        except Exception as exc:
            logger.error("Failed to restore %s: %s", rel, exc)

    logger.info("Restored %d original files", restored)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply or restore translated images for an RPG Maker game"
    )
    parser.add_argument("game_path", help="Path to game folder")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup")
    parser.add_argument("--force-all", action="store_true",
                        help="Apply ALL encrypted files, not just translated ones")
    parser.add_argument("--restore", action="store_true",
                        help="Restore originals from backup")
    args = parser.parse_args()

    if not Path(args.game_path).exists():
        logger.error("Game path not found: %s", args.game_path)
        return 1

    if args.restore:
        return 0 if restore_from_backup(args.game_path) else 1

    ok = apply_translated_images(
        args.game_path,
        create_backup=not args.no_backup,
        check_modified=not args.force_all,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
