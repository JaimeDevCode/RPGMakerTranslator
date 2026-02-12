#!/usr/bin/env python3
"""
RPG Maker MV/MZ File Encryption / Decryption
==============================================
Handles ``.rpgmvp`` / ``.rpgmvm`` / ``.rpgmvo`` files used by RPG Maker MV
and the ``.png_`` / ``.ogg_`` / ``.m4a_`` variants used by MZ.

Encryption format
-----------------
- 16-byte RPGMV header (signature + version + padding)
- First 16 bytes of the payload are XOR'd with the encryption key
- Remainder of the file is **not** encrypted

Author: JaimeDevCode
License: MIT
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RPGMV_HEADER_LENGTH = 16
RPGMV_SIGNATURE = bytes.fromhex("5250474d56000000")  # "RPGMV\0\0\0"
RPGMV_VERSION = bytes.fromhex("000301")

PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
                       0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52])
OGG_SIGNATURE = bytes([0x4F, 0x67, 0x67, 0x53])

ENCRYPTED_EXTENSIONS = {
    ".rpgmvp": ".png",
    ".rpgmvm": ".m4a",
    ".rpgmvo": ".ogg",
    ".png_": ".png",
    ".ogg_": ".ogg",
    ".m4a_": ".m4a",
}

DECRYPTED_EXTENSIONS = {
    v: k for k, v in ENCRYPTED_EXTENSIONS.items() if not k.endswith("_")
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GameInfo:
    """Metadata extracted from an RPG Maker game folder."""

    game_path: Path
    game_title: str = ""
    is_mz: bool = False
    has_encrypted_images: bool = False
    has_encrypted_audio: bool = False
    encryption_key: Optional[str] = None
    data_path: Optional[Path] = None
    img_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------


class RPGMVKeyManager:
    """Extract and manage RPG Maker encryption keys."""

    @staticmethod
    def get_game_info(game_path: str) -> GameInfo:
        """Inspect a game folder and return :class:`GameInfo`."""
        root = Path(game_path)
        info = GameInfo(game_path=root)

        # MV vs MZ detection.
        info.is_mz = (
            (root / "game.rmmzproject").exists()
            or ((root / "data").exists() and not (root / "www").exists())
        )

        if info.is_mz:
            info.data_path = root / "data"
            info.img_path = root / "img"
        else:
            info.data_path = root / "www" / "data"
            info.img_path = root / "www" / "img"
            if not info.data_path.exists():
                info.data_path = root / "data"
                info.img_path = root / "img"

        # Read System.json.
        system = info.data_path / "System.json" if info.data_path else None
        if system and system.exists():
            try:
                with open(system, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                info.game_title = data.get("gameTitle", "")
                info.has_encrypted_images = data.get("hasEncryptedImages", False)
                info.has_encrypted_audio = data.get("hasEncryptedAudio", False)
                info.encryption_key = data.get("encryptionKey")
            except Exception as exc:
                logger.warning("Failed to read System.json: %s", exc)

        return info

    @staticmethod
    def detect_key_from_file(encrypted_file: str) -> Optional[str]:
        """Try to recover the encryption key from an encrypted PNG."""
        with open(encrypted_file, "rb") as fh:
            data = fh.read(32)

        if len(data) < 32 or data[:5] != b"RPGMV":
            return None

        encrypted_header = data[RPGMV_HEADER_LENGTH : RPGMV_HEADER_LENGTH + len(PNG_SIGNATURE)]
        key_bytes = bytes(a ^ b for a, b in zip(encrypted_header, PNG_SIGNATURE))
        return key_bytes.hex()


# ---------------------------------------------------------------------------
# Decryptor
# ---------------------------------------------------------------------------


class RPGMVDecryptor:
    """Decrypt RPG Maker MV/MZ encrypted files."""

    def __init__(self, encryption_key: Optional[str] = None) -> None:
        self.key = bytes.fromhex(encryption_key) if encryption_key else None

    def decrypt_data(self, data: bytes) -> bytes:
        if len(data) < RPGMV_HEADER_LENGTH:
            return data
        if data[:5] != b"RPGMV":
            return data  # Not encrypted.

        content = bytearray(data[RPGMV_HEADER_LENGTH:])
        if self.key:
            for i in range(min(16, len(content))):
                content[i] ^= self.key[i % len(self.key)]
        else:
            content[: len(PNG_SIGNATURE)] = PNG_SIGNATURE
        return bytes(content)

    def decrypt_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        inp = Path(input_path)
        if output_path is None:
            ext = ENCRYPTED_EXTENSIONS.get(inp.suffix.lower(), ".png")
            output_path = str(inp.with_suffix(ext))

        with open(inp, "rb") as fh:
            raw = fh.read()

        decrypted = self.decrypt_data(raw)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            fh.write(decrypted)
        return str(out)

    def decrypt_folder(
        self, input_folder: str, output_folder: Optional[str] = None, recursive: bool = True
    ) -> List[str]:
        inp = Path(input_folder)
        out = Path(output_folder) if output_folder else inp.parent / f"{inp.name}_decrypted"
        out.mkdir(parents=True, exist_ok=True)

        pattern = "**/*" if recursive else "*"
        decrypted: List[str] = []
        for ext, plain_ext in ENCRYPTED_EXTENSIONS.items():
            for fpath in inp.glob(f"{pattern}{ext}"):
                rel = fpath.relative_to(inp)
                dest = out / rel.with_suffix(plain_ext)
                try:
                    self.decrypt_file(str(fpath), str(dest))
                    decrypted.append(str(dest))
                except Exception as exc:
                    logger.error("Decrypt failed for %s: %s", fpath, exc)
        return decrypted


# ---------------------------------------------------------------------------
# Encryptor
# ---------------------------------------------------------------------------


class RPGMVEncryptor:
    """Encrypt files back to RPG Maker format."""

    def __init__(self, encryption_key: str) -> None:
        if not encryption_key:
            raise ValueError("Encryption key is required")
        self.key = bytes.fromhex(encryption_key)

    def encrypt_data(self, data: bytes) -> bytes:
        header = bytearray(RPGMV_HEADER_LENGTH)
        header[:8] = RPGMV_SIGNATURE
        header[8:11] = RPGMV_VERSION
        content = bytearray(data)
        for i in range(min(16, len(content))):
            content[i] ^= self.key[i % len(self.key)]
        return bytes(header) + bytes(content)

    def encrypt_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        inp = Path(input_path)
        if output_path is None:
            ext = DECRYPTED_EXTENSIONS.get(inp.suffix.lower(), ".rpgmvp")
            output_path = str(inp.with_suffix(ext))

        with open(inp, "rb") as fh:
            raw = fh.read()

        encrypted = self.encrypt_data(raw)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            fh.write(encrypted)
        return str(out)

    def encrypt_folder(
        self, input_folder: str, output_folder: Optional[str] = None
    ) -> List[str]:
        inp = Path(input_folder)
        out = Path(output_folder) if output_folder else inp
        encrypted: List[str] = []
        for ext, enc_ext in DECRYPTED_EXTENSIONS.items():
            for fpath in inp.rglob(f"*{ext}"):
                rel = fpath.relative_to(inp)
                dest = out / rel.with_suffix(enc_ext)
                try:
                    self.encrypt_file(str(fpath), str(dest))
                    encrypted.append(str(dest))
                except Exception as exc:
                    logger.error("Encrypt failed for %s: %s", fpath, exc)
        return encrypted


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def decrypt_game_images(game_path: str, output_folder: Optional[str] = None) -> List[str]:
    """Decrypt every image in a game's ``img/`` folder."""
    info = RPGMVKeyManager.get_game_info(game_path)
    dec = RPGMVDecryptor(info.encryption_key)
    if info.img_path and info.img_path.exists():
        return dec.decrypt_folder(str(info.img_path), output_folder)
    return []


def encrypt_game_images(
    game_path: str, input_folder: str, output_folder: Optional[str] = None
) -> List[str]:
    """Encrypt images back to RPG Maker format."""
    info = RPGMVKeyManager.get_game_info(game_path)
    if not info.encryption_key:
        raise ValueError("No encryption key found in game")
    enc = RPGMVEncryptor(info.encryption_key)
    return enc.encrypt_folder(input_folder, output_folder or str(info.img_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="RPG Maker MV/MZ File Crypto")
    parser.add_argument("command", choices=["decrypt-game", "encrypt-game", "info"])
    parser.add_argument("game_path", help="Path to game folder")
    parser.add_argument("-o", "--output", help="Output folder")
    parser.add_argument("-i", "--input", help="Input folder (encryption)")
    args = parser.parse_args()

    if args.command == "info":
        gi = RPGMVKeyManager.get_game_info(args.game_path)
        print(f"Game:       {gi.game_title}")
        print(f"Type:       {'MZ' if gi.is_mz else 'MV'}")
        print(f"Encrypted:  images={gi.has_encrypted_images}  audio={gi.has_encrypted_audio}")
        print(f"Key:        {gi.encryption_key or 'Not found'}")
    elif args.command == "decrypt-game":
        result = decrypt_game_images(args.game_path, args.output)
        print(f"Decrypted {len(result)} files")
    elif args.command == "encrypt-game":
        if not args.input:
            print("Error: --input required for encryption")
        else:
            result = encrypt_game_images(args.game_path, args.input, args.output)
            print(f"Encrypted {len(result)} files")
