#!/usr/bin/env python3
"""
RPG Maker MV/MZ Image Translator
==================================
OCR-based image translation pipeline for RPG Maker game graphics.

Features:
  - Strict CJK text validation (only modifies images with real text)
  - Automatic GPU detection for EasyOCR
  - Transparency-aware inpainting & rendering
  - Encrypted .rpgmvp / .png_ file support
  - Free translation via Google Translate (default)

Author: JaimeDevCode
License: MIT
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CJK_PATTERN = re.compile(
    r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf\uac00-\ud7af]"
)

NOISE_PATTERNS = [
    re.compile(r"^[\d\s\.\-\+\*\/\\\|\[\]\(\)\{\}<>@#$%^&=_~`]+$"),
    re.compile(r"^[a-zA-Z]{1,2}$"),
    re.compile(r"^\W+$"),
    re.compile(r"^\s+$"),
]

# Filenames / folders that should never be processed (UI graphics, etc.).
SKIP_FILE_PATTERNS = [
    r"Window", r"IconSet", r"Balloon", r"Shadow[0-9]*", r"Damage",
    r"States", r"Weapons[0-9]*", r"Armors[0-9]*", r"Loading", r"GameOver",
    r"Button", r"Gauge", r"^ui_", r"^system", r"cursor", r"Cursor",
    r"Actor[0-9]+", r"Enemy[0-9]+", r"Character", r"Face[0-9]*",
    r"Tileset", r"Animation", r"Battleback", r"Parallax",
    r"^bg_", r"^ev_", r"^_", r"mask", r"Mask",
]

SKIP_FOLDERS = {
    "characters", "enemies", "faces", "sv_actors", "sv_enemies",
    "tilesets", "animations", "battlebacks1", "battlebacks2",
    "parallaxes", "system",
}

TEXT_FOLDERS = ["pictures", "titles1", "titles2"]

DEFAULT_FONTS: Dict[str, List[str]] = {
    "ja": ["NotoSansJP-Regular.ttf", "Yu Gothic", "MS Gothic", "arial.ttf"],
    "en": ["arial.ttf", "DejaVuSans.ttf", "NotoSans-Regular.ttf"],
    "zh": ["NotoSansSC-Regular.ttf", "SimHei", "Microsoft YaHei"],
    "ko": ["NotoSansKR-Regular.ttf", "Malgun Gothic"],
    "es": ["arial.ttf", "DejaVuSans.ttf", "NotoSans-Regular.ttf"],
    "fr": ["arial.ttf", "DejaVuSans.ttf", "NotoSans-Regular.ttf"],
    "de": ["arial.ttf", "DejaVuSans.ttf", "NotoSans-Regular.ttf"],
    "pt": ["arial.ttf", "DejaVuSans.ttf", "NotoSans-Regular.ttf"],
    "ru": ["arial.ttf", "DejaVuSans.ttf", "NotoSans-Regular.ttf"],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TextRegion:
    """A detected text region inside an image."""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    text: str = ""
    confidence: float = 0.0
    translated: str = ""
    font_size: int = 0

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    def has_cjk(self) -> bool:
        """True if the region text contains CJK characters."""
        return bool(self.text and CJK_PATTERN.search(self.text))

    def is_noise(self) -> bool:
        """True if the detected text looks like OCR noise."""
        text = self.text.strip()
        if not text:
            return True
        for pat in NOISE_PATTERNS:
            if pat.match(text):
                return True
        if len(text) < 2 and not self.has_cjk():
            return True
        return False

    def is_valid(self, min_confidence: float = 0.6) -> bool:
        """True if this region should be translated."""
        if not self.has_cjk():
            return False
        if self.is_noise():
            return False
        if self.area < 100:
            return False
        if self.confidence >= min_confidence:
            return True
        # Slightly lower confidence but big region → accept.
        if self.confidence >= (min_confidence - 0.2) and self.area >= 800:
            return True
        return False


@dataclass
class ImageTranslationResult:
    """Result of processing a single image."""

    original_path: str
    output_path: str = ""
    regions: List[TextRegion] = field(default_factory=list)
    success: bool = False
    modified: bool = False
    error: str = ""
    skip_reason: str = ""


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_gpu() -> bool:
    """Return True if a CUDA-capable GPU is available."""
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            name = torch.cuda.get_device_name(0)
            logger.info("GPU detected: %s", name)
        return available
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# OCR Backend
# ---------------------------------------------------------------------------

class EasyOCREngine:
    """EasyOCR wrapper with automatic GPU detection."""

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: Optional[bool] = None,
    ) -> None:
        try:
            import easyocr  # noqa: F401
            self._easyocr = easyocr
        except ImportError:
            raise ImportError(
                "EasyOCR is required for image translation. "
                "Install it with: pip install easyocr"
            )
        self.languages = languages or ["ja", "en"]
        self.use_gpu = use_gpu if use_gpu is not None else detect_gpu()
        self._reader = None

    def _ensure_ready(self) -> None:
        if self._reader is not None:
            return
        mode = "GPU" if self.use_gpu else "CPU"
        logger.info("Initialising EasyOCR (%s) — first run may download models …", mode)
        self._reader = self._easyocr.Reader(self.languages, gpu=self.use_gpu)

    def detect(
        self,
        image: np.ndarray,
        min_size: int = 10,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
    ) -> List[TextRegion]:
        """Run OCR and return detected text regions."""
        self._ensure_ready()
        try:
            results = self._reader.readtext(
                image,
                min_size=min_size,
                text_threshold=text_threshold,
                low_text=low_text,
                width_ths=0.7,
                paragraph=False,
            )
        except Exception as exc:
            logger.warning("OCR failed: %s", exc)
            return []

        regions: List[TextRegion] = []
        for bbox_points, text, confidence in results:
            pts = np.array(bbox_points)
            x1, y1 = pts.min(axis=0).astype(int)
            x2, y2 = pts.max(axis=0).astype(int)
            regions.append(TextRegion(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                text=text,
                confidence=float(confidence),
                font_size=max(12, int((y2 - y1) * 0.8)),
            ))
        return regions


# ---------------------------------------------------------------------------
# Inpainter
# ---------------------------------------------------------------------------

class Inpainter:
    """Remove text from images via OpenCV inpainting or simple fill."""

    @staticmethod
    def create_mask(
        shape: Tuple[int, ...],
        regions: List[TextRegion],
        padding: int = 2,
    ) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for r in regions:
            x1 = max(0, r.bbox[0] - padding)
            y1 = max(0, r.bbox[1] - padding)
            x2 = min(w, r.bbox[2] + padding)
            y2 = min(h, r.bbox[3] + padding)
            mask[y1:y2, x1:x2] = 255
        return mask

    @staticmethod
    def inpaint(
        image: np.ndarray, mask: np.ndarray, radius: int = 3
    ) -> np.ndarray:
        """OpenCV inpainting – handles RGBA by preserving alpha."""
        has_alpha = len(image.shape) == 3 and image.shape[2] == 4
        if has_alpha:
            bgr = image[:, :, :3]
            alpha = image[:, :, 3]
            result_bgr = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
            return cv2.merge([
                result_bgr[:, :, 0], result_bgr[:, :, 1],
                result_bgr[:, :, 2], alpha,
            ])
        return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)

    @staticmethod
    def simple_fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Simple background-color fill (better for transparent UI elements)."""
        result = image.copy()
        has_alpha = len(image.shape) == 3 and image.shape[2] == 4

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Sample background colour from edges.
            border: List[np.ndarray] = []
            for ix in range(max(0, x - 3), min(image.shape[1], x + w + 3)):
                if y > 3:
                    border.append(image[y - 3, ix])
                if y + h + 3 < image.shape[0]:
                    border.append(image[y + h + 2, ix])

            if border:
                colour = np.mean(border, axis=0).astype(np.uint8)
            else:
                colour = np.array(
                    [255, 255, 255, 255] if has_alpha else [255, 255, 255],
                    dtype=np.uint8,
                )
            result[mask == 255] = colour

        return result


# ---------------------------------------------------------------------------
# Text Renderer
# ---------------------------------------------------------------------------

class TextRenderer:
    """Render translated text onto images with outline and transparency support."""

    _FONT_DIRS = [
        "C:/Windows/Fonts",
        "/usr/share/fonts",
        os.path.expanduser("~/.fonts"),
        "./fonts",
    ]

    def __init__(self, extra_font_paths: Optional[List[str]] = None) -> None:
        self._extra = extra_font_paths or []
        self._cache: Dict[Tuple[str, int], Any] = {}

    def _find_font(self, target_lang: str = "en") -> Optional[str]:
        candidates = list(self._extra) + DEFAULT_FONTS.get(target_lang, DEFAULT_FONTS["en"])
        for name in candidates:
            if os.path.isfile(name):
                return name
            for d in self._FONT_DIRS:
                full = os.path.join(d, name)
                if os.path.isfile(full):
                    return full
        return None

    def _get_font(self, path: Optional[str], size: int):
        key = (path or "", size)
        if key not in self._cache:
            try:
                self._cache[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
            except Exception:
                self._cache[key] = ImageFont.load_default()
        return self._cache[key]

    @staticmethod
    def _text_width(text: str, font) -> int:
        if hasattr(font, "getlength"):
            return int(font.getlength(text))
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]

    @staticmethod
    def _line_height(font) -> int:
        bbox = font.getbbox("Ay")
        return bbox[3] - bbox[1]

    def _wrap(self, text: str, font, max_w: int) -> List[str]:
        lines: List[str] = []
        for part in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            words = part.split()
            if not words:
                lines.append("")
                continue
            if self._text_width(words[0], font) > max_w:
                cur = ""
                for ch in part:
                    test = cur + ch
                    if self._text_width(test, font) <= max_w or not cur:
                        cur = test
                    else:
                        lines.append(cur)
                        cur = ch
                if cur:
                    lines.append(cur)
                continue
            cur = words[0]
            for w in words[1:]:
                test = cur + " " + w
                if self._text_width(test, font) <= max_w:
                    cur = test
                else:
                    lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
        return lines

    def render(
        self,
        image: np.ndarray,
        regions: List[TextRegion],
        target_lang: str = "en",
    ) -> np.ndarray:
        """Draw translated text onto *image*, returning the modified array."""
        has_alpha = len(image.shape) == 3 and image.shape[2] == 4
        if has_alpha:
            pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
        else:
            pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(pil)
        font_path = self._find_font(target_lang)

        for region in regions:
            if not region.translated:
                continue

            x1, y1, x2, y2 = region.bbox
            box_w, box_h = x2 - x1, y2 - y1
            max_w = max(10, box_w - 4)

            font_size = min(box_h - 4, region.font_size or box_h - 4)
            font_size = max(10, font_size)
            lines: List[str] = [region.translated]

            # Shrink font until text fits vertically.
            while font_size >= 10:
                font = self._get_font(font_path, font_size)
                lh = self._line_height(font) + 2
                lines = self._wrap(region.translated, font, max_w)
                if len(lines) * lh <= max(10, box_h - 4):
                    break
                font_size -= 1

            font = self._get_font(font_path, font_size)
            lh = self._line_height(font) + 2
            total_h = len(lines) * lh
            text_y = y1 + (box_h - total_h) // 2

            # Choose colours based on local background brightness.
            sy, sx = max(0, y1 - 5), max(0, x1)
            if sy < image.shape[0] and sx < image.shape[1]:
                brightness = float(np.mean(image[sy, sx][:3]))
            else:
                brightness = 255.0

            if brightness > 128:
                fg = (0, 0, 0, 255) if has_alpha else (0, 0, 0)
                outline = (255, 255, 255, 255) if has_alpha else (255, 255, 255)
            else:
                fg = (255, 255, 255, 255) if has_alpha else (255, 255, 255)
                outline = (0, 0, 0, 255) if has_alpha else (0, 0, 0)

            for line in lines:
                tw = self._text_width(line, font)
                tx = x1 + (box_w - tw) // 2
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    draw.text((tx + dx, text_y + dy), line, font=font, fill=outline)
                draw.text((tx, text_y), line, font=font, fill=fg)
                text_y += lh

        if has_alpha:
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGBA2BGRA)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Image Text Translator (translation wrapper for image text)
# ---------------------------------------------------------------------------

class ImageTextTranslator:
    """Lightweight translation wrapper for OCR-detected text (no escape codes)."""

    def __init__(
        self,
        backend: str = "google",
        source_lang: str = "ja",
        target_lang: str = "en",
        api_key: Optional[str] = None,
    ) -> None:
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._backend_name = backend.lower()
        self._client: Any = None

        if self._backend_name == "deepl":
            key = api_key or os.getenv("DEEPL_API_KEY", "")
            if not key:
                raise ValueError("DeepL API key required for image translation.")
            import deepl
            self._client = deepl.Translator(key)
        elif self._backend_name == "google":
            from deep_translator import GoogleTranslator
            self._client = GoogleTranslator(source=source_lang, target=target_lang)
        elif self._backend_name in ("marian", "marianmt"):
            from rpgmaker_translator import MarianMTBackend
            self._marian = MarianMTBackend(
                source_lang=source_lang, target_lang=target_lang
            )
        else:
            raise ValueError(f"Unknown image translation backend: '{backend}'")

    @staticmethod
    def _deepl_lang(code: str, *, target: bool = False) -> Optional[str]:
        c = code.lower()
        if c in ("ja", "jp"):
            return "JA"
        if c.startswith("en"):
            return "EN-US" if target else "EN"
        if c in ("zh", "zh-cn"):
            return "ZH-HANS" if target else "ZH"
        return c.upper()

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return text
        try:
            if self._backend_name == "deepl":
                res = self._client.translate_text(
                    text,
                    source_lang=self._deepl_lang(self.source_lang),
                    target_lang=self._deepl_lang(self.target_lang, target=True),
                )
                return res.text
            if self._backend_name in ("marian", "marianmt"):
                return self._marian.translate(
                    text, self.source_lang, self.target_lang
                )
            return self._client.translate(text) or text
        except Exception as exc:
            logger.warning("Image text translation failed for '%s': %s", text[:30], exc)
            return text


# ---------------------------------------------------------------------------
# Core Image Translator
# ---------------------------------------------------------------------------

class ImageTranslator:
    """Detect, translate, and render text in a single image."""

    def __init__(
        self,
        source_lang: str = "ja",
        target_lang: str = "en",
        min_confidence: float = 0.6,
        backend: str = "google",
        api_key: Optional[str] = None,
        use_gpu: Optional[bool] = None,
    ) -> None:
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.min_confidence = min_confidence

        # Map source language to EasyOCR code.
        lang_map = {"ja": "ja", "zh": "ch_sim", "ko": "ko"}
        ocr_langs = [lang_map.get(source_lang, source_lang), "en"]

        self.ocr = EasyOCREngine(languages=ocr_langs, use_gpu=use_gpu)
        self.inpainter = Inpainter()
        self.renderer = TextRenderer()
        self.translator = ImageTextTranslator(
            backend=backend,
            source_lang=source_lang,
            target_lang=target_lang,
            api_key=api_key,
        )
        self._skip_re = [re.compile(p, re.IGNORECASE) for p in SKIP_FILE_PATTERNS]

    def should_skip(self, filepath: str) -> Tuple[bool, str]:
        """Check if a file should be skipped based on name/path."""
        path = Path(filepath)
        for pat in self._skip_re:
            if pat.search(path.stem):
                return True, f"matches skip pattern: {pat.pattern}"
        for folder in SKIP_FOLDERS:
            if folder.lower() in [p.lower() for p in path.parts]:
                return True, f"in skip folder: {folder}"
        return False, ""

    def process(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        force: bool = False,
    ) -> ImageTranslationResult:
        """Process a single image. Only modifies it if CJK text is found."""
        result = ImageTranslationResult(original_path=image_path)

        if not force:
            skip, reason = self.should_skip(image_path)
            if skip:
                result.success = True
                result.skip_reason = reason
                return result

        # Load image with alpha channel.
        try:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                result.error = "Failed to load image"
                return result
        except Exception as exc:
            result.error = str(exc)
            return result

        has_alpha = len(image.shape) == 3 and image.shape[2] == 4

        # Composite on white background for OCR.
        if has_alpha:
            bgr = image[:, :, :3].copy()
            alpha = image[:, :, 3].astype(np.float32) / 255.0
            alpha3 = np.stack([alpha] * 3, axis=-1)
            white = np.ones_like(bgr, dtype=np.float32) * 255
            bgr_for_ocr = (bgr.astype(np.float32) * alpha3 + white * (1 - alpha3)).astype(np.uint8)
        else:
            bgr_for_ocr = image

        # OCR.
        all_regions = self.ocr.detect(bgr_for_ocr)
        valid = [r for r in all_regions if r.is_valid(self.min_confidence)]

        if not valid:
            result.success = True
            result.skip_reason = "No translatable CJK text found"
            return result

        logger.info("Found %d text region(s) in %s", len(valid), Path(image_path).name)

        # Translate.
        for region in valid:
            region.translated = self.translator.translate(region.text)
            logger.debug("  '%s' → '%s'", region.text, region.translated)

        # Inpaint.
        mask = self.inpainter.create_mask(image.shape, valid)
        if has_alpha:
            filled = self.inpainter.simple_fill(image, mask)
        else:
            filled = self.inpainter.inpaint(image, mask)

        # Render translated text.
        final = self.renderer.render(filled, valid, self.target_lang)

        # Save.
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_translated{ext}"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if has_alpha and output_path.lower().endswith(".png") and final.shape[2] == 4:
            pil_img = Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGRA2RGBA))
            pil_img.save(output_path, "PNG")
        else:
            cv2.imwrite(output_path, final)

        result.output_path = output_path
        result.regions = valid
        result.success = True
        result.modified = True
        logger.info("Translated image → %s", Path(output_path).name)
        return result


# ---------------------------------------------------------------------------
# RPG Maker integration (handles game structure + encryption)
# ---------------------------------------------------------------------------

class RPGMakerImageTranslator(ImageTranslator):
    """Translate images within an RPG Maker MV/MZ game directory."""

    def __init__(self, game_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.game_path = Path(game_path)
        self.decryptor = None
        self.encryptor = None
        self._init_crypto()

    def _init_crypto(self) -> None:
        try:
            from rpgmv_crypto import RPGMVKeyManager, RPGMVDecryptor, RPGMVEncryptor

            info = RPGMVKeyManager.get_game_info(str(self.game_path))
            if info.has_encrypted_images or self._has_encrypted():
                self.decryptor = RPGMVDecryptor(info.encryption_key)
                if info.encryption_key:
                    self.encryptor = RPGMVEncryptor(info.encryption_key)
                key_preview = info.encryption_key[:8] if info.encryption_key else "none"
                logger.info("Crypto ready (key: %s…)", key_preview)
        except ImportError:
            logger.warning("rpgmv_crypto unavailable — encrypted images won't be processed")
        except Exception as exc:
            logger.warning("Crypto init failed: %s", exc)

    def _has_encrypted(self) -> bool:
        for img in [self.game_path / "www" / "img", self.game_path / "img"]:
            if img.exists() and list(img.rglob("*.rpgmvp")):
                return True
        return False

    def _img_base(self) -> Path:
        www = self.game_path / "www" / "img"
        return www if www.exists() else self.game_path / "img"

    # Alias for backwards-compat with main.py
    def process_image(
        self, image_path: str, output_path: Optional[str] = None, force: bool = False
    ) -> ImageTranslationResult:
        return self.process(image_path, output_path, force)

    def translate_all_rpgmvp_files(
        self,
        replace_originals: bool = True,
        backup: bool = True,
        folders: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Translate all encrypted images. Only modifies those with real text."""
        stats: Dict[str, Any] = {
            "total_files": 0,
            "files_with_text": 0,
            "files_translated": 0,
            "files_skipped": 0,
            "files_failed": 0,
            "output_folder": None,
            "details": [],
        }

        if not self.decryptor:
            stats["error"] = "No decryptor available"
            return stats

        img_base = self._img_base()
        project = self.game_path / "translation_project"
        dec_dir = project / "img_decrypted"
        tr_dir = project / "img_translated"
        enc_dir = project / "img_encrypted"
        bak_dir = project / "backup" / "img"

        for d in (dec_dir, tr_dir, enc_dir):
            d.mkdir(parents=True, exist_ok=True)
        stats["output_folder"] = str(tr_dir)

        # Collect files.
        search = folders or TEXT_FOLDERS
        files: List[Path] = []
        for folder_name in search:
            folder = img_base / folder_name
            if folder.exists():
                files.extend(folder.rglob("*.rpgmvp"))
                files.extend(folder.rglob("*.png_"))
        files = list(set(files))
        stats["total_files"] = len(files)

        if not files:
            logger.info("No encrypted images found in text folders")
            return stats

        logger.info("Scanning %d images for text …", len(files))

        for idx, src in enumerate(files):
            tag = f"[{idx + 1}/{len(files)}]"
            logger.info("%s %s", tag, src.name)

            detail: Dict[str, Any] = {
                "file": str(src.relative_to(self.game_path)),
                "status": "pending",
            }

            try:
                rel = src.relative_to(img_base)

                # Decrypt.
                dec_path = dec_dir / rel.with_suffix(".png")
                dec_path.parent.mkdir(parents=True, exist_ok=True)
                self.decryptor.decrypt_file(str(src), str(dec_path))

                # Process.
                tr_path = tr_dir / rel.with_suffix(".png")
                tr_path.parent.mkdir(parents=True, exist_ok=True)
                result = self.process(str(dec_path), str(tr_path), force=True)

                if not result.success:
                    detail["status"] = "failed"
                    detail["error"] = result.error
                    stats["files_failed"] += 1
                    stats["details"].append(detail)
                    continue

                if not result.modified:
                    detail["status"] = "skipped"
                    stats["files_skipped"] += 1
                    stats["details"].append(detail)
                    continue

                stats["files_with_text"] += 1

                # Re-encrypt & replace.
                if self.encryptor:
                    enc_path = enc_dir / rel
                    enc_path.parent.mkdir(parents=True, exist_ok=True)
                    self.encryptor.encrypt_file(str(tr_path), str(enc_path))

                    if replace_originals:
                        if backup:
                            bak = bak_dir / rel
                            bak.parent.mkdir(parents=True, exist_ok=True)
                            if not bak.exists():
                                shutil.copy2(src, bak)
                        shutil.copy2(enc_path, src)
                        logger.info("  → Replaced: %s", src.name)

                detail["status"] = "translated"
                stats["files_translated"] += 1

            except Exception as exc:
                detail["status"] = "failed"
                detail["error"] = str(exc)
                stats["files_failed"] += 1
                logger.error("  → Error: %s", exc)

            stats["details"].append(detail)

        # Summary.
        logger.info("=" * 50)
        logger.info("Image translation complete")
        logger.info("  Scanned:    %d", stats["total_files"])
        logger.info("  With text:  %d", stats["files_with_text"])
        logger.info("  Translated: %d", stats["files_translated"])
        logger.info("  Skipped:    %d", stats["files_skipped"])
        logger.info("  Failed:     %d", stats["files_failed"])

        return stats
