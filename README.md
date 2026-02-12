<p align="center">
  <h1 align="center">RPG Maker Translator</h1>
  <p align="center">
    <strong>Automatic, free translation of RPG Maker MV / MZ games — text <em>and</em> images.</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#how-it-works">How It Works</a> •
    <a href="#faq">FAQ</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white" alt="Python 3.8+">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
    <img src="https://img.shields.io/badge/RPG%20Maker-MV%20%7C%20MZ-orange" alt="MV | MZ">
    <img src="https://img.shields.io/badge/GPU-CUDA%20supported-brightgreen?logo=nvidia" alt="CUDA">
    <img src="https://img.shields.io/badge/cost-100%25%20free-success" alt="Free">
  </p>
</p>

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Translate an entire game (Japanese → English)
python main.py /path/to/game
```

That's it. The tool auto-detects the game engine (MV or MZ), extracts all dialogue, menus, and system text, translates everything with Google Translate (free), patches plugins for compatibility, word-wraps text to fit message windows, OCRs images for embedded text, and writes translations back — all in one command.

---

## Features

| Feature | Description |
|---|---|
| **One-command translation** | `python main.py /path/to/game` handles everything |
| **100% free** | Google Translate by default — no API keys needed |
| **Offline mode** | MarianMT backend — no internet, no API key, no limits |
| **Text translation** | Dialogue, choices, menus, skills, items, system messages, … |
| **Image translation** | OCR → inpaint → re-render translated text on game images |
| **Encrypted image support** | Decrypts `.rpgmvp` files, translates, re-encrypts seamlessly |
| **GPU acceleration** | Auto-detects CUDA for faster OCR and MarianMT inference |
| **MV & MZ support** | Handles both RPG Maker MV and MZ folder structures |
| **Plugin patching** | Fixes known issues (e.g. LL_GalgeChoiceWindowMV space bug) |
| **Word wrap** | Auto-generates a JS plugin to prevent text overflow |
| **Escape code safety** | Preserves `\V[n]`, `\C[n]`, `\N[n]`, `\I[n]` etc. during translation |
| **Resume support** | Interrupted? Just re-run — progress is auto-saved |
| **CSV export/import** | Edit translations manually, then re-apply |
| **Full backup** | Original files backed up before any modification |
| **DeepL (optional)** | Premium backend for higher-quality translations |

---

## Installation

### Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **4 GB RAM** minimum (8 GB recommended)
- **NVIDIA GPU** optional but recommended for faster image OCR

### Standard Install (text + images)

```bash
git clone https://github.com/Meganano/RPGMakerTranslator.git
cd RPGMakerTranslator
pip install -r requirements.txt
```

### Text-Only Install (lightest)

```bash
pip install deep-translator python-dotenv
```

### Offline / MarianMT Install (no internet needed after setup)

```bash
# Installs the Helsinki-NLP MarianMT models via Hugging Face
pip install deep-translator python-dotenv transformers sentencepiece torch
```

The first run downloads the translation model (~300 MB). After that, translation works fully offline with no API key and no rate limits.

### GPU Install (fastest for images)

```bash
# Install base dependencies
pip install deep-translator python-dotenv Pillow opencv-python numpy easyocr

# Install PyTorch with CUDA (pick your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Usage

### Basic — translate everything

```bash
python main.py /path/to/game
```

Defaults: Japanese → English, Google Translate, images included, GPU auto-detected.

### Specify languages

```bash
# Japanese → Spanish
python main.py /path/to/game -s ja -t es

# Chinese → English
python main.py /path/to/game -s zh -t en

# Korean → French
python main.py /path/to/game -s ko -t fr
```

### Text only (skip images)

```bash
python main.py /path/to/game --no-images
```

### Export CSV for manual editing

```bash
# Export
python main.py /path/to/game --export-only

# Edit the CSV at: /path/to/game/translation_project/translations_en_v2.csv

# Import and apply
python main.py /path/to/game --import-csv
```

### Use DeepL (premium, higher quality)

```bash
# With API key (free tier: https://www.deepl.com/pro-api)
python main.py /path/to/game --backend deepl --api-key YOUR_KEY

# Or set environment variable
export DEEPL_API_KEY=YOUR_KEY
python main.py /path/to/game --backend deepl
```

### Use MarianMT (offline, unlimited)

```bash
# No API key, no internet (after first model download)
python main.py /path/to/game --backend marian
```

MarianMT uses Helsinki-NLP models via Hugging Face. Dependencies (`transformers`, `sentencepiece`) are auto-installed on first use if missing. GPU is used automatically when available.

### All options

```
usage: main.py game_path [-s SOURCE] [-t TARGET] [-b {google,deepl,marian}]
                         [--api-key KEY] [--export-only] [--import-csv]
                         [--no-images] [--no-backup] [--no-patch]
                         [--no-wordwrap] [--no-replace-images]
                         [--batch-size N] [--delay SECS] [-o DIR] [-v]
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Translation Pipeline                      │
├──────┬──────────────────────────────────────────────────────┤
│ 1.   │  EXTRACT     Parse all JSON data files               │
│      │              Collect dialogue, menus, system text     │
│      │              Handle escape codes (\V[n], \C[n], …)   │
├──────┼──────────────────────────────────────────────────────┤
│ 2.   │  PATCH       Fix JS plugins for translation compat   │
│      │              Generate word-wrap plugin                │
├──────┼──────────────────────────────────────────────────────┤
│ 3.   │  TRANSLATE   Batch translate via Google / DeepL       │
│      │              Protect escape codes with placeholders   │
│      │              Auto-retry on failure                    │
├──────┼──────────────────────────────────────────────────────┤
│ 4.   │  APPLY       Write translations back to JSON files   │
│      │              Word-wrap text to fit message windows    │
├──────┼──────────────────────────────────────────────────────┤
│ 5.   │  IMAGES      Decrypt .rpgmvp → OCR → translate →     │
│      │              inpaint → render → re-encrypt            │
│      │              Only modifies images with actual text    │
└──────┴──────────────────────────────────────────────────────┘
```

### Architecture

```
main.py                        CLI entry point & pipeline orchestrator
rpgmaker_translator.py         Text extraction, translation backends (Google/DeepL/MarianMT), application
rpgmaker_image_translator.py   Image OCR, inpainting, text rendering
rpgmv_crypto.py                .rpgmvp encryption / decryption
apply_translated_images.py     Standalone image apply / restore utility
```

### Output Structure

```
GameFolder/
├── translation_project/
│   ├── project_v2.json          # Translation database (auto-saved)
│   ├── translations_en_v2.csv   # Editable CSV
│   ├── img_decrypted/           # Decrypted PNGs (for review)
│   ├── img_translated/          # Translated PNGs
│   ├── img_encrypted/           # Re-encrypted .rpgmvp files
│   ├── backup/                  # Original JSON backups
│   └── js_backup/               # Original plugin JS backups
└── [game data files — modified in place]
```

---

## Supported Content

### Text

| Data File | Translated Fields |
|---|---|
| Map001.json … | Dialogue, choices, scrolling text, speaker names, event names |
| Actors.json | Name, nickname, profile |
| Skills.json | Name, description, battle messages |
| Items / Weapons / Armors | Name, description |
| Enemies.json | Name |
| States.json | Name, messages |
| System.json | Game title, currency, terms, battle messages |
| CommonEvents / Troops | Event commands |

### Images

| Folder | What It Contains |
|---|---|
| `pictures/` | CGs, UI overlays, tutorial images |
| `titles1/` | Title screen graphics |
| `titles2/` | Title screen overlays |

Images are only modified when OCR detects actual CJK text (strict validation prevents false positives on UI elements).

### Plugin Commands

| Plugin | Support |
|---|---|
| LL_GalgeChoiceWindowMV | Full (choices + text + auto-patch) |
| Generic MV plugins | Text/message sub-commands |

---

## Translation Backends

| Backend | Cost | Quality | Speed | Setup | Internet |
|---|---|---|---|---|---|
| **Google** (default) | Free | Good | ~50 strings/sec | None | Required |
| **DeepL** | Free tier / Paid | Excellent | ~100 strings/sec | API key | Required |
| **MarianMT** | Free | Good | ~30-80 strings/sec | Auto-install | Offline |

- **Google Translate** is the default — zero setup and handles all supported languages well.
- **DeepL** offers the highest translation quality — [get a free API key](https://www.deepl.com/pro-api) and use `--backend deepl`.
- **MarianMT** runs entirely locally with no API key and no rate limits — ideal for large games or restricted environments. Uses GPU automatically when available. First run downloads the model (~300 MB); after that, no internet needed.

---

## Restoring Originals

All original files are backed up before modification.

```bash
# Restore original images
python apply_translated_images.py /path/to/game --restore

# Restore original JSON data (manual)
# Copy files from: /path/to/game/translation_project/backup/
```

---

## FAQ

<details>
<summary><strong>How long does translation take?</strong></summary>

Depends on game size. A typical 10-hour RPG with ~5,000 text entries:
- Text: 2–5 minutes (Google), 1–2 minutes (DeepL)
- Images: 1–10 minutes depending on count and GPU availability

</details>

<details>
<summary><strong>Can I edit translations before applying?</strong></summary>

Yes! Use `--export-only` to generate a CSV, edit it in any spreadsheet editor, then `--import-csv` to apply your changes.

</details>

<details>
<summary><strong>What if translation is interrupted?</strong></summary>

Progress is auto-saved after every batch. Just run the same command again — it picks up where it left off.

</details>

<details>
<summary><strong>Does it support MV and MZ?</strong></summary>

Yes. Both RPG Maker MV (www/data/) and MZ (data/) are auto-detected.

</details>

<details>
<summary><strong>Will it break my game?</strong></summary>

Backups are created automatically. If something goes wrong, copy the backup files back. The tool preserves all escape codes, JSON formatting, and plugin compatibility.

</details>

<details>
<summary><strong>How do I use GPU?</strong></summary>

Install the CUDA version of PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
The tool auto-detects GPU and uses it for OCR. No flags needed.

</details>

<details>
<summary><strong>Can I translate to/from any language?</strong></summary>

Google Translate supports 100+ languages. Common pairs: ja, en, zh, ko, es, fr, de, pt, ru, it, ar, th, vi. DeepL supports 30+ languages with higher quality. MarianMT supports 100+ language pairs via Helsinki-NLP models (coverage varies by pair).

</details>

<details>
<summary><strong>Can I translate without internet?</strong></summary>

Yes! Use the MarianMT backend: `python main.py /path/to/game --backend marian`. The first run downloads the translation model, but after that everything works fully offline.

</details>

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Keep code PEP 8 compliant with type hints
4. Test with at least one MV and one MZ game
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/Meganano/RPGMakerTranslator.git
cd RPGMakerTranslator
pip install -r requirements.txt
```

---

## License

[MIT License](LICENSE) — use it freely, modify it, distribute it, commercial use allowed.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/Meganano">Meganano</a>
</p>
