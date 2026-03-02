# Epstein Leaked Files - LLM Recognition Outputs

## Overview
This repository stores recognition outputs generated from leaked Epstein-case source files (PDF/PNG/video-derived content).
The outputs are produced by an automated recursive pipeline that routes pages, runs OCR/visual analysis, and writes per-file `.txt` results.

## Provenance
All recognition in this repository was generated using the script:
`./recursive_llm_processor.py`

The same script is included in this repository as:
`recursive_llm_processor.py`

## Research-Only Disclaimer
This repository is provided **strictly for research, auditing, and technical analysis purposes only**.
Do not use these materials for harassment, doxxing, defamation, or any unlawful activity.
Users are responsible for complying with all applicable laws, privacy requirements, and platform policies.

## Requirements
- Python 3.10+
- Ollama running locally or remotely (default endpoint: `http://localhost:11434/api/generate`)
- Python packages:
  - `requests`
  - `Pillow`
  - `tqdm`
  - `pdf2image`
  - `opencv-python` (for video processing)
- Poppler utilities available to `pdf2image` (for PDF page conversion)

Install dependencies (example):

```bash
python -m pip install requests pillow tqdm pdf2image opencv-python
```

## Usage
Run from repository root:

```bash
python recursive_llm_processor.py --root "."
```

### Common examples
Resume processing from saved state:

```bash
python recursive_llm_processor.py \
  --root "." \
  --resume
```

Re-run and overwrite existing `.txt` outputs:

```bash
python recursive_llm_processor.py \
  --root "." \
  --overwrite
```

Fast-scan mode (fewer model calls, faster throughput):

```bash
python recursive_llm_processor.py \
  --root "." \
  --fast-scan
```

Process only PDFs and PNGs (skip videos):

```bash
python recursive_llm_processor.py \
  --root "." \
  --no-video
```

## Important CLI options
- `--root`: root directory to scan recursively.
- `--state-file`: resumable progress file (default: `.doj_progress.json`).
- `--resume` / `--no-resume`: continue from prior progress or start fresh.
- `--overwrite`: replace existing output `.txt` files.
- `--ollama-url`: Ollama generate API URL.
- `--timeout`, `--retries`: API reliability controls.
- `--pdf`, `--png`, `--video`: include/exclude file types.
- `--video-interval`: seconds between extracted frames.
- `--router-votes`: number of route votes per page.
- `--fast-scan`: streamlined, lower-cost processing mode.

## Output behavior
- For each source file, output is written next to the source as a `.txt` file.
- Progress/state is tracked in `.doj_progress.json`.
- Optional scan cache and event logs are written to `.doj_scan_cache.json.gz` and `.doj_logs/`.

## Notes
- Default model mapping in `recursive_llm_processor.py` currently points all stages to `qwen3-vl:8b-instruct-q8_0`.
- You can edit `DEFAULT_MODELS` in the script if you want per-stage model customization.

## Support This Project
If you find it useful, consider supporting its maintenance and hosting:

ETH: 0x550c828553e4d382cd6651c7d759317b8d615006