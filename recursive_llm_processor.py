import os
import sys
import base64
import gzip
import io
import json
import re
import time
import argparse
import logging
import atexit
import platform
import shutil
import socket
import subprocess
import threading
import traceback
import uuid
from datetime import datetime
from typing import Any, cast

import requests
from PIL import Image
from tqdm import tqdm

from pdf2image import convert_from_path, pdfinfo_from_path
from logging.handlers import RotatingFileHandler

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_OLLAMA_API_URL = "http://localhost:11434/api/generate"

# 模型名称配置 (已更新为 qwen3.5:9b)
DEFAULT_MODELS = {
    "router": "qwen3.5:9b",
    "ocr_glm": "glm-ocr:latest",
    "ocr_qwen": "qwen3.5:9b",
    "desc_qwen": "qwen3.5:9b",
    "refiner": "qwen3.5:9b"
}

OLLAMA_REQUEST_OPTIONS: dict[str, Any] | None = None
MAX_NUM_CTX = 8192
OLLAMA_STREAM: bool = True

_GLOBAL_EVENT_LOG: "_DiskEventLog | None" = None
_GLOBAL_EVENT_LOG_ARGS: "argparse.Namespace | None" = None
_GLOBAL_CONSOLE_SINK: "Any | None" = None
_GLOBAL_CONSOLE_LOCK = threading.Lock()


def _emit_event(event: str, **fields: Any) -> None:
    ev = _GLOBAL_EVENT_LOG
    if ev is None:
        return
    try:
        ev.emit(event, **fields)
    except Exception:
        pass


def _thinking_prefix_zh() -> str:
    return (
        "你可以在心里逐步思考，但不要输出推理过程。\n"
        "只输出最终答案，保持格式与要求严格一致。\n"
    )


def _apply_thinking(prompt: str, enable: bool) -> str:
    if not enable:
        return prompt
    return _thinking_prefix_zh() + prompt


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class _DiskEventLog:
    def __init__(self, path: str, *, run_id: str, fsync: bool = False, fsync_every: int = 0) -> None:
        self.path = path
        self.run_id = run_id
        self._fsync = bool(fsync)
        self._fsync_every = int(fsync_every or 0)
        self._emit_count = 0
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fp = open(path, "a", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        payload: dict[str, Any] = {
            "ts": _now_iso(),
            "run_id": self.run_id,
            "event": event,
        }
        payload.update(fields)
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            self._emit_count += 1
            self._fp.write(line + "\n")
            self._fp.flush()
            if self._fsync:
                every = self._fsync_every
                if every <= 0 or (self._emit_count % every) == 0:
                    try:
                        os.fsync(self._fp.fileno())
                    except Exception:
                        pass

    def close(self) -> None:
        with self._lock:
            try:
                self._fp.close()
            except Exception:
                pass


def _cmd_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _read_proc_meminfo() -> dict[str, int]:
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(r"^([A-Za-z0-9_()]+):\s+(\d+)\s+kB\s*$", line)
                if not m:
                    continue
                info[m.group(1)] = int(m.group(2)) * 1024
    except Exception:
        pass
    return info


def _read_proc_self_rss_bytes() -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        pass
    return None


def _gpu_stats_nvidia_smi() -> list[dict[str, Any]] | None:
    if not _cmd_exists("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            timeout=5,
            text=True,
        )
        gpus: list[dict[str, Any]] = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "mem_total_mib": int(parts[2]),
                    "mem_used_mib": int(parts[3]),
                    "temp_c": int(parts[4]),
                    "util_pct": int(parts[5]),
                }
            )
        return gpus
    except Exception:
        return None


def _system_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    snap: dict[str, Any] = {"host": socket.gethostname(), "pid": os.getpid()}

    try:
        if hasattr(os, "getloadavg"):
            la = os.getloadavg()
            snap["loadavg"] = {"1m": la[0], "5m": la[1], "15m": la[2]}
    except Exception:
        pass

    meminfo = _read_proc_meminfo()
    if meminfo:
        snap["mem"] = {
            "total_bytes": meminfo.get("MemTotal"),
            "avail_bytes": meminfo.get("MemAvailable"),
            "swap_total_bytes": meminfo.get("SwapTotal"),
            "swap_free_bytes": meminfo.get("SwapFree"),
        }

    rss = _read_proc_self_rss_bytes()
    if rss is not None:
        snap["proc"] = {"rss_bytes": rss}

    try:
        root = os.path.abspath(str(getattr(args, "root", os.getcwd()) or os.getcwd()))
        usage = shutil.disk_usage(root)
        snap["disk"] = {
            "path": root,
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
        }
    except Exception:
        pass

    gpus = _gpu_stats_nvidia_smi()
    if gpus is not None:
        snap["gpus"] = gpus

    return snap


def _start_metrics_thread(args: argparse.Namespace, ev: _DiskEventLog) -> None:
    interval_s = float(getattr(args, "metrics_interval", 0) or 0)
    if interval_s <= 0:
        return
    stop = threading.Event()
    setattr(args, "_metrics_stop", stop)

    def _loop() -> None:
        while not stop.is_set():
            try:
                ev.emit("system_snapshot", **_system_snapshot(args))
            except Exception:
                pass
            stop.wait(interval_s)

    t = threading.Thread(target=_loop, name="doj-metrics", daemon=True)
    setattr(args, "_metrics_thread", t)
    t.start()


def _configure_disk_logging(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "disk_log", True)):
        return

    log_dir = str(getattr(args, "log_dir", ".doj_logs") or ".doj_logs").strip() or ".doj_logs"
    if not os.path.isabs(log_dir):
        log_dir = os.path.abspath(os.path.join(os.getcwd(), log_dir))
    os.makedirs(log_dir, exist_ok=True)

    run_id = str(getattr(args, "run_id", "") or "").strip()
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    setattr(args, "_run_id", run_id)

    global _GLOBAL_EVENT_LOG, _GLOBAL_EVENT_LOG_ARGS, _GLOBAL_CONSOLE_SINK
    _GLOBAL_EVENT_LOG_ARGS = args

    level_name = str(getattr(args, "log_level", "INFO") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO

    # Text log (rotating)
    log_path = os.path.join(log_dir, f"{run_id}.log")
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Replace root handlers so console + file are consistent.
    logging.basicConfig(level=level, handlers=[logging.StreamHandler(), file_handler], force=True)

    ev_path = os.path.join(log_dir, f"{run_id}.events.jsonl")
    ev = _DiskEventLog(
        ev_path,
        run_id=run_id,
        fsync=bool(getattr(args, "event_fsync", False)),
        fsync_every=int(getattr(args, "event_fsync_every", 0) or 0),
    )
    setattr(args, "_event_log", ev)
    _GLOBAL_EVENT_LOG = ev

    console_path = os.path.join(log_dir, f"{run_id}.console.log")
    if bool(getattr(args, "capture_console", True)):
        try:
            _GLOBAL_CONSOLE_SINK = open(console_path, "a", encoding="utf-8")
        except Exception:
            _GLOBAL_CONSOLE_SINK = None

    def _cleanup() -> None:
        try:
            st: threading.Event | None = getattr(args, "_metrics_stop", None)
            if st is not None:
                st.set()
        except Exception:
            pass
        try:
            thr: threading.Thread | None = getattr(args, "_metrics_thread", None)
            if thr is not None:
                thr.join(timeout=2.0)
        except Exception:
            pass
        try:
            ev.emit("run_end")
        except Exception:
            pass
        try:
            ev.close()
        except Exception:
            pass
        try:
            sink = _GLOBAL_CONSOLE_SINK
            if sink is not None:
                with _GLOBAL_CONSOLE_LOCK:
                    try:
                        sink.flush()
                    except Exception:
                        pass
                    try:
                        sink.close()
                    except Exception:
                        pass
        except Exception:
            pass

    atexit.register(_cleanup)

    ev.emit(
        "run_start",
        argv=list(getattr(args, "_argv", []) or []),
        python=sys.version,
        platform=platform.platform(),
        ollama_url=str(getattr(args, "ollama_url", "") or ""),
        models=dict(getattr(args, "models", {}) or {}),
        keep_alive=int(getattr(args, "keep_alive", 0) or 0),
        num_ctx=int(getattr(args, "num_ctx", 0) or 0),
        num_predict=int(getattr(args, "num_predict", 0) or 0),
        temperature_refine=float(getattr(args, "temperature_refine", 0.2) or 0.2),
        detail_multiplier=float(getattr(args, "detail_multiplier", 3.0) or 3.0),
        clean_tag=str(getattr(args, "clean_tag", "DOJ_CLEAN_V2") or "DOJ_CLEAN_V2"),
        enrich_from_output=bool(getattr(args, "enrich_from_output", True)),
        root=os.path.abspath(str(getattr(args, "root", os.getcwd()) or os.getcwd())),
        log_dir=log_dir,
        log_path=log_path,
        events_path=ev_path,
        console_path=console_path if bool(getattr(args, "capture_console", True)) else "",
        capture_console=bool(getattr(args, "capture_console", True)),
        event_fsync=bool(getattr(args, "event_fsync", False)),
        event_fsync_every=int(getattr(args, "event_fsync_every", 0) or 0),
        model_events=bool(getattr(args, "model_events", True)),
        metrics_log=bool(getattr(args, "metrics_log", True)),
        metrics_interval=int(getattr(args, "metrics_interval", 0) or 0),
    )

    if bool(getattr(args, "metrics_log", True)):
        _start_metrics_thread(args, ev)


def _norm_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _resolve_output_path(src_path: str, output_suffix: str) -> str:
    """Map source file path to output txt path using a configurable suffix."""
    stem, _ = os.path.splitext(os.path.abspath(src_path))
    suffix = str(output_suffix or ".txt").strip()
    if not suffix:
        suffix = ".txt"
    return stem + suffix


def _trace_for_file(file_path: str, *, kind: str, page_num: int | None = None, pages_total: int | None = None) -> str:
    base = os.path.abspath(file_path)
    if page_num is not None:
        if pages_total:
            return f"{base} [{kind} page {page_num}/{pages_total}]"
        return f"{base} [{kind} page {page_num}]"
    return f"{base} [{kind}]"


def _extract_signature_blocks(raw: str) -> list[str]:
    blocks: list[str] = []
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().lower().startswith("signed"):
            block_lines: list[str] = [lines[i].rstrip()]
            j = i + 1
            while j < len(lines) and len(block_lines) < 8:
                if lines[j].strip() == "":
                    break
                block_lines.append(lines[j].rstrip())
                j += 1
            blocks.append("\n".join(block_lines).strip())
            i = j
            continue
        i += 1
    return [b for b in blocks if b]


def _extract_id_tokens(raw: str) -> list[str]:
    tokens = sorted(set(re.findall(r"\bEFTA\d{6,}\b", raw)))
    return tokens


def _append_missing_anchors(raw: str, refined: str) -> str:
    missing_sections: list[str] = []

    for block in _extract_signature_blocks(raw):
        if block and block not in refined:
            missing_sections.append(block)

    ids = _extract_id_tokens(raw)
    missing_ids = [t for t in ids if t not in refined]
    if missing_ids:
        missing_sections.append("IDs: " + ", ".join(missing_ids))

    if not missing_sections:
        return refined

    suffix = "\n\n---\n## Missing (verbatim)\n\n" + "\n\n".join(missing_sections).strip() + "\n"
    return (refined.rstrip() + suffix)


def _enforce_3x_detail(
    args: argparse.Namespace,
    source_text: str,
    refined_text: str,
    *,
    trace: str | None = None,
    step: tuple[int, int] | None = None,
) -> str:
    src_len = len(source_text or "")
    if src_len == 0:
        return refined_text
    multiplier = float(getattr(args, "detail_multiplier", 3.0) or 3.0)
    if multiplier < 1.2:
        multiplier = 1.2
    target_len = int(src_len * multiplier)

    if len(refined_text or "") >= target_len:
        return refined_text

    expand_prompt = f"""
你将扩写已有文本，目标是“信息量接近原始输入的 3 倍”。

要求：
1) 保留原文事实，不得杜撰；
2) 逐段细化：时间、地点、人物、动作、证据、编号、上下文；
3) 对可见文本做更完整转录（含编号、页眉页脚、签名线索）；
4) 使用 Markdown 结构输出，层级清晰；
5) 若信息缺失，用“未见明确证据”标注，不可脑补；
6) 输出尽可能详细，至少接近目标长度。

--- 原始输入 ---
{source_text}

--- 当前版本（需扩写）---
{refined_text}
"""
    expanded = call_ollama(
        args.ollama_url,
        args.models["refiner"],
        expand_prompt,
        timeout_s=args.timeout,
        retries=args.retries,
        print_model_output=args.print_model_output,
        keep_alive=args.keep_alive,
        num_ctx=int(getattr(args, "num_ctx_heavy", 8192) or 8192),
        num_predict=max(int(getattr(args, "num_predict_heavy", 4096) or 4096), 6144),
        temperature=float(getattr(args, "temperature_refine", 0.2) or 0.2),
        trace=f"{(trace or '').strip()} [refine_expand]".strip(),
        step=step,
    )
    return expanded or refined_text


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _extract_numbered_sections(text: str, header_label: str) -> dict[int, str]:
    sections: dict[int, str] = {}
    if not text.strip():
        return sections
    pattern = re.compile(rf"^##\s+{re.escape(header_label)}\s+(\d+)\b[^\n]*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        try:
            n = int(m.group(1))
        except Exception:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections[n] = body
    return sections


def _extract_existing_page_context(output_path: str, page_num: int) -> str:
    text = _read_text_file(output_path)
    return _extract_numbered_sections(text, "Page").get(int(page_num), "")


def _extract_existing_frame_context(output_path: str, frame_num: int) -> str:
    text = _read_text_file(output_path)
    return _extract_numbered_sections(text, "Frame").get(int(frame_num), "")


def _file_meta_marker(args: argparse.Namespace, *, file_path: str, file_type: str) -> str:
    tag = str(getattr(args, "clean_tag", "DOJ_CLEAN_V2") or "DOJ_CLEAN_V2").strip()
    mode = "enrich" if bool(getattr(args, "enrich_from_output", True)) else "normal"
    run_id = str(getattr(args, "_run_id", "") or "").strip()
    return (
        "<!-- DOJ_FILE_META "
        f"tag={tag} "
        f"type={file_type} "
        f"mode={mode} "
        f"detail_multiplier={float(getattr(args, 'detail_multiplier', 3.0) or 3.0):.2f} "
        f"run_id={run_id or 'na'} "
        f"source={os.path.abspath(file_path)} "
        f"ts={_now_iso()}"
        " -->\n\n"
    )


def _output_has_clean_tag(output_path: str, clean_tag: str) -> bool:
    if not output_path or not os.path.exists(output_path):
        return False
    try:
        with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(4096)
    except Exception:
        return False
    if f"tag={clean_tag}" in head:
        return True
    if f"[TAG:{clean_tag}]" in head:
        return True
    return False


def _output_is_truncated(output_path: str, file_type: str) -> bool:
    """
    Quick integrity check for resumed outputs.
    Treat as truncated if file has <=1 line or missing expected section header.
    """
    if not output_path or not os.path.exists(output_path):
        return True
    expected_prefix = "## Frame " if file_type == "video" else "## Page "
    line_count = 0
    has_header = False
    try:
        with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_count += 1
                if line.startswith(expected_prefix):
                    has_header = True
                if line_count > 1 and has_header:
                    return False
    except Exception:
        return True
    return True


def _atomic_write_json(path: str, data: dict[str, Any]) -> None:
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_state(path: str) -> dict[str, Any]:
    if not path:
        return {"version": 1, "files": {}}
    if not os.path.exists(path):
        return {"version": 1, "files": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            return {"version": 1, "files": {}}
        state.setdefault("version", 1)
        state.setdefault("files", {})
        if not isinstance(state["files"], dict):
            state["files"] = {}
        return state
    except Exception as e:
        logging.error(f"Failed to load state file: {path} ({e})")
        return {"version": 1, "files": {}}


def save_state(path: str, state: dict[str, Any]) -> None:
    if not path:
        return
    try:
        _atomic_write_json(path, state)
    except Exception as e:
        logging.error(f"Failed to save state file: {path} ({e})")


def _scan_cache_key(root: str, include_pdf: bool, include_png: bool, include_video: bool, video_exts: set[str]) -> str:
    exts = ",".join(sorted(video_exts)) if video_exts else ""
    return f"root={os.path.normcase(os.path.abspath(root))}|pdf={int(include_pdf)}|png={int(include_png)}|video={int(include_video)}|exts={exts}"


def _load_scan_cache(cache_path: str) -> dict[str, Any] | None:
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        with gzip.open(cache_path, "rt", encoding="utf-8") as f:
            cache = json.load(f)
        if not isinstance(cache, dict):
            return None
        return cache
    except Exception as e:
        logging.warning(f"Failed to load scan cache: {cache_path} ({e})")
        return None


def _save_scan_cache(cache_path: str, cache: dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        tmp = f"{cache_path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, cache_path)
    except Exception as e:
        logging.warning(f"Failed to save scan cache: {cache_path} ({e})")


def tqdm_print(message: str) -> None:
    sink = _GLOBAL_CONSOLE_SINK
    if sink is not None:
        try:
            with _GLOBAL_CONSOLE_LOCK:
                sink.write(str(message) + "\n")
                sink.flush()
                if getattr(_GLOBAL_EVENT_LOG_ARGS, "console_fsync", False):
                    try:
                        os.fsync(sink.fileno())
                    except Exception:
                        pass
        except Exception:
            pass
    try:
        tqdm.write(message)
    except Exception:
        print(message)

def encode_image_to_base64(image: Image.Image) -> str:
    """将 PIL Image 对象转换为 Base64 字符串"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_ollama(
    api_url: str,
    model: str,
    prompt: str,
    image: Image.Image | None = None,
    images: list[Image.Image] | None = None,
    timeout_s: int = 120,
    retries: int = 2,
    retry_backoff_s: float = 2.0,
    print_model_output: bool = True,
    keep_alive: int | str | None = None,
    *,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    temperature: float | None = None,
    trace: str | None = None,
    step: tuple[int, int] | None = None,
    stream: bool | None = None,
) -> str:
    """调用 Ollama API（带超时/重试/健壮 JSON 解析）"""
    options: dict[str, Any] = dict(OLLAMA_REQUEST_OPTIONS or {})
    if num_ctx is not None:
        if int(num_ctx) > MAX_NUM_CTX:
            options["num_ctx"] = MAX_NUM_CTX
        else:
            options["num_ctx"] = int(num_ctx)
    if num_predict is not None:
        options["num_predict"] = int(num_predict)
    if temperature is not None:
        options["temperature"] = float(temperature)

    if images is not None and len(images) > 0 and options.get("num_ctx") == 4096 and num_ctx is None:
        options["num_ctx"] = min(6144, MAX_NUM_CTX)

    use_stream = OLLAMA_STREAM if stream is None else bool(stream)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": use_stream,
    }
    if options:
        payload["options"] = options
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    if images is not None and len(images) > 0:
        payload["images"] = [encode_image_to_base64(img) for img in images]
    elif image is not None:
        payload["images"] = [encode_image_to_base64(image)]

    if options:
        logging.debug(
            "Ollama request: model=%s options=%s images=%s trace=%s step=%s",
            model,
            options,
            (len(images) if images else (1 if image is not None else 0)),
            trace or "",
            f"{step[0]}/{step[1]}" if step else "",
        )

    last_error: str | None = None
    prompt_len = len(prompt or "")
    images_count = 0
    if images is not None:
        images_count = len(images)
    elif image is not None:
        images_count = 1

    for attempt in range(retries + 1):
        attempt_started = time.monotonic()
        model_events = True
        try:
            if _GLOBAL_EVENT_LOG_ARGS is not None:
                model_events = bool(getattr(_GLOBAL_EVENT_LOG_ARGS, "model_events", True))
        except Exception:
            model_events = True
        if model_events:
            _emit_event(
                "model_call_start",
                api_url=api_url,
                model=model,
                attempt=int(attempt + 1),
                attempts_total=int(retries + 1),
                stream=bool(use_stream),
                keep_alive=keep_alive,
                prompt_len=int(prompt_len),
                images_count=int(images_count),
                num_ctx=(payload.get("options") or {}).get("num_ctx") if isinstance(payload.get("options"), dict) else None,
                num_predict=(payload.get("options") or {}).get("num_predict") if isinstance(payload.get("options"), dict) else None,
                temperature=(payload.get("options") or {}).get("temperature") if isinstance(payload.get("options"), dict) else None,
                trace=trace or "",
                step=(f"{step[0]}/{step[1]}" if step else ""),
            )
        try:
            if use_stream:
                resp = requests.post(api_url, json=payload, timeout=timeout_s, stream=True)
                resp.raise_for_status()
                result_parts: list[str] = []
                thinking_parts: list[str] = []
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(chunk, dict):
                        if chunk.get("error"):
                            raise RuntimeError(str(chunk.get("error") or "Ollama error"))
                        r = chunk.get("response")
                        if isinstance(r, str) and r:
                            result_parts.append(r)
                        t = chunk.get("thinking")
                        if isinstance(t, str) and t:
                            thinking_parts.append(t)
                        if chunk.get("done") is True:
                            break
                result = "".join(result_parts)
                if not result.strip():
                    if thinking_parts:
                        logging.warning(
                            "Stream finished with empty response but non-empty thinking. Retrying non-stream once. model=%s trace=%s step=%s",
                            model,
                            trace or "",
                            f"{step[0]}/{step[1]}" if step else "",
                        )
                        try:
                            payload_ns = dict(payload)
                            payload_ns["stream"] = False
                            resp2 = requests.post(api_url, json=payload_ns, timeout=timeout_s)
                            resp2.raise_for_status()
                            data2: Any
                            try:
                                data2 = resp2.json()
                            except Exception:
                                data2 = None
                            if isinstance(data2, dict):
                                result2 = str(data2.get("response", "") or "")
                                if result2.strip():
                                    result = result2
                            if not result.strip():
                                result = "".join(thinking_parts)
                        except Exception:
                            result = "".join(thinking_parts)
                    else:
                        logging.warning(
                            "Stream finished with empty response. model=%s trace=%s step=%s",
                            model,
                            trace or "",
                            f"{step[0]}/{step[1]}" if step else "",
                        )
            else:
                resp = requests.post(api_url, json=payload, timeout=timeout_s)
                resp.raise_for_status()
                data: Any
                result = ""
                try:
                    data = resp.json()
                except Exception as e:
                    data = None
                    raw_text = (resp.text or "").strip()
                    if raw_text:
                        result = raw_text
                    logging.warning(
                        "Non-JSON Ollama response (attempt %d/%d) model=%s trace=%s step=%s err=%s",
                        attempt + 1,
                        retries + 1,
                        model,
                        trace or "",
                        f"{step[0]}/{step[1]}" if step else "",
                        str(e),
                    )

                if isinstance(data, dict):
                    if data.get("error"):
                        result = str(data.get("error") or "")
                    else:
                        result = str(data.get("response", "") or "")

                if not result.strip():
                    raw_preview = (resp.text or "").strip().replace("\r", " ").replace("\n", " ")
                    raw_preview = raw_preview[:300]
                    logging.warning(
                        "Empty Ollama response model=%s trace=%s step=%s preview=%s",
                        model,
                        trace or "",
                        f"{step[0]}/{step[1]}" if step else "",
                        raw_preview,
                    )

            if print_model_output:
                step_str = f" step {step[0]}/{step[1]}" if step else ""
                trace_str = f" | {trace}" if trace else ""
                ctx_str = ""
                if isinstance(payload.get("options"), dict) and payload["options"].get("num_ctx"):
                    ctx_str = f" ctx={payload['options'].get('num_ctx')}"
                tqdm_print(f"\n\n>>> [{model}]{ctx_str}{step_str}{trace_str} Result:\n{result}\n<<<\n")
            if model_events:
                _emit_event(
                    "model_call_done",
                    model=model,
                    attempt=int(attempt + 1),
                    duration_s=round(time.monotonic() - attempt_started, 3),
                    response_len=int(len(result or "")),
                    empty=bool(not str(result or "").strip()),
                    trace=trace or "",
                    step=(f"{step[0]}/{step[1]}" if step else ""),
                )
            return result
        except Exception as e:
            last_error = str(e)
            logging.error(f"Error calling {model} (attempt {attempt + 1}/{retries + 1}): {e}")
            err_lower = last_error.lower()
            is_cuda_err = ("cuda error" in err_lower) or ("cublas" in err_lower) or ("cudnn" in err_lower)
            is_oom = ("out of memory" in err_lower) or ("oom" in err_lower)
            snap: dict[str, Any] | None = None
            try:
                if _GLOBAL_EVENT_LOG_ARGS is not None:
                    snap = _system_snapshot(_GLOBAL_EVENT_LOG_ARGS)
            except Exception:
                snap = None
            if model_events:
                _emit_event(
                    "model_call_error",
                    model=model,
                    attempt=int(attempt + 1),
                    duration_s=round(time.monotonic() - attempt_started, 3),
                    error=last_error,
                    error_type=type(e).__name__,
                    is_cuda=bool(is_cuda_err),
                    is_oom=bool(is_oom),
                    trace=trace or "",
                    step=(f"{step[0]}/{step[1]}" if step else ""),
                    system=snap,
                )
            if attempt < retries:
                if is_cuda_err or is_oom:
                    try:
                        unload_model(api_url, model)
                    except Exception:
                        pass
                    time.sleep(max(retry_backoff_s * (attempt + 1), 3.0))
                else:
                    time.sleep(retry_backoff_s * (attempt + 1))

    raise RuntimeError(f"[Error calling {model}] {last_error or ''}".strip())


def unload_model(api_url: str, model: str) -> None:
    """Best-effort unload to free VRAM (Ollama keep_alive=0)."""
    try:
        requests.post(
            api_url,
            json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            timeout=30,
        )
    except Exception:
        pass


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if not it:
            continue
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _maybe_unload_after_stage(
    args: argparse.Namespace,
    *,
    stages: list[list[str]],
    stage_index: int,
) -> None:
    if not getattr(args, "unload_between_stages", False):
        return

    all_models = {m for stage in stages for m in stage if m}
    if len(all_models) <= 1:
        if not getattr(args, "_unload_skip_single_model_logged", False):
            only = next(iter(all_models), "")
            logging.info(
                "unload_between_stages is enabled but all stages share the same model (%s); skipping unloads to avoid reload churn.",
                only,
            )
            setattr(args, "_unload_skip_single_model_logged", True)
        return

    future_models = {m for stage in stages[stage_index + 1 :] for m in stage if m}
    current_models = _unique_preserve_order(stages[stage_index])
    for m in current_models:
        if m not in future_models:
            unload_model(args.ollama_url, m)

# --- 节点实现 ---

def router_node(args: argparse.Namespace, image: Image.Image) -> str:
    """
    Router: 图像属性判断
    判别：仅含文字 (纯文档) vs 包含图片/插图
    """
    prompt = (
        "Analyze this image. Does it consist primarily of text (like a document page) "
        "or does it contain significant images, illustrations, or charts?\n"
        "Respond with exactly 'DOC' for document/text-heavy or 'VISION' for image/illustration-heavy.\n"
        "Do not output anything else."
    )
    prompt = _apply_thinking(prompt, args.think_qwen_vl)
    votes = max(int(getattr(args, "router_votes", 1) or 1), 1)
    doc_votes = 0
    vision_votes = 0
    for i in range(votes):
        vote_trace = str(getattr(args, "_trace", "") or "")
        if vote_trace:
            vote_trace = f"{vote_trace} [router] [vote {i + 1}/{votes}]"
        response = call_ollama(
            args.ollama_url,
            args.models["router"],
            prompt,
            image=image,
            images=None,
            timeout_s=args.timeout,
            retries=args.retries,
            print_model_output=args.print_model_output,
            keep_alive=args.keep_alive,
            trace=vote_trace,
            step=(1, 3),
        ).strip().upper()
        if "VISION" in response:
            vision_votes += 1
        elif "DOC" in response:
            doc_votes += 1
        else:
            doc_votes += 1

    decision = "VISION" if vision_votes > doc_votes else "DOC"
    logging.info(f"Router decision: {decision} (VISION={vision_votes}, DOC={doc_votes}, votes={votes})")
    return decision


def _program_route_image(image: Image.Image) -> str:
    """
    Best-effort programmatic routing (no LLM call).
    Returns: "DOC" | "VISION" | "AUTO"
    """
    if cv2 is None:
        return "AUTO"
    try:
        import numpy as np  # type: ignore

        rgb = image.convert("RGB")
        # NumPy 2.x compatibility: avoid passing copy=... through np.array(PIL.Image).
        arr = np.asarray(rgb, dtype=np.uint8)
        h, w = arr.shape[:2]
        max_dim = max(h, w)
        if max_dim > 320:
            scale = 320.0 / float(max_dim)
            arr = cv2.resize(arr, (max(int(w * scale), 1), max(int(h * scale), 1)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        mean = float(gray.mean())
        std = float(gray.std())
        white_frac = float((gray >= 230).mean())

        # Heuristic: most documents have a large white background and lower variance.
        if white_frac >= 0.55 and std <= 90.0:
            return "DOC"
        if mean >= 175.0 and std <= 85.0:
            return "DOC"
        return "VISION"
    except Exception:
        return "AUTO"


def process_single_image_fast(args: argparse.Namespace, image: Image.Image, page_num: int = 1) -> str:
    """Fast scan for a single image/page: 1 model call, no router votes, no refine."""
    logging.info(f"[Fast] Processing Page/Image {page_num}...")

    route = _program_route_image(image)
    if route == "DOC":
        model_key = "ocr_qwen"
        prompt = (
            "You are a forensic transcriptionist.\n"
            "Task: transcribe the document page EXACTLY as it appears.\n"
            "Rules:\n"
            "- Do NOT omit anything (numbers, IDs, dates, stamps, signatures, marginal notes, headers/footers).\n"
            "- Preserve line breaks and approximate layout; keep separate blocks separate.\n"
            "- If any content is redacted/black-boxed, write [REDACTED].\n"
            "- If a region is unreadable, write [ILLEGIBLE]. Do NOT guess.\n"
            "- After the verbatim transcription, add a short \"## Structured Fields\" section (derived),\n"
            "  listing any Date/Case ID/Location/Names/Organizations/Doc IDs you can see. Use bullets.\n"
            "Output Markdown only.\n"
        )
    elif route == "VISION":
        model_key = "desc_qwen"
        prompt = (
            "You are an expert visual analyst.\n"
            "Task: produce a detailed, information-dense description.\n"
            "Rules:\n"
            "- Be specific: objects, people, actions, scene context, relative positions, notable details.\n"
            "- If there is visible text, transcribe it verbatim in a separate section.\n"
            "- Do NOT invent facts that are not visible.\n"
            "- Prefer structured output:\n"
            "  ## Detailed Description\n"
            "  ## Visible Text (verbatim)\n"
            "  ## Notable Details / Clues\n"
            "- Aim for 12+ bullet points across sections when possible.\n"
            "Output Markdown only.\n"
        )
    else:
        model_key = "desc_qwen"
        prompt = (
            "You are an expert analyst.\n"
            "Decide whether this looks like a document page or a photo/diagram.\n"
            "Then:\n"
            "- Always include: ## Visible Text (verbatim) (even if empty)\n"
            "- If document-like: provide full verbatim transcription with preserved line breaks.\n"
            "- If photo/diagram-like: provide a detailed description and any transcribed text.\n"
            "Rules:\n"
            "- Do NOT omit visible information.\n"
            "- Do NOT guess unreadable content.\n"
            "- Prefer structured Markdown and be as detailed as the image allows.\n"
        )

    prompt = _apply_thinking(prompt, False)
    result = call_ollama(
        args.ollama_url,
        args.models[model_key],
        prompt,
        image=image,
        images=None,
        timeout_s=args.timeout,
        retries=args.retries,
        print_model_output=args.print_model_output,
        keep_alive=args.keep_alive,
        trace=f"{str(getattr(args, '_trace', '') or '')} [fast] [{model_key}]".strip(),
        step=(1, 1),
    )

    out_route = route if route in ("DOC", "VISION") else "FAST"
    tag = str(getattr(args, "clean_tag", "DOJ_CLEAN_V2") or "DOJ_CLEAN_V2").strip()
    return f"## Page {page_num} ({out_route}) [TAG:{tag}]\n\n{result.strip()}\n\n"

def doc_path_process(args: argparse.Namespace, image: Image.Image) -> str:
    """
    文档处理轨 (交叉验证)
    GLM_OCR_Doc + Qwen_VL_Doc -> Merge
    """
    logging.info("Entering Document Path...")
    
    qwen_prompt = (
        "你是专业文档转录专家。请极其详细地逐行逐字提取图中文字。\n"
        "规则：\n"
        "- 不遗漏任何内容：数字、编号、日期、签名、页眉页脚、批注、印章。\n"
        "- 保留换行与块结构。\n"
        "- 涂黑/遮挡写 [REDACTED]；不可辨认写 [ILLEGIBLE]。\n"
        "- 不要总结，不要改写，只做忠实转录。\n"
        "- 转录后增加“## 结构化字段”小节，列出可见日期/ID/人名/组织/地点。\n"
        "- 输出信息尽可能详尽。\n"
    )
    glm_prompt = (
        "Extract ALL text from this image with maximum fidelity.\n"
        "Rules:\n"
        "- Character-accurate transcription; keep punctuation and line breaks.\n"
        "- Include headers/footers, IDs, signatures, notes, table text.\n"
        "- Redacted => [REDACTED], unreadable => [ILLEGIBLE], do not guess.\n"
        "- No summarization.\n"
    )

    # 串行调用（避免显存争抢；并且便于日志按文件/步骤定位）
    res_glm = call_ollama(
        args.ollama_url,
        args.models["ocr_glm"],
        glm_prompt,
        image=image,
        images=None,
        timeout_s=args.timeout,
        retries=args.retries,
        print_model_output=args.print_model_output,
        keep_alive=args.keep_alive,
        num_ctx=int(getattr(args, "num_ctx_heavy", 8192) or 8192),
        num_predict=max(int(getattr(args, "num_predict_heavy", 2048) or 2048), 4096),
        temperature=float(getattr(args, "temperature_ocr", 0.05) or 0.05),
        trace=f"{str(getattr(args, '_trace', '') or '')} [doc ocr_glm]".strip(),
        step=(2, 3),
    )
    res_qwen = call_ollama(
        args.ollama_url,
        args.models["ocr_qwen"],
        qwen_prompt,
        image=image,
        images=None,
        timeout_s=args.timeout,
        retries=args.retries,
        print_model_output=args.print_model_output,
        keep_alive=args.keep_alive,
        num_ctx=int(getattr(args, "num_ctx_heavy", 8192) or 8192),
        num_predict=max(int(getattr(args, "num_predict_heavy", 2048) or 2048), 4096),
        temperature=float(getattr(args, "temperature_ocr", 0.05) or 0.05),
        trace=f"{str(getattr(args, '_trace', '') or '')} [doc ocr_qwen]".strip(),
        step=(2, 3),
    )
    
    merged_content = f"--- Source: GLM-OCR ---\n{res_glm}\n\n--- Source: Qwen-VL ---\n{res_qwen}"
    return merged_content

def vision_path_process(args: argparse.Namespace, image: Image.Image) -> str:
    """
    视觉处理轨 (视觉与文字混合处理)
    
    已移除 gemma，只使用 qwen3.5:9b 进行描述和交叉验证。
    """
    logging.info("Entering Vision Path...")
    
    # Qwen3.5:9b - 详细描述（主描述）
    desc_qwen_prompt = """
    你是一位专业的视觉分析专家。请对以下图片进行极其详细的、信息密集的描述。
    
    描述规则：
    1. 尽可能详细地描述所有可见元素：人物、物体、动作、场景背景
    2. 描述每个元素的相对位置、大小、颜色、形状等细节
    3. 如果存在文字，必须逐字转录到独立的"## 可见文字（原文）"部分
    4. 不要编造任何图片中看不到的事实
    5. 输出格式（Markdown）：
       - ## 详细描述
         在此部分提供详细描述，每个要点单独成行使用项目符号
       - ## 可见文字（原文）
         在此部分提供逐字转录的文字内容
       - ## 值得注意的细节/线索
    6. 目标：尽可能输出超过 12 个项目点
   
    请开始描述：
    """
    desc_qwen = call_ollama(
        args.ollama_url,
        args.models["desc_qwen"],
        desc_qwen_prompt,
        image=image,
        images=None,
        timeout_s=args.timeout,
        retries=args.retries,
        print_model_output=args.print_model_output,
        keep_alive=args.keep_alive,
        num_ctx=int(getattr(args, "num_ctx_heavy", 8192) or 8192),
        num_predict=int(getattr(args, "num_predict_heavy", 4096) or 4096),
        temperature=float(getattr(args, "temperature_ocr", 0.05) or 0.05),
        trace=f"{str(getattr(args, '_trace', '') or '')} [vision desc_qwen]".strip(),
        step=(2, 3),
    )
    
    # Qwen3.5:9b - 文字检测检查
    has_text_resp = call_ollama(
        args.ollama_url,
        args.models["router"],
        "Is there any visible text in this image? Respond with exactly YES or NO.",
        image=image,
        images=None,
        timeout_s=args.timeout,
        retries=args.retries,
        print_model_output=args.print_model_output,
        keep_alive=args.keep_alive,
        trace=f"{str(getattr(args, '_trace', '') or '')} [vision has_text]".strip(),
        step=(2, 3),
    ).strip().upper()
    
    ocr_tail = ""
    if "YES" in has_text_resp:
        logging.info("Text detected in vision path, running supplementary OCR with glm-ocr...")
        # GLM-OCR - 文字补充提取
        ocr_tail = call_ollama(
            args.ollama_url,
            args.models["ocr_glm"],
            """
            你是一个专业的 OCR 助手。请仔细检查图片中的所有文字，并使用以下规则进行提取：
            
            1. 精确转录所有可见文字
            2. 包括所有数字、ID（如 EFTAxxxxxx）、日期等
            3. 如果某些文字被涂黑/红 action，写入 [REDACTED]
            4. 如果某个区域无法辨认，写入 [ILLEGIBLE]，不要猜测
            5. 注意表格、页眉页脚、小字批注等内容
            
            Begin extraction:
            """,
            image=image,
            images=None,
            timeout_s=args.timeout,
            retries=args.retries,
            print_model_output=args.print_model_output,
            keep_alive=args.keep_alive,
            num_ctx=int(getattr(args, "num_ctx_heavy", 8192) or 8192),
            num_predict=int(getattr(args, "num_predict_heavy", 4096) or 4096),
            temperature=float(getattr(args, "temperature_ocr", 0.05) or 0.05),
            trace=f"{str(getattr(args, '_trace', '') or '')} [vision ocr_tail]".strip(),
            step=(2, 3),
        )
    
    # 使用 qwen3.5:9b 进行二次交叉验证（重新描述以确保一致性）
    desc_qwen_verify = call_ollama(
        args.ollama_url,
        args.models["desc_qwen"],
        f"""
        请根据以下信息，提供一个更全面的视觉描述：
        
        --- 原始描述 ---
        {desc_qwen}
        
        --- 可见文字（OCR）---
        {ocr_tail if ocr_tail else "无 OCR 提取结果"}
        
        请将以上信息合并为一个完整、详细的 Markdown 格式描述。确保不会遗漏任何细节，文本内容要尽可能详细。
        """,
        image=image,
        images=None,
        timeout_s=args.timeout,
        retries=args.retries,
        print_model_output=args.print_model_output,
        keep_alive=args.keep_alive,
        num_ctx=int(getattr(args, "num_ctx_heavy", 8192) or 8192),
        num_predict=int(getattr(args, "num_predict_heavy", 4096) or 4096),
        temperature=float(getattr(args, "temperature_ocr", 0.05) or 0.05),
        trace=f"{str(getattr(args, '_trace', '') or '')} [vision desc_qwen_verify]".strip(),
        step=(2, 3),
    )
    
    merged_content = f"--- Visual Description (Qwen) ---\n{desc_qwen_verify}\n\n"
    if ocr_tail:
        merged_content += f"--- Supplementary Text (OCR from GLM-OCR) ---\n{ocr_tail}"
        
    return merged_content

def refiner_node(
    args: argparse.Namespace,
    content: str,
    *,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    trace: str | None = None,
    step: tuple[int, int] | None = None,
) -> str:
    """
    Refiner: 智能润色与排版
    """
    logging.info("Refining content...")
    prompt = (
        "You are an expert forensic editor.\n"
        "Refine and EXPAND the following text into detailed Markdown.\n"
        "Hard rules:\n"
        "- Do NOT omit any information from the source.\n"
        "- Do NOT drop names, email addresses, dates, IDs, or signature blocks.\n"
        "- Preserve verbatim evidence-like content; only fix obvious OCR mistakes.\n"
        "- Add detailed structure and context labels so final output is substantially fuller (target around 3x detail density).\n"
        "- Never fabricate facts; when uncertain, say '未见明确证据'.\n"
        "Output only Markdown content.\n"
    )
    prompt = _apply_thinking(prompt, args.think_qwen3)
    refine_ctx = int(getattr(args, "num_ctx_refine", 4096) or 4096)
    refine_predict = int(getattr(args, "num_predict_refine", 2048) or 2048)
    refined = call_ollama(
        args.ollama_url,
        args.models["refiner"],
        prompt + "\n\n" + content,
        image=None,
        images=None,
        timeout_s=args.timeout,
        retries=args.retries,
        print_model_output=args.print_model_output,
        keep_alive=args.keep_alive,
        num_ctx=num_ctx if num_ctx is not None else refine_ctx,
        num_predict=num_predict if num_predict is not None else refine_predict,
        trace=trace if trace is not None else f"{str(getattr(args, '_trace', '') or '')} [refine]".strip(),
        step=step if step is not None else (3, 3),
    )
    if not refined.strip():
        logging.warning("Refiner returned empty output, falling back to unrefined content. trace=%s", trace or str(getattr(args, "_trace", "") or ""))
        return content
    try:
        refined = _enforce_3x_detail(args, content, refined, trace=trace, step=step)
        return _append_missing_anchors(content, refined)
    except Exception as e:
        logging.warning("Anchor-preserve failed; returning refined as-is. err=%s trace=%s", str(e), trace or str(getattr(args, "_trace", "") or ""))
        return refined

def process_single_image(
    args: argparse.Namespace,
    image: Image.Image,
    page_num: int = 1,
    *,
    existing_context: str = "",
) -> str:
    """
    处理单张图片 (流程总控)
    """
    if bool(getattr(args, "fast_scan", False)):
        return process_single_image_fast(args, image, page_num)

    logging.info(f"Processing Page/Image {page_num}...")
    
    # 1. Router
    route = router_node(args, image)
    
    # 2. Path Branching
    if route == "VISION":
        raw_result = vision_path_process(args, image)
    else:
        raw_result = doc_path_process(args, image)
        
    # 3. Refiner
    if existing_context.strip() and bool(getattr(args, "enrich_from_output", True)):
        raw_result = (
            raw_result
            + "\n\n--- Existing Output (Historical, same page) ---\n"
            + existing_context.strip()
        )

    final_result = refiner_node(args, raw_result)
    
    tqdm_print("\n##################################################")
    tqdm_print(f"### FINISHED Page {page_num} ({route})")
    tqdm_print(f"##################################################\n{final_result}\n")

    tag = str(getattr(args, "clean_tag", "DOJ_CLEAN_V2") or "DOJ_CLEAN_V2").strip()
    return f"## Page {page_num} ({route}) [TAG:{tag}]\n\n{final_result}\n\n"

# --- 文件处理 ---

def infer_pages_done_from_output(output_path: str) -> int:
    if not os.path.exists(output_path):
        return 0
    done = 0
    try:
        with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("## Page "):
                    done += 1
    except Exception:
        return 0
    return done


def infer_contiguous_pdf_pages_done(output_path: str) -> tuple[int, int]:
    """
    Returns (contiguous_done, total_headers_found).
    Contiguous means pages 1..N all exist in output (by header "## Page X").
    """
    if not os.path.exists(output_path):
        return 0, 0
    pages: set[int] = set()
    found = 0
    try:
        with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("## Page "):
                    continue
                found += 1
                rest = line[len("## Page ") :].strip()
                num_str = ""
                for ch in rest:
                    if ch.isdigit():
                        num_str += ch
                    else:
                        break
                if num_str:
                    try:
                        pages.add(int(num_str))
                    except Exception:
                        pass
    except Exception:
        return 0, 0

    n = 0
    while (n + 1) in pages:
        n += 1
    return n, found


def infer_contiguous_video_frames_done(output_path: str) -> tuple[int, int]:
    """
    Returns (contiguous_done, total_headers_found) for headers "## Frame N".
    """
    if not os.path.exists(output_path):
        return 0, 0
    frames: set[int] = set()
    found = 0
    try:
        with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("## Frame "):
                    continue
                found += 1
                rest = line[len("## Frame ") :].strip()
                num_str = ""
                for ch in rest:
                    if ch.isdigit():
                        num_str += ch
                    else:
                        break
                if num_str:
                    try:
                        frames.add(int(num_str))
                    except Exception:
                        pass
    except Exception:
        return 0, 0

    n = 0
    while (n + 1) in frames:
        n += 1
    return n, found


def process_png_file(args: argparse.Namespace, file_path: str, output_path: str) -> int:
    with Image.open(file_path) as image:
        args._trace = _trace_for_file(file_path, kind="png", page_num=1, pages_total=1)  # type: ignore[attr-defined]
        existing_context = ""
        if bool(getattr(args, "enrich_from_output", True)) and os.path.exists(output_path) and not args.overwrite:
            existing_context = _extract_existing_page_context(output_path, 1)
            if not existing_context.strip():
                existing_context = _read_text_file(output_path)
        page_result = process_single_image(args, image, 1, existing_context=existing_context)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(_file_meta_marker(args, file_path=file_path, file_type="png"))
        f.write(page_result)
        f.flush()
        os.fsync(f.fileno())
    return 1


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) not available. Please install opencv-python.")


def extract_video_keyframes(
    video_path: str,
    interval_sec: float,
    max_frames: int = 0,
) -> list[tuple[float, Image.Image]]:
    """抽取关键帧（按固定时间间隔）。返回 [(timestamp_sec, PIL.Image), ...]"""
    _require_cv2()
    assert cv2 is not None

    # 检查文件是否存在且可读
    if not os.path.exists(video_path):
        raise RuntimeError(f"Video file not found: {video_path}")
    if os.path.getsize(video_path) == 0:
        raise RuntimeError(f"Video file is empty: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        fps = 25.0

    step_frames = max(int(round(fps * max(interval_sec, 0.01))), 1)

    frames: list[tuple[float, Image.Image]] = []
    next_frame = 0
    extracted = 0

    while True:
        if frame_count > 0 and next_frame >= frame_count:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(next_frame))
        ret, frame = cap.read()
        if not ret:
            break

        ts = float(next_frame) / float(fps)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        frames.append((ts, img))

        extracted += 1
        if max_frames and extracted >= max_frames:
            break

        next_frame += step_frames

    cap.release()
    return frames


def _has_text_in_frame(args: argparse.Namespace, image: Image.Image) -> bool:
    prompt = "Is there any visible text in this image? Respond with exactly YES or NO."
    prompt = _apply_thinking(prompt, args.think_qwen_vl)
    resp = call_ollama(
        args.ollama_url,
        args.models["router"],
        prompt,
        image=image,
        images=None,
        timeout_s=args.timeout,
        retries=args.retries,
        print_model_output=args.print_model_output,
        keep_alive=args.keep_alive,
        trace=str(getattr(args, "_trace", "") or ""),
        step=(2, 5),
    ).strip().upper()
    return "YES" in resp


def _qwen_file_batch_stage(args: argparse.Namespace, items: list[dict[str, Any]]) -> None:
    """qwen stage for PNG/PDF items: route + qwen outputs for each item."""
    for it in items:
        img: Image.Image = it["image"]
        src_path = str(it.get("src_path") or "")
        page_num = int(it.get("page_num") or 1)
        pages_total = int(it.get("pages_total") or 0)
        kind = str(it.get("type") or "")
        trace = _trace_for_file(src_path, kind=kind, page_num=page_num, pages_total=(pages_total if pages_total > 0 else None))

        # Router
        route_prompt = (
            "Analyze this image. Does it consist primarily of text (like a document page) "
            "or does it contain significant images, illustrations, or charts?\n"
            "Respond with exactly 'DOC' for document/text-heavy or 'VISION' for image/illustration-heavy.\n"
            "Do not output anything else."
        )
        route_prompt = _apply_thinking(route_prompt, args.think_qwen_vl)
        route = call_ollama(
            args.ollama_url,
            args.models["router"],
            route_prompt,
            image=img,
            timeout_s=args.timeout,
            retries=args.retries,
            print_model_output=args.print_model_output,
            keep_alive=args.keep_alive,
            trace=trace,
            step=(1, 4),
        ).strip().upper()
        it["route"] = "VISION" if "VISION" in route else "DOC"

        if it["route"] == "DOC":
            qwen_ocr = call_ollama(
                args.ollama_url,
                args.models["ocr_qwen"],
                _apply_thinking(
                    "Transcribe this document image in very high detail. Keep all lines, IDs, dates, names, signatures, headers/footers, and notes. "
                    "Use [REDACTED] for redacted text and [ILLEGIBLE] for unreadable text. No summarization.",
                    args.think_qwen_vl,
                ),
                image=img,
                timeout_s=args.timeout,
                retries=args.retries,
                print_model_output=args.print_model_output,
                keep_alive=args.keep_alive,
                num_ctx=int(getattr(args, "num_ctx_heavy", 8192) or 8192),
                num_predict=max(int(getattr(args, "num_predict_heavy", 2048) or 2048), 4096),
                temperature=float(getattr(args, "temperature_ocr", 0.05) or 0.05),
                trace=trace,
                step=(1, 3),
            )
            it["qwen_ocr"] = qwen_ocr
        else:
            # qwen description + text check
            qwen_desc = call_ollama(
                args.ollama_url,
                args.models["desc_qwen"],
                _apply_thinking(
                    "Describe this image in extreme detail with structured Markdown. Include object relations, scene layout, actions, and clues. "
                    "If visible text exists, transcribe it verbatim in a dedicated section.",
                    args.think_qwen_vl,
                ),
                image=img,
                timeout_s=args.timeout,
                retries=args.retries,
                print_model_output=args.print_model_output,
                keep_alive=args.keep_alive,
                num_ctx=int(getattr(args, "num_ctx_heavy", 8192) or 8192),
                num_predict=max(int(getattr(args, "num_predict_heavy", 2048) or 2048), 4096),
                temperature=float(getattr(args, "temperature_ocr", 0.05) or 0.05),
                trace=trace,
                step=(1, 3),
            )
            it["qwen_desc"] = qwen_desc

            has_text = call_ollama(
                args.ollama_url,
                args.models["router"],
                _apply_thinking("Is there any visible text in this image? Respond with YES or NO.", args.think_qwen_vl),
                image=img,
                timeout_s=args.timeout,
                retries=args.retries,
                print_model_output=args.print_model_output,
                keep_alive=args.keep_alive,
                trace=trace,
                step=(1, 3),
            ).strip().upper()
            it["has_text"] = "YES" in has_text


def _glm_file_batch_stage(args: argparse.Namespace, items: list[dict[str, Any]]) -> None:
    for it in items:
        img: Image.Image = it["image"]
        src_path = str(it.get("src_path") or "")
        page_num = int(it.get("page_num") or 1)
        pages_total = int(it.get("pages_total") or 0)
        kind = str(it.get("type") or "")
        trace = _trace_for_file(src_path, kind=kind, page_num=page_num, pages_total=(pages_total if pages_total > 0 else None))
        route = it.get("route")
        if route == "DOC":
            it["glm_ocr"] = call_ollama(
                args.ollama_url,
                args.models["ocr_glm"],
                "Extract all text from this image exactly as it appears.",
                image=img,
                timeout_s=args.timeout,
                retries=args.retries,
                print_model_output=args.print_model_output,
                keep_alive=args.keep_alive,
                num_ctx=int(getattr(args, "num_ctx_heavy", 8192) or 8192),
                num_predict=max(int(getattr(args, "num_predict_heavy", 2048) or 2048), 4096),
                temperature=float(getattr(args, "temperature_ocr", 0.05) or 0.05),
                trace=trace,
                step=(2, 3),
            )
        elif route == "VISION" and it.get("has_text"):
            it["glm_ocr_tail"] = call_ollama(
                args.ollama_url,
                args.models["ocr_glm"],
                "Extract all text found in this image.",
                image=img,
                timeout_s=args.timeout,
                retries=args.retries,
                print_model_output=args.print_model_output,
                keep_alive=args.keep_alive,
                num_ctx=int(getattr(args, "num_ctx_heavy", 8192) or 8192),
                num_predict=max(int(getattr(args, "num_predict_heavy", 2048) or 2048), 4096),
                temperature=float(getattr(args, "temperature_ocr", 0.05) or 0.05),
                trace=trace,
                step=(2, 3),
            )


def _qwen3_refine_file_batch_stage(args: argparse.Namespace, items: list[dict[str, Any]]) -> None:
    for it in items:
        src_path = str(it.get("src_path") or "")
        page_num = int(it.get("page_num") or 1)
        pages_total = int(it.get("pages_total") or 0)
        kind = str(it.get("type") or "")
        trace = _trace_for_file(
            src_path,
            kind=kind,
            page_num=page_num,
            pages_total=(pages_total if pages_total > 0 else None),
        )

        route = it.get("route")
        if route == "DOC":
            merged = (
                f"--- Source: Qwen3.5 OCR ---\n{it.get('qwen_ocr','')}\n\n"
                f"--- Source: GLM OCR (Cross-check) ---\n{it.get('glm_ocr','')}"
            )
        else:
            merged = f"--- Visual Description (Qwen3.5) ---\n{it.get('qwen_desc','')}"
            if it.get("glm_ocr_tail"):
                merged += f"\n\n--- Supplementary Text (GLM OCR) ---\n{it.get('glm_ocr_tail','')}"
        existing_context = str(it.get("existing_context") or "").strip()
        if existing_context and bool(getattr(args, "enrich_from_output", True)):
            merged += "\n\n--- Existing Output (Historical, same page) ---\n" + existing_context

        refined = refiner_node(args, merged, trace=trace, step=(3, 3))
        it["refined"] = refined


def _write_item_output_and_update_state(
    args: argparse.Namespace,
    it: dict[str, Any],
    state_path: str,
    state: dict[str, Any],
) -> None:
    """Write output for one PNG/PDF page item and update state."""
    src_path = it["src_path"]
    file_key = _norm_key(src_path)
    entry: dict[str, Any] = state["files"].setdefault(file_key, {})
    output_path = it["output_path"]
    tag = str(getattr(args, "clean_tag", "DOJ_CLEAN_V2") or "DOJ_CLEAN_V2").strip()

    if it["type"] == "png":
        content = f"## Page 1 ({it.get('route','DOC')}) [TAG:{tag}]\n\n{it.get('refined','')}\n\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(_file_meta_marker(args, file_path=src_path, file_type="png"))
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        entry["pages_total"] = 1
        entry["pages_done"] = 1
        entry["status"] = "done"
        entry["updated_at"] = _now_iso()
        save_state(state_path, state)
        return

    # PDF page
    page_num = int(it["page_num"])
    content = f"## Page {page_num} ({it.get('route','DOC')}) [TAG:{tag}]\n\n{it.get('refined','')}\n\n"
    mode = "a" if (args.resume and page_num > 1 and os.path.exists(output_path) and not args.overwrite) else "w"
    if args.overwrite and page_num == 1 and os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass
    with open(output_path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(_file_meta_marker(args, file_path=src_path, file_type="pdf"))
        f.write(content)
        f.flush()

    entry["pages_total"] = it.get("pages_total")
    entry["pages_done"] = max(int(entry.get("pages_done", 0) or 0), page_num)
    entry["updated_at"] = _now_iso()
    pages_total_i = int(entry.get("pages_total") or 0)
    if pages_total_i and entry["pages_done"] >= pages_total_i:
        entry["status"] = "done"
    else:
        entry["status"] = "in_progress"
    save_state(state_path, state)


def process_pdf_png_batched(
    args: argparse.Namespace,
    pdf_png_targets: list[str],
    state_path: str,
    state: dict[str, Any],
    pbar: Any,
) -> tuple[int, int, int]:
    """Batch scheduling for PNG/PDF: rotate models per batch. Returns (pdf_done, png_done, images_done)."""
    poppler_path = cast(Any, args.poppler_path or None)
    pdf_dpi = int(getattr(args, "pdf_dpi", 300) or 300)

    pdf_done = 0
    png_done = 0
    images_done = 0
    adjusted_pdf_keys: set[str] = set()
    pdfinfo_failed_keys: set[str] = set()

    # Helper to find next pending item
    def next_item() -> dict[str, Any] | None:
        for fp in pdf_png_targets:
            lower = fp.lower()
            key = _norm_key(fp)
            entry: dict[str, Any] = state["files"].setdefault(key, {})
            output_path = _resolve_output_path(fp, args.output_suffix)
            entry.setdefault("path", fp)
            entry["output"] = output_path
            entry["type"] = "pdf" if lower.endswith(".pdf") else "png"

            clean_tag = str(getattr(args, "clean_tag", "DOJ_CLEAN_V2") or "DOJ_CLEAN_V2").strip()
            already_tagged = _output_has_clean_tag(str(output_path), clean_tag)
            truncated = _output_is_truncated(str(output_path), entry.get("type", "pdf"))
            if args.resume and truncated and os.path.exists(output_path):
                logging.warning("Detected truncated output in batch mode, rebuilding: %s", output_path)
                try:
                    os.remove(output_path)
                except Exception:
                    pass
                if lower.endswith(".png"):
                    entry["pages_done"] = 0
                else:
                    entry["pages_done"] = 0
                entry["status"] = "in_progress"
            if entry.get("status") == "done" and os.path.exists(output_path) and not args.overwrite and already_tagged and not truncated:
                continue

            if lower.endswith(".png"):
                # single image
                existing_context = ""
                if bool(getattr(args, "enrich_from_output", True)) and os.path.exists(output_path) and not args.overwrite:
                    existing_context = _extract_existing_page_context(output_path, 1)
                    if not existing_context.strip():
                        existing_context = _read_text_file(output_path)
                return {
                    "type": "png",
                    "src_path": fp,
                    "output_path": output_path,
                    "page_num": 1,
                    "pages_total": 1,
                    "image": Image.open(fp),
                    "existing_context": existing_context,
                }

            # pdf: next page
            pages_done = int(entry.get("pages_done", 0) or 0)
            if args.overwrite or not args.resume:
                pages_done = 0
                entry["pages_done"] = 0

            assumed_total_pages = int(entry.get("pages_total", 0) or 0)
            total_pages = assumed_total_pages
            if total_pages <= 0 and key not in pdfinfo_failed_keys:
                try:
                    info = pdfinfo_from_path(fp, poppler_path=poppler_path)
                    total_pages = int(info.get("Pages", 0))
                    entry["pages_total"] = total_pages

                    assumed_for_progress = assumed_total_pages if assumed_total_pages > 0 else 1
                    assumed_remaining = max(assumed_for_progress - pages_done, 0)
                    actual_remaining = max(total_pages - pages_done, 0)
                    delta = actual_remaining - assumed_remaining
                    if delta > 0 and key not in adjusted_pdf_keys and getattr(pbar, "total", None) is not None:
                        try:
                            pbar.total = int(pbar.total) + int(delta)
                            pbar.refresh()
                            adjusted_pdf_keys.add(key)
                            logging.info("[Progress] Increased total units by %d after pdfinfo: %s", int(delta), fp)
                        except Exception:
                            pass
                except Exception:
                    pdfinfo_failed_keys.add(key)
                    total_pages = 0

            next_page = pages_done + 1
            if total_pages and next_page > total_pages:
                entry["status"] = "done"
                entry["updated_at"] = _now_iso()
                save_state(state_path, state)
                continue

            try:
                images = convert_from_path(
                    fp,
                    first_page=next_page,
                    last_page=next_page,
                    poppler_path=poppler_path,
                    dpi=pdf_dpi,
                )
                if not images:
                    if total_pages <= 0 and pages_done > 0:
                        entry["pages_total"] = pages_done
                        entry["status"] = "done"
                        entry["updated_at"] = _now_iso()
                        save_state(state_path, state)
                        continue
                    raise RuntimeError("convert_from_path returned no images")
                return {
                    "type": "pdf",
                    "src_path": fp,
                    "output_path": output_path,
                    "page_num": next_page,
                    "pages_total": total_pages,
                    "image": images[0],
                    "existing_context": (
                        _extract_existing_page_context(output_path, next_page)
                        if bool(getattr(args, "enrich_from_output", True)) and os.path.exists(output_path) and not args.overwrite
                        else ""
                    ),
                }
            except Exception as e:
                entry["status"] = "failed"
                entry["last_error"] = str(e)
                entry["updated_at"] = _now_iso()
                save_state(state_path, state)
                tqdm_print(f"[PDF] Failed to convert page {next_page}: {fp} ({e})")
                continue

        return None

    while True:
        batch: list[dict[str, Any]] = []
        for _ in range(max(int(args.file_batch_size), 1)):
            it = next_item()
            if it is None:
                break
            batch.append(it)

        if not batch:
            break

        stages = [
            [args.models["router"], args.models["ocr_qwen"], args.models["desc_qwen"]],
            [args.models["ocr_glm"]],
            [args.models["refiner"]],
        ]

        # Stage 1: Qwen3-VL for the whole batch
        tqdm_print(f"\n[Batch] qwen stage for {len(batch)} items")
        _qwen_file_batch_stage(args, batch)
        _maybe_unload_after_stage(args, stages=stages, stage_index=0)

        # Stage 2: GLM-OCR
        tqdm_print(f"[Batch] glm-ocr stage")
        _glm_file_batch_stage(args, batch)
        _maybe_unload_after_stage(args, stages=stages, stage_index=1)

        # Stage 3: Qwen3 refine
        tqdm_print(f"[Batch] qwen3 refine stage")
        _qwen3_refine_file_batch_stage(args, batch)
        _maybe_unload_after_stage(args, stages=stages, stage_index=2)

        # Write outputs + update state + progress
        for it in batch:
            try:
                _write_item_output_and_update_state(args, it, state_path, state)
                images_done += 1
                pbar.update(1)
            finally:
                try:
                    it["image"].close()
                except Exception:
                    pass

        # Update done file counts opportunistically
        for fp in pdf_png_targets:
            key = _norm_key(fp)
            entry = state.get("files", {}).get(key, {})
            outp = entry.get("output")
            if entry.get("status") == "done" and outp and os.path.exists(outp):
                if entry.get("type") == "pdf":
                    pass
                else:
                    pass

    # Final counts
    for fp in pdf_png_targets:
        key = _norm_key(fp)
        entry = state.get("files", {}).get(key, {})
        outp = entry.get("output")
        if entry.get("status") == "done" and outp and os.path.exists(outp):
            if entry.get("type") == "pdf":
                pdf_done += 1
            elif entry.get("type") == "png":
                png_done += 1

    return pdf_done, png_done, images_done


def process_video_file(
    args: argparse.Namespace,
    file_path: str,
    output_path: str,
    state_entry: dict[str, Any],
    state_path: str,
    state: dict[str, Any],
) -> int:
    """视频：逐帧处理（不做多帧 batch），支持断点续跑。返回本次处理的帧数。"""

    if bool(getattr(args, "fast_scan", False)):
        return _process_video_file_fast(args, file_path, output_path, state_entry, state_path, state)

    if args.overwrite:
        state_entry["frames_done"] = 0
        state_entry["status"] = "in_progress"

    base_trace = _trace_for_file(file_path, kind="video")
    args._trace = base_trace  # type: ignore[attr-defined]

    heavy_ctx = int(getattr(args, "num_ctx_heavy", 8192) or 8192)
    heavy_predict = int(getattr(args, "num_predict_heavy", 2048) or 2048)
    if heavy_ctx > MAX_NUM_CTX:
        heavy_ctx = MAX_NUM_CTX

    frames = extract_video_keyframes(
        file_path,
        interval_sec=args.video_interval,
        max_frames=args.video_max_frames,
    )
    state_entry["frames_total"] = len(frames)
    state_entry["video_interval"] = args.video_interval

    frames_done = int(state_entry.get("frames_done", 0) or 0)
    if args.resume and os.path.exists(output_path) and not args.overwrite:
        if _output_is_truncated(output_path, "video"):
            logging.warning("Video output looks truncated; restarting output: %s", output_path)
            try:
                os.remove(output_path)
            except Exception:
                pass
            frames_done = 0
            state_entry["frames_done"] = 0
        contiguous_done, found = infer_contiguous_video_frames_done(output_path)
        if found and contiguous_done == 0:
            logging.warning("Video output has frame headers but missing Frame 1; restarting video output: %s", output_path)
            try:
                os.remove(output_path)
            except Exception:
                pass
            contiguous_done = 0
        if contiguous_done != frames_done:
            logging.info("Video resume sync: state frames_done=%d, output contiguous_done=%d (%s)", frames_done, contiguous_done, output_path)
            frames_done = contiguous_done
            state_entry["frames_done"] = frames_done

    if args.overwrite or not args.resume:
        frames_done = 0
        state_entry["frames_done"] = 0

    if frames_done >= len(frames) and len(frames) > 0 and not args.overwrite:
        if os.path.exists(output_path):
            state_entry["status"] = "done"
            save_state(state_path, state)
            return 0
        # State says done but output is missing; rebuild output from scratch.
        frames_done = 0
        state_entry["frames_done"] = 0

    mode = "a" if (args.resume and frames_done > 0 and os.path.exists(output_path) and not args.overwrite) else "w"
    processed = 0
    existing_frame_sections: dict[int, str] = {}
    existing_final_report = ""
    if bool(getattr(args, "enrich_from_output", True)) and os.path.exists(output_path) and not args.overwrite:
        existing_text = _read_text_file(output_path)
        existing_frame_sections = _extract_numbered_sections(existing_text, "Frame")
        m_final = re.search(r"(?ms)^#\s*Final Report\s*\n+(.*)$", existing_text)
        if m_final:
            existing_final_report = str(m_final.group(1) or "").strip()
    tag = str(getattr(args, "clean_tag", "DOJ_CLEAN_V2") or "DOJ_CLEAN_V2").strip()

    remaining = frames[frames_done:]
    video_pbar = None
    if args.show_video_progress:
        video_pbar = tqdm(total=len(remaining), desc="Video frames", unit="frame", leave=False)

    frame_notes: list[str] = []
    with open(output_path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(_file_meta_marker(args, file_path=file_path, file_type="video"))
        for idx, (ts, img) in enumerate(remaining, start=frames_done + 1):
            frame_trace = f"{base_trace} [frame {idx}/{len(frames)} ts={ts:.1f}s]"
            args._trace = frame_trace  # type: ignore[attr-defined]

            prior_frame_context = existing_frame_sections.get(int(idx), "")
            desc_prompt = (
                "Describe this video frame in detail. Mention visible actions, objects, and scene context.\n"
                "Output with rich structure and as much concrete detail as possible."
            )
            if prior_frame_context.strip():
                desc_prompt += (
                    "\n\nHistorical notes for the same frame are provided below. "
                    "Preserve facts, correct mistakes, and expand details instead of repeating blindly.\n\n"
                    + prior_frame_context
                )
            desc_prompt = _apply_thinking(desc_prompt, args.think_qwen_vl)
            qwen_desc = call_ollama(
                args.ollama_url,
                args.models["desc_qwen"],
                desc_prompt,
                image=img,
                timeout_s=args.timeout,
                retries=args.retries,
                print_model_output=args.print_model_output,
                keep_alive=args.keep_alive,
                num_ctx=heavy_ctx,
                num_predict=heavy_predict,
                trace=frame_trace,
                step=(1, 3),
            )

            ocr_text = ""
            if args.video_ocr_mode != "never":
                do_ocr = args.video_ocr_mode == "always"
                if args.video_ocr_mode == "auto":
                    do_ocr = _has_text_in_frame(args, img)
                if do_ocr:
                    ocr_text = call_ollama(
                        args.ollama_url,
                        args.models["ocr_glm"],
                        f"Extract all readable text from this frame. Timestamp: {ts:.1f}s",
                        image=img,
                        timeout_s=args.timeout,
                        retries=args.retries,
                        print_model_output=args.print_model_output,
                        keep_alive=args.keep_alive,
                        num_ctx=heavy_ctx,
                        num_predict=heavy_predict,
                        temperature=float(getattr(args, "temperature_ocr", 0.1) or 0.1),
                        trace=frame_trace,
                        step=(2, 3),
                    )

            f.write(f"## Frame {idx} ({ts:.1f}s) [TAG:{tag}]\n\n")
            f.write(qwen_desc.strip() + "\n\n")
            if ocr_text.strip():
                f.write("### OCR\n\n")
                f.write(ocr_text.strip() + "\n\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

            note_line = qwen_desc.strip().replace("\r", " ").replace("\n", " ")
            note_line = note_line[:400]
            frame_notes.append(f"- {ts:.1f}s: {note_line}")
            if ocr_text.strip():
                ocr_one = ocr_text.strip().replace("\r", " ").replace("\n", " ")
                ocr_one = ocr_one[:300]
                frame_notes.append(f"  - OCR: {ocr_one}")

            processed += 1
            if video_pbar is not None:
                video_pbar.update(1)

            state_entry["frames_done"] = idx
            state_entry["updated_at"] = _now_iso()
            state_entry["status"] = "in_progress"
            save_state(state_path, state)

        raw = "\n".join(frame_notes)

    if video_pbar is not None:
        video_pbar.close()

    final_prompt = (
        "你是一个视频分析专家。下面是逐帧摘要（含时间戳，可能包含 OCR 文字）。\n"
        "请输出：\n"
        "1) 按时间轴组织的动作/事件摘要（尽量给出秒级范围）；\n"
        "2) 把 OCR 文字按时间点归并；\n"
        "3) 最后给出整体结论与可能的场景描述。\n"
        "4) 信息密度要求提高，细节量接近历史版本的 3 倍（若历史版本存在）。\n"
        "仅输出最终报告（Markdown）。\n\n"
        + raw
    )
    if existing_final_report:
        final_prompt += "\n\n--- Historical Final Report (for enrichment) ---\n" + existing_final_report
    final_prompt = _apply_thinking(final_prompt, args.think_qwen3)
    final_report = call_ollama(
        args.ollama_url,
        args.models["refiner"],
        final_prompt,
        timeout_s=args.timeout,
        retries=args.retries,
        print_model_output=args.print_model_output,
        keep_alive=args.keep_alive,
        num_ctx=heavy_ctx,
        num_predict=heavy_predict,
        trace=f"{base_trace} [final report]",
        step=(3, 3),
    )

    with open(output_path, "a", encoding="utf-8") as wf:
        wf.write("# Final Report\n\n")
        wf.write(final_report)
        wf.write("\n")
        wf.flush()
        try:
            os.fsync(wf.fileno())
        except Exception:
            pass

    state_entry["status"] = "done"
    state_entry["updated_at"] = _now_iso()
    save_state(state_path, state)
    return processed


def _parse_fast_video_frames(text: str, k: int) -> dict[int, str]:
    parts: dict[int, str] = {}
    if not text:
        return {i: "" for i in range(1, k + 1)}
    matches = list(re.finditer(r"\[\[FRAME\s*(\d+)\]\]", text))
    if not matches:
        parts[1] = text.strip()
        for i in range(2, k + 1):
            parts[i] = ""
        return parts
    for idx, m in enumerate(matches):
        try:
            num = int(m.group(1))
        except Exception:
            continue
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        parts[num] = text[start:end].strip()
    for i in range(1, k + 1):
        parts.setdefault(i, "")
    return parts


def _process_video_file_fast(
    args: argparse.Namespace,
    file_path: str,
    output_path: str,
    state_entry: dict[str, Any],
    state_path: str,
    state: dict[str, Any],
) -> int:
    """Fast scan for videos: process frames in small multi-image batches, no per-frame OCR, no final report."""

    if args.overwrite:
        state_entry["frames_done"] = 0
        state_entry["status"] = "in_progress"

    base_trace = _trace_for_file(file_path, kind="video")
    args._trace = base_trace  # type: ignore[attr-defined]

    frames = extract_video_keyframes(
        file_path,
        interval_sec=args.video_interval,
        max_frames=args.video_max_frames,
    )
    state_entry["frames_total"] = len(frames)
    state_entry["video_interval"] = args.video_interval

    frames_done = int(state_entry.get("frames_done", 0) or 0)
    if args.resume and os.path.exists(output_path) and not args.overwrite:
        if _output_is_truncated(output_path, "video"):
            logging.warning("Video output looks truncated; restarting output: %s", output_path)
            try:
                os.remove(output_path)
            except Exception:
                pass
            frames_done = 0
            state_entry["frames_done"] = 0
        contiguous_done, found = infer_contiguous_video_frames_done(output_path)
        if found and contiguous_done == 0:
            logging.warning("Video output has frame headers but missing Frame 1; restarting video output: %s", output_path)
            try:
                os.remove(output_path)
            except Exception:
                pass
            contiguous_done = 0
        if contiguous_done != frames_done:
            logging.info("Video resume sync: state frames_done=%d, output contiguous_done=%d (%s)", frames_done, contiguous_done, output_path)
            frames_done = contiguous_done
            state_entry["frames_done"] = frames_done

    if args.overwrite or not args.resume:
        frames_done = 0
        state_entry["frames_done"] = 0

    if frames_done >= len(frames) and len(frames) > 0 and not args.overwrite:
        if os.path.exists(output_path):
            state_entry["status"] = "done"
            save_state(state_path, state)
            return 0
        # State says done but output is missing; rebuild output from scratch.
        frames_done = 0
        state_entry["frames_done"] = 0

    mode = "a" if (args.resume and frames_done > 0 and os.path.exists(output_path) and not args.overwrite) else "w"
    processed = 0
    existing_frame_sections: dict[int, str] = {}
    if bool(getattr(args, "enrich_from_output", True)) and os.path.exists(output_path) and not args.overwrite:
        existing_frame_sections = _extract_numbered_sections(_read_text_file(output_path), "Frame")
    tag = str(getattr(args, "clean_tag", "DOJ_CLEAN_V2") or "DOJ_CLEAN_V2").strip()

    batch_size = int(getattr(args, "fast_video_batch_size", 2) or 2)
    if batch_size <= 0:
        batch_size = 2

    remaining = frames[frames_done:]
    video_pbar = None
    if args.show_video_progress:
        video_pbar = tqdm(total=len(remaining), desc="Video frames", unit="frame", leave=False)

    with open(output_path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(_file_meta_marker(args, file_path=file_path, file_type="video"))
        i = 0
        total = len(frames)
        while i < len(remaining):
            chunk = remaining[i : i + batch_size]
            idx_start = frames_done + i + 1
            imgs = [img for _, img in chunk]
            ts_list = [ts for ts, _ in chunk]
            frame_trace = f"{base_trace} [frames {idx_start}-{idx_start + len(chunk) - 1}/{total}]"
            args._trace = frame_trace  # type: ignore[attr-defined]

            prompt_lines = [
                "You will be given video frames in chronological order.",
                "For each frame, output a section starting with exactly [[FRAME N]] on its own line (N starts from 1).",
                "Under each section, use Markdown with these subsections:",
                "- ### Detailed Description (be specific and thorough)",
                "- ### Visible Text (verbatim) (write [NONE] if no text)",
                "- ### Notable Changes (vs previous frame, if applicable)",
                "Rules:",
                "- Do NOT invent facts that are not visible.",
                "- Transcribe text exactly; if unreadable write [ILLEGIBLE].",
                "- Aim for 8+ bullet points per frame when possible.",
                "Output ONLY these [[FRAME N]] sections; do not add any extra text outside them.",
            ]
            for j, ts in enumerate(ts_list, start=1):
                prompt_lines.append(f"FRAME {j} timestamp: {ts:.1f}s")
                if bool(getattr(args, "enrich_from_output", True)):
                    frame_idx = idx_start + (j - 1)
                    hist = existing_frame_sections.get(frame_idx, "").strip()
                    if hist:
                        prompt_lines.append(
                            f"FRAME {j} historical notes (enrich, preserve facts, expand details): {hist[:1600]}"
                        )
            prompt = "\n".join(prompt_lines)

            resp = call_ollama(
                args.ollama_url,
                args.models["desc_qwen"],
                prompt,
                image=None,
                images=imgs,
                timeout_s=args.timeout,
                retries=args.retries,
                print_model_output=args.print_model_output,
                keep_alive=args.keep_alive,
                trace=f"{frame_trace} [fast video]",
                step=(1, 1),
            )
            parts = _parse_fast_video_frames(resp, len(chunk))

            for j, ts in enumerate(ts_list, start=1):
                frame_idx = idx_start + (j - 1)
                f.write(f"## Frame {frame_idx} ({ts:.1f}s) [TAG:{tag}]\n\n")
                body = parts.get(j, "").strip()
                if not body:
                    body = resp.strip()
                f.write(body + "\n\n")
                processed += 1
                if video_pbar is not None:
                    video_pbar.update(1)

            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

            state_entry["frames_done"] = idx_start + len(chunk) - 1
            state_entry["updated_at"] = _now_iso()
            state_entry["status"] = "in_progress"
            save_state(state_path, state)

            i += len(chunk)

    if video_pbar is not None:
        video_pbar.close()

    state_entry["status"] = "done"
    state_entry["updated_at"] = _now_iso()
    save_state(state_path, state)
    return processed


def process_pdf_file(
    args: argparse.Namespace,
    file_path: str,
    output_path: str,
    state_entry: dict[str, Any],
    state_path: str,
    state: dict[str, Any],
) -> int:
    """按页转换 PDF 并逐页处理，支持断点续跑。返回处理的图片页数。"""
    poppler_path = cast(Any, args.poppler_path or None)
    pdf_dpi = int(getattr(args, "pdf_dpi", 300) or 300)

    # 检查文件是否为空
    if os.path.getsize(file_path) == 0:
        logging.warning(f"Skipping empty PDF file: {file_path}")
        state_entry["status"] = "skipped"
        state_entry["last_error"] = "Empty PDF file"
        save_state(state_path, state)
        return 0

    try:
        info = pdfinfo_from_path(file_path, poppler_path=poppler_path)
        total_pages = int(info.get("Pages", 0))
    except Exception as e:
        raise RuntimeError(f"pdfinfo failed: {e}")

    if args.overwrite:
        state_entry["pages_done"] = 0
    state_entry.setdefault("pages_total", total_pages)

    pages_done = int(state_entry.get("pages_done", 0) or 0)
    if args.resume and os.path.exists(output_path) and not args.overwrite:
        if _output_is_truncated(output_path, "pdf"):
            logging.warning("PDF output looks truncated; restarting output: %s", output_path)
            try:
                os.remove(output_path)
            except Exception:
                pass
            pages_done = 0
            state_entry["pages_done"] = 0
        contiguous_done, found = infer_contiguous_pdf_pages_done(output_path)
        if found and contiguous_done == 0:
            logging.warning("PDF output has page headers but missing Page 1; restarting output: %s", output_path)
            try:
                os.remove(output_path)
            except Exception:
                pass
            contiguous_done = 0
        if contiguous_done != pages_done:
            logging.info("PDF resume sync: state pages_done=%d, output contiguous_done=%d (%s)", pages_done, contiguous_done, output_path)
            pages_done = contiguous_done
            state_entry["pages_done"] = pages_done

    if args.overwrite or not args.resume:
        pages_done = 0
        state_entry["pages_done"] = 0

    if pages_done >= total_pages and total_pages > 0 and not args.overwrite:
        if os.path.exists(output_path):
            # Resume short-circuit: ensure file is marked done so next run can skip it.
            state_entry["status"] = "done"
            state_entry["updated_at"] = _now_iso()
            save_state(state_path, state)
            return 0
        # State says done but output is missing; rebuild output from scratch.
        pages_done = 0
        state_entry["pages_done"] = 0

    mode = "a" if (args.resume and pages_done > 0 and os.path.exists(output_path) and not args.overwrite) else "w"
    processed_images = 0
    existing_page_sections: dict[int, str] = {}
    if bool(getattr(args, "enrich_from_output", True)) and os.path.exists(output_path) and not args.overwrite:
        existing_page_sections = _extract_numbered_sections(_read_text_file(output_path), "Page")

    page_iter = range(pages_done + 1, total_pages + 1)

    def _process_one_page(fh, page_num: int) -> None:
        nonlocal processed_images
        args._trace = _trace_for_file(file_path, kind="pdf", page_num=page_num, pages_total=total_pages)  # type: ignore[attr-defined]
        images = convert_from_path(
            file_path,
            first_page=page_num,
            last_page=page_num,
            poppler_path=poppler_path,
            dpi=pdf_dpi,
        )
        if not images:
            raise RuntimeError("convert_from_path returned no images")
        img = images[0]

        existing_context = existing_page_sections.get(int(page_num), "")
        page_result = process_single_image(args, img, page_num, existing_context=existing_context)
        fh.write(page_result)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            pass

        state_entry["pages_done"] = page_num
        state_entry["updated_at"] = _now_iso()
        state_entry["status"] = "in_progress"
        save_state(state_path, state)
        processed_images += 1

    with open(output_path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(_file_meta_marker(args, file_path=file_path, file_type="pdf"))
        if args.show_page_progress:
            page_pbar = tqdm(page_iter, desc="PDF pages", unit="page", leave=False)
            try:
                for page_num in page_pbar:
                    try:
                        _process_one_page(f, int(page_num))
                    except Exception as e:
                        state_entry["last_error"] = str(e)
                        state_entry["updated_at"] = _now_iso()
                        state_entry["status"] = "failed"
                        save_state(state_path, state)
                        raise
            finally:
                page_pbar.close()
        else:
            for page_num in page_iter:
                try:
                    _process_one_page(f, int(page_num))
                except Exception as e:
                    state_entry["last_error"] = str(e)
                    state_entry["updated_at"] = _now_iso()
                    state_entry["status"] = "failed"
                    save_state(state_path, state)
                    raise

    state_entry["status"] = "done"
    state_entry["updated_at"] = _now_iso()
    save_state(state_path, state)
    return processed_images


def process_file(args: argparse.Namespace, file_path: str, state_path: str, state: dict[str, Any]) -> tuple[int, str, bool]:
    """处理单个文件，返回 (处理的图片页数, 文件类型 'pdf'|'png', 本次是否做了实际处理)."""
    file_path = os.path.abspath(file_path)
    file_key = _norm_key(file_path)
    output_path = _resolve_output_path(file_path, args.output_suffix)

    entry: dict[str, Any] = state["files"].setdefault(file_key, {})
    entry.setdefault("path", file_path)
    entry["output"] = output_path
    entry["updated_at"] = _now_iso()
    entry["clean_tag"] = str(getattr(args, "clean_tag", "DOJ_CLEAN_V2") or "DOJ_CLEAN_V2").strip()
    entry["detail_multiplier"] = float(getattr(args, "detail_multiplier", 3.0) or 3.0)
    entry["enrich_from_output"] = bool(getattr(args, "enrich_from_output", True))

    lower = file_path.lower()
    if lower.endswith(".pdf"):
        file_type = "pdf"
    elif lower.endswith(".png"):
        file_type = "png"
    else:
        file_type = "video"
    entry["type"] = file_type
    args._trace = _trace_for_file(file_path, kind=file_type)  # type: ignore[attr-defined]

    if args.overwrite and os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass

    clean_tag = str(getattr(args, "clean_tag", "DOJ_CLEAN_V2") or "DOJ_CLEAN_V2").strip()
    already_tagged = _output_has_clean_tag(output_path, clean_tag)
    if args.resume and os.path.exists(output_path) and _output_is_truncated(output_path, file_type):
        logging.warning("Detected truncated output (<=1 line or missing header), rebuilding: %s", output_path)
        try:
            os.remove(output_path)
        except Exception:
            pass
        if file_type == "pdf":
            entry["pages_done"] = 0
        elif file_type == "png":
            entry["pages_done"] = 0
        else:
            entry["frames_done"] = 0
        entry["status"] = "in_progress"
        save_state(state_path, state)
        already_tagged = False
    if not args.overwrite and entry.get("status") == "done" and os.path.exists(output_path) and already_tagged:
        return 0, file_type, False

    tqdm_print(f"\n=== Processing file: {file_path} ===")
    logging.info(f"Starting processing: {file_path}")

    ev: _DiskEventLog | None = getattr(args, "_event_log", None)
    started = time.monotonic()
    if ev is not None:
        ev.emit(
            "file_start",
            path=file_path,
            type=file_type,
            output=str(output_path),
            trace=str(getattr(args, "_trace", "") or ""),
        )

    try:
        if file_type == "png":
            entry["pages_total"] = 1
            entry["pages_done"] = 0
            entry["status"] = "in_progress"
            save_state(state_path, state)

            images_processed = process_png_file(args, file_path, output_path)
            entry["pages_done"] = 1
            entry["status"] = "done"
            entry["updated_at"] = _now_iso()
            save_state(state_path, state)
            if ev is not None:
                ev.emit(
                    "file_done",
                    path=file_path,
                    type=file_type,
                    processed=int(images_processed),
                    duration_s=round(time.monotonic() - started, 3),
                )
            return images_processed, file_type, True

        if file_type == "pdf":
            entry["status"] = "in_progress"
            save_state(state_path, state)
            images_processed = process_pdf_file(args, file_path, output_path, entry, state_path, state)
            if ev is not None:
                ev.emit(
                    "file_done",
                    path=file_path,
                    type=file_type,
                    processed=int(images_processed),
                    duration_s=round(time.monotonic() - started, 3),
                )
            return images_processed, file_type, True

        entry["status"] = "in_progress"
        save_state(state_path, state)
        frames_processed = process_video_file(args, file_path, output_path, entry, state_path, state)
        if ev is not None:
            ev.emit(
                "file_done",
                path=file_path,
                type=file_type,
                processed=int(frames_processed),
                duration_s=round(time.monotonic() - started, 3),
            )
        return frames_processed, file_type, True
    except Exception as e:
        tb = traceback.format_exc(limit=50)
        entry["status"] = "failed"
        entry["last_error"] = str(e)
        entry["updated_at"] = _now_iso()
        save_state(state_path, state)
        logging.exception("Failed to process file %s: %s", file_path, e)
        tqdm_print(f"ERROR: Failed to process file {file_path}: {e}")
        tqdm_print(tb.rstrip("\n"))
        if ev is not None:
            ev.emit(
                "file_failed",
                path=file_path,
                type=file_type,
                error=str(e),
                error_type=type(e).__name__,
                traceback=tb,
                duration_s=round(time.monotonic() - started, 3),
            )
        return 0, file_type, True

def recursive_scan(args: argparse.Namespace) -> None:
    """递归扫描并处理文件"""
    root = os.path.abspath(args.root)
    logging.info(f"Scanning directory: {root}")
    ev: _DiskEventLog | None = getattr(args, "_event_log", None)
    if ev is not None:
        ev.emit("scan_start", root=root)

    if bool(getattr(args, "fast_scan", False)) and bool(getattr(args, "batched_files", False)):
        logging.info("fast_scan enabled: disabling batched_files to use the fast per-page pipeline.")
        args.batched_files = False

    state_path = args.state_file
    if not os.path.isabs(state_path):
        state_path = os.path.join(root, state_path)
    state_path = os.path.abspath(state_path)
    state: dict[str, Any] = load_state(state_path)

    pdf_png_targets: list[str] = []
    video_targets: list[str] = []
    include_pdf = bool(args.include_pdf)
    include_png = bool(args.include_png)
    include_video = bool(args.include_video)
    video_exts = {str(ext).lstrip(".").lower() for ext in (args.video_exts or []) if str(ext).strip()}

    scan_cache_enabled = bool(getattr(args, "scan_cache", True))
    scan_cache_refresh = bool(getattr(args, "scan_cache_refresh", False))
    scan_cache_file = str(getattr(args, "scan_cache_file", ".doj_scan_cache.json.gz") or "").strip()
    if scan_cache_file and not os.path.isabs(scan_cache_file):
        scan_cache_file = os.path.abspath(os.path.join(root, scan_cache_file))
    scan_key = _scan_cache_key(root, include_pdf, include_png, include_video, video_exts)
    used_scan_cache = False

    logging.info(
        "Scan filters: pdf=%s png=%s video=%s video_exts=%s",
        include_pdf,
        include_png,
        include_video,
        ",".join(sorted(video_exts)) if video_exts else "(none)",
    )
    logging.info(
        "Scan cache: enabled=%s refresh=%s file=%s",
        scan_cache_enabled,
        scan_cache_refresh,
        scan_cache_file if scan_cache_file else "(none)",
    )

    if scan_cache_enabled and not scan_cache_refresh and scan_cache_file:
        cache = _load_scan_cache(scan_cache_file)
        if cache and cache.get("key") == scan_key:
            pdf_png_targets = list(cache.get("pdf_png_targets") or [])
            video_targets = list(cache.get("video_targets") or [])
            pdf_png_targets = sorted(set(os.path.abspath(p) for p in pdf_png_targets if isinstance(p, str) and p))
            video_targets = sorted(set(os.path.abspath(p) for p in video_targets if isinstance(p, str) and p))
            created_at = cache.get("created_at") or ""
            used_scan_cache = True
            logging.info(
                "Using scan cache (created_at=%s): targets(pdf/png=%d, video=%d). Use --scan-cache-refresh to rescan.",
                created_at,
                len(pdf_png_targets),
                len(video_targets),
            )

    if not used_scan_cache:
        scan_progress_every = int(getattr(args, "scan_progress_every", 0) or 0)
        dirs_seen = 0
        files_seen = 0
        matched_pdf = 0
        matched_png = 0
        matched_video = 0
        last_progress_at = 0

        for dirpath, _, filenames in os.walk(root):
            dirs_seen += 1
            for filename in filenames:
                files_seen += 1
                lower_name = filename.lower()
                if include_pdf and lower_name.endswith(".pdf"):
                    pdf_png_targets.append(os.path.abspath(os.path.join(dirpath, filename)))
                    matched_pdf += 1
                    continue
                if include_png and lower_name.endswith(".png"):
                    pdf_png_targets.append(os.path.abspath(os.path.join(dirpath, filename)))
                    matched_png += 1
                    continue
                if include_video and video_exts:
                    ext = os.path.splitext(lower_name)[1].lstrip(".")
                    if ext and ext in video_exts:
                        video_targets.append(os.path.abspath(os.path.join(dirpath, filename)))
                        matched_video += 1

                if scan_progress_every > 0 and (files_seen - last_progress_at) >= scan_progress_every:
                    last_progress_at = files_seen
                    logging.info(
                        "Scanning... dirs=%d files=%d matched(pdf=%d png=%d video=%d)",
                        dirs_seen,
                        files_seen,
                        matched_pdf,
                        matched_png,
                        matched_video,
                    )

        pdf_png_targets = sorted(set(pdf_png_targets))
        video_targets = sorted(set(video_targets))

        logging.info(
            "Scan complete: dirs=%d files=%d targets(pdf/png=%d, video=%d)",
            dirs_seen,
            files_seen,
            len(pdf_png_targets),
            len(video_targets),
        )

        if scan_cache_enabled and scan_cache_file:
            cache_payload: dict[str, Any] = {
                "version": 1,
                "created_at": _now_iso(),
                "key": scan_key,
                "root": root,
                "include_pdf": include_pdf,
                "include_png": include_png,
                "include_video": include_video,
                "video_exts": sorted(video_exts),
                "pdf_png_targets": pdf_png_targets,
                "video_targets": video_targets,
            }
            _save_scan_cache(scan_cache_file, cache_payload)

    total_files_seen = len(pdf_png_targets) + len(video_targets)
    if ev is not None:
        ev.emit(
            "scan_targets",
            root=root,
            total=int(total_files_seen),
            pdf_png=int(len(pdf_png_targets)),
            video=int(len(video_targets)),
            used_scan_cache=bool(used_scan_cache),
        )

    # No batching: treat "work unit" as a file; per-PDF page progress is shown by the inner progress bar.
    pdf_png_pending: list[str] = []
    video_pending: list[str] = []
    skipped_done = 0
    for fp in pdf_png_targets:
        key = _norm_key(fp)
        entry = state.get("files", {}).get(key, {})
        outp = _resolve_output_path(fp, args.output_suffix)
        clean_tag = str(getattr(args, "clean_tag", "DOJ_CLEAN_V2") or "DOJ_CLEAN_V2").strip()
        already_tagged = _output_has_clean_tag(str(outp), clean_tag)
        truncated = _output_is_truncated(str(outp), "pdf" if fp.lower().endswith(".pdf") else "png")
        if entry.get("status") == "done" and os.path.exists(outp) and not args.overwrite and already_tagged and not truncated:
            skipped_done += 1
            continue
        pdf_png_pending.append(fp)

    for vp in video_targets:
        key = _norm_key(vp)
        entry = state.get("files", {}).get(key, {})
        outp = _resolve_output_path(vp, args.output_suffix)
        clean_tag = str(getattr(args, "clean_tag", "DOJ_CLEAN_V2") or "DOJ_CLEAN_V2").strip()
        already_tagged = _output_has_clean_tag(str(outp), clean_tag)
        truncated = _output_is_truncated(str(outp), "video")
        if entry.get("status") == "done" and os.path.exists(outp) and not args.overwrite and already_tagged and not truncated:
            skipped_done += 1
            continue
        video_pending.append(vp)

    total_units = len(pdf_png_pending) + len(video_pending)
    logging.info(
        "Work files to process: %d (pdf/png files + video files). Skipping already-done outputs: %d",
        total_units,
        skipped_done,
    )

    pdf_files_done = 0
    png_files_done = 0
    video_files_done = 0
    units_done = 0

    with tqdm(total=total_units, desc="Processing", unit="file") as pbar:
        for fp in pdf_png_pending:
            images_processed, file_type, did_process = process_file(args, fp, state_path, state)
            if did_process:
                if file_type == "pdf":
                    pdf_files_done += 1
                elif file_type == "png":
                    png_files_done += 1
            pbar.update(1)
            units_done += 1
            pbar.set_postfix(PDFs=pdf_files_done, PNGs=png_files_done, Videos=video_files_done, Files=units_done)

        # Videos: one video at a time, do not interleave
        for vp in video_pending:
            key = _norm_key(vp)
            entry: dict[str, Any] = state["files"].setdefault(key, {})
            entry.setdefault("path", vp)
            entry["output"] = _resolve_output_path(vp, args.output_suffix)
            entry["type"] = "video"

            outp = _resolve_output_path(vp, args.output_suffix)
            entry["output"] = outp

            ev: _DiskEventLog | None = getattr(args, "_event_log", None)
            started = time.monotonic()

            tqdm_print(f"\n=== Processing video: {vp} ===")
            logging.info(f"Starting processing: {vp}")

            if ev is not None:
                ev.emit(
                    "file_start",
                    path=vp,
                    type="video",
                    output=str(outp),
                    trace=str(getattr(args, "_trace", "") or ""),
                )

            try:
                entry["status"] = "in_progress"
                entry["updated_at"] = _now_iso()
                save_state(state_path, state)

                process_video_file(args, vp, outp, entry, state_path, state)

                if ev is not None:
                    ev.emit(
                        "file_done",
                        path=vp,
                        type="video",
                        processed=int(entry.get("frames_done", 0)),
                        duration_s=round(time.monotonic() - started, 3),
                    )
                video_files_done += 1
            except Exception as e:
                tb = traceback.format_exc(limit=50)
                entry["status"] = "failed"
                entry["last_error"] = str(e)
                entry["updated_at"] = _now_iso()
                save_state(state_path, state)
                logging.exception("Failed to process video %s: %s", vp, e)
                tqdm_print(f"ERROR: Failed to process video {vp}: {e}")
                tqdm_print(tb.rstrip("\n"))
                if ev is not None:
                    ev.emit(
                        "file_failed",
                        path=vp,
                        type="video",
                        error=str(e),
                        error_type=type(e).__name__,
                        traceback=tb,
                        duration_s=round(time.monotonic() - started, 3),
                    )

            pbar.update(1)
            units_done += 1
            pbar.set_postfix(PDFs=pdf_files_done, PNGs=png_files_done, Videos=video_files_done, Files=units_done)

    tqdm_print("\n" + "=" * 50)
    tqdm_print("Summary:")
    tqdm_print(f"Root: {root}")
    tqdm_print(f"State file: {state_path}")
    tqdm_print(f"Total Files Seen: {total_files_seen}")
    tqdm_print(f"  - PDF Files Done: {pdf_files_done}")
    tqdm_print(f"  - PNG Files Done: {png_files_done}")
    tqdm_print(f"  - Video Files Done: {video_files_done}")
    tqdm_print(f"  - Units Done This Run: {units_done}")
    tqdm_print("=" * 50 + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursive PDF/PNG processor with resumable state.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO). Use DEBUG for more details.",
    )
    parser.add_argument("--stream", dest="stream", action="store_true", help="Use Ollama streaming responses (recommended).")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming; use single JSON response.")
    parser.set_defaults(stream=True)
    parser.add_argument("--root", default=os.getcwd(), help="Root directory to scan recursively.")
    parser.add_argument("--state-file", default=".doj_progress.json", help="Path to resumable state json.")
    parser.add_argument("--output-suffix", default=".txt", help="Output suffix appended to source stem. Examples: .txt, _2.txt")
    parser.add_argument("--resume", dest="resume", action="store_true", help="Resume from state/output if possible.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume; always start from scratch.")
    parser.set_defaults(resume=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt outputs and restart progress.")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_API_URL, help="Ollama generate API URL.")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds for model calls.")
    parser.add_argument("--retries", type=int, default=2, help="Retries for failed model calls.")
    parser.add_argument("--keep-alive", type=int, default=86400, help="Ollama keep_alive seconds per request (0 to disable). Default: 86400 (24h).")
    parser.add_argument("--num-ctx", type=int, default=4096, help="Ollama num_ctx (context window). Default: 4096. Set 0 to not force.")
    parser.add_argument("--num-predict", type=int, default=1024, help="Ollama num_predict (max tokens). Default: 1024. Set 0 to not force.")
    parser.add_argument("--num-ctx-refine", type=int, default=4096, help="Refiner num_ctx. Default: 4096.")
    parser.add_argument("--num-predict-refine", type=int, default=2048, help="Refiner num_predict. Default: 2048.")
    parser.add_argument("--temperature-refine", type=float, default=0.2, help="Refiner temperature. Default: 0.2.")
    parser.add_argument("--num-ctx-heavy", type=int, default=8192, help="Heavy num_ctx for OCR/video. Default: 8192.")
    parser.add_argument("--num-predict-heavy", type=int, default=2048, help="Heavy num_predict for OCR/video. Default: 2048.")
    parser.add_argument("--detail-multiplier", type=float, default=3.0, help="Target detail expansion multiplier (vs source). Default: 3.0.")
    parser.add_argument("--clean-tag", default="DOJ_CLEAN_V2", help="Tag written into per-file metadata markers.")
    parser.add_argument("--enrich-from-output", dest="enrich_from_output", action="store_true", help="Use existing txt output as historical context for enrichment (default: enabled).")
    parser.add_argument("--no-enrich-from-output", dest="enrich_from_output", action="store_false", help="Ignore historical txt context and only use current image/video inputs.")
    parser.set_defaults(enrich_from_output=True)
    parser.add_argument("--temperature-ocr", type=float, default=0.1, help="Temperature for OCR calls. Default: 0.1.")
    parser.add_argument("--unload-between-stages", dest="unload_between_stages", action="store_true", help="Unload model between stages to free VRAM.")
    parser.add_argument("--no-unload-between-stages", dest="unload_between_stages", action="store_false", help="Do not unload models between stages (default).")
    parser.set_defaults(unload_between_stages=False)
    parser.add_argument("--poppler-path", default="", help="Optional poppler bin path (Windows).")
    parser.add_argument("--pdf-dpi", type=int, default=300, help="DPI used when converting PDF pages to images. Default: 300.")
    parser.add_argument("--no-print-model-output", dest="print_model_output", action="store_false", help="Disable printing every model response.")
    parser.set_defaults(print_model_output=True)
    parser.add_argument("--show-page-progress", action="store_true", help="Show per-PDF page progress bar.")
    parser.add_argument("--show-video-progress", action="store_true", help="Show per-video frame progress bar.")

    parser.add_argument("--file-batch-size", type=int, default=10, help="Batch size for PNG/PDF scheduling (pages/images per batch).")
    parser.add_argument("--batched-files", dest="batched_files", action="store_true", help="(Deprecated) Enable batch scheduler for PNG/PDF.")
    parser.add_argument("--no-batched-files", dest="batched_files", action="store_false", help="Disable batch scheduler; process per-file sequentially.")
    parser.set_defaults(batched_files=False)

    parser.add_argument("--fast-scan", dest="fast_scan", action="store_true", help="Fast scan: no router votes, 1 call per page/image, no refine; video in small multi-frame batches.")
    parser.add_argument("--no-fast-scan", dest="fast_scan", action="store_false", help="Disable fast scan mode (default).")
    parser.set_defaults(fast_scan=False)
    parser.add_argument("--fast-video-batch-size", type=int, default=2, help="Fast scan: how many video frames per request. Default: 2.")

    parser.add_argument("--pdf", dest="include_pdf", action="store_true", help="Include .pdf files.")
    parser.add_argument("--no-pdf", dest="include_pdf", action="store_false", help="Exclude .pdf files.")
    parser.set_defaults(include_pdf=True)
    parser.add_argument("--png", dest="include_png", action="store_true", help="Include .png files.")
    parser.add_argument("--no-png", dest="include_png", action="store_false", help="Exclude .png files.")
    parser.set_defaults(include_png=True)

    parser.add_argument("--video", dest="include_video", action="store_true", help="Include video files.")
    parser.add_argument("--no-video", dest="include_video", action="store_false", help="Exclude video files.")
    parser.set_defaults(include_video=True)
    parser.add_argument("--video-ext", dest="video_exts", action="append", default=["mp4"], help="Video extension to include (repeatable). Default: mp4")
    parser.add_argument("--video-interval", type=float, default=3.0, help="Extract one frame every N seconds.")
    parser.add_argument("--video-batch-size", type=int, default=10, help="How many frames to send per multi-image request.")
    parser.add_argument("--video-chunk-multiplier", type=int, default=2, help="Process 2*batch frames per model rotation chunk (multiplier).")
    parser.add_argument("--video-max-frames", type=int, default=0, help="Max frames to extract (0 = no limit).")
    parser.add_argument("--video-ocr-mode", choices=["auto", "always", "never"], default="auto", help="When to run GLM-OCR on frames.")

    parser.add_argument("--think-qwen-vl", dest="think_qwen_vl", action="store_true", help="Enable thinking-mode prompt for qwen3-vl calls.")
    parser.add_argument("--no-think-qwen-vl", dest="think_qwen_vl", action="store_false", help="Disable thinking-mode prompt for qwen3-vl calls.")
    parser.set_defaults(think_qwen_vl=True)

    parser.add_argument("--think-qwen3", dest="think_qwen3", action="store_true", help="Enable thinking-mode prompt for qwen3 (refiner) calls.")
    parser.add_argument("--no-think-qwen3", dest="think_qwen3", action="store_false", help="Disable thinking-mode prompt for qwen3 (refiner) calls.")
    parser.set_defaults(think_qwen3=True)

    parser.add_argument(
        "--scan-progress-every",
        type=int,
        default=5000,
        help="Log scan progress every N files during directory walk (0 to disable). Default: 5000",
    )
    parser.add_argument("--scan-cache", dest="scan_cache", action="store_true", help="Reuse scan cache on next run (default: enabled).")
    parser.add_argument("--no-scan-cache", dest="scan_cache", action="store_false", help="Disable scan cache; always rescan directory tree.")
    parser.set_defaults(scan_cache=True)
    parser.add_argument("--scan-cache-refresh", action="store_true", help="Ignore cache and rescan once, then refresh scan cache file.")
    parser.add_argument("--scan-cache-file", default=".doj_scan_cache.json.gz", help="Path to scan cache (gzipped json). Default: .doj_scan_cache.json.gz")
    parser.add_argument("--router-votes", type=int, default=3, help="How many router votes per page (majority wins). Default: 3.")

    parser.add_argument("--disk-log", dest="disk_log", action="store_true", help="Write logs to disk (default: enabled).")
    parser.add_argument("--no-disk-log", dest="disk_log", action="store_false", help="Disable disk logging.")
    parser.set_defaults(disk_log=True)
    parser.add_argument("--log-dir", default=".doj_logs", help="Directory to store disk logs. Default: .doj_logs (relative to CWD).")
    parser.add_argument("--run-id", default="", help="Optional run id used as log filename prefix.")
    parser.add_argument("--capture-console", dest="capture_console", action="store_true", help="Capture tqdm_print/model outputs to <run-id>.console.log (default: enabled).")
    parser.add_argument("--no-capture-console", dest="capture_console", action="store_false", help="Disable <run-id>.console.log capture.")
    parser.set_defaults(capture_console=True)
    parser.add_argument("--console-fsync", dest="console_fsync", action="store_true", help="fsync console log on each write (slow).")
    parser.add_argument("--no-console-fsync", dest="console_fsync", action="store_false", help="Disable per-write fsync for console log (default).")
    parser.set_defaults(console_fsync=False)
    parser.add_argument("--event-fsync", dest="event_fsync", action="store_true", help="fsync events.jsonl (safer but slower).")
    parser.add_argument("--no-event-fsync", dest="event_fsync", action="store_false", help="Disable fsync for events.jsonl (default).")
    parser.set_defaults(event_fsync=False)
    parser.add_argument("--event-fsync-every", type=int, default=0, help="fsync every N events when --event-fsync is enabled (0 = every event). Default: 0.")
    parser.add_argument("--model-events", dest="model_events", action="store_true", help="Emit per-model-call events into events.jsonl (default: enabled).")
    parser.add_argument("--no-model-events", dest="model_events", action="store_false", help="Disable per-model-call events to reduce events.jsonl volume.")
    parser.set_defaults(model_events=True)
    parser.add_argument("--metrics-log", dest="metrics_log", action="store_true", help="Write periodic system snapshots to events log (default: enabled).")
    parser.add_argument("--no-metrics-log", dest="metrics_log", action="store_false", help="Disable periodic system snapshots.")
    parser.set_defaults(metrics_log=True)
    parser.add_argument("--metrics-interval", type=int, default=60, help="Seconds between system snapshots (0 to disable). Default: 60.")

    ns = parser.parse_args(argv)
    ns.models = dict(DEFAULT_MODELS)
    ns.video_interval = float(ns.video_interval)
    return ns

        
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    setattr(args, "_argv", list(sys.argv))
    _configure_disk_logging(args)
    logging.getLogger().setLevel(str(getattr(args, "log_level", "INFO")).upper())
    OLLAMA_STREAM = bool(getattr(args, "stream", True))
    opts: dict[str, Any] = {}
    num_ctx = int(getattr(args, "num_ctx", 0) or 0)
    num_predict = int(getattr(args, "num_predict", 0) or 0)
    if num_ctx > 0:
        opts["num_ctx"] = num_ctx
    if num_predict > 0:
        opts["num_predict"] = num_predict
    OLLAMA_REQUEST_OPTIONS = opts or None
    if OLLAMA_REQUEST_OPTIONS:
        logging.info("Ollama options (normal): %s", OLLAMA_REQUEST_OPTIONS)
    logging.info(
        "Ollama options (heavy): num_ctx=%d num_predict=%d temperature_ocr=%.3f",
        min(int(getattr(args, "num_ctx_heavy", 8192) or 8192), MAX_NUM_CTX),
        int(getattr(args, "num_predict_heavy", 2048) or 2048),
        float(getattr(args, "temperature_ocr", 0.1) or 0.1),
    )
    recursive_scan(args)
