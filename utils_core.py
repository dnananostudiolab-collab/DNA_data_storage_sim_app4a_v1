from __future__ import annotations

import gzip
import hashlib
import io
import math
import os
import tempfile
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

WORK_ROOT = Path(tempfile.gettempdir()) / "sm_rinf_ecc_dna_app"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

TEXT_EXTENSIONS = {".txt", ".text", ".md", ".csv", ".tsv", ".json", ".xml", ".html", ".htm", ".py", ".log", ".yaml", ".yml"}


def _can_decode_as_text(data: bytes) -> bool:
    data = bytes(data or b"")
    if not data:
        return True
    try:
        text = data[:8192].decode("utf-8")
    except Exception:
        return False
    # Allow common whitespace controls only; reject binary-like data.
    control = sum(1 for ch in text if ord(ch) < 32 and ch not in "\r\n\t")
    return control <= max(1, len(text) // 100)


def bytes_to_preview_text(data: bytes, limit: int = 12000) -> str:
    return bytes(data or b"").decode("utf-8", errors="replace")[:int(limit)]


@dataclass
class MagicInfo:
    kind: str
    ext: str
    mime: str
    confidence: float = 1.0
    note: str = ""


def safe_basename(name: str) -> str:
    name = os.path.basename(str(name or "file.bin"))
    out = []
    for ch in name:
        if ch.isalnum() or ch in "._- ()":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip() or "file.bin"


def fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "—"
    try:
        x = float(n)
    except Exception:
        return "—"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024.0 or unit == "TB":
            return f"{int(x)} B" if unit == "B" else f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TB"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(bytes(data or b"")).hexdigest()


def bytes_to_bitstring(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in bytes(data or b""))


def bitstring_to_bytes(bits: str, *, pad_to_byte: bool = True) -> Tuple[bytes, int]:
    bits = "".join(ch for ch in str(bits or "") if ch in "01")
    pad = 0
    if pad_to_byte and len(bits) % 8:
        pad = 8 - (len(bits) % 8)
        bits += "0" * pad
    out = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        if len(chunk) == 8:
            out.append(int(chunk, 2))
    return bytes(out), pad


def detect_magic(data: bytes, name: str = "") -> Optional[MagicInfo]:
    data = bytes(data or b"")
    head = data[:64]
    lower_name = str(name or "").lower()

    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return MagicInfo("png", ".png", "image/png")
    if head.startswith(b"\xff\xd8\xff"):
        return MagicInfo("jpeg", ".jpg", "image/jpeg")
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return MagicInfo("webp", ".webp", "image/webp")
    if head.startswith((b"GIF87a", b"GIF89a")):
        return MagicInfo("gif", ".gif", "image/gif")
    if head.startswith(b"BM"):
        return MagicInfo("bmp", ".bmp", "image/bmp")
    if head.startswith(b"%PDF"):
        return MagicInfo("pdf", ".pdf", "application/pdf")
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06") or head.startswith(b"PK\x07\x08"):
        # Office formats are ZIP containers; use the extension when present.
        if lower_name.endswith(".docx"):
            return MagicInfo("docx", ".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        if lower_name.endswith(".pptx"):
            return MagicInfo("pptx", ".pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation")
        if lower_name.endswith(".xlsx"):
            return MagicInfo("xlsx", ".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        return MagicInfo("zip", ".zip", "application/zip")
    if head.startswith(b"\x1f\x8b"):
        return MagicInfo("gzip", ".gz", "application/gzip")
    if head.startswith(b"BZh"):
        return MagicInfo("bz2", ".bz2", "application/x-bzip2")
    if head.startswith(b"\xfd7zXZ\x00"):
        return MagicInfo("xz", ".xz", "application/x-xz")
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return MagicInfo("wav", ".wav", "audio/wav")
    if head.startswith(b"ID3") or head.startswith(b"\xff\xfb"):
        return MagicInfo("mp3", ".mp3", "audio/mpeg")
    if head.startswith(b"fLaC"):
        return MagicInfo("flac", ".flac", "audio/flac")
    if head.startswith(b"OggS"):
        return MagicInfo("ogg", ".ogg", "audio/ogg")
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return MagicInfo("mp4", ".mp4", "video/mp4")

    # Lightweight text detection.
    if data:
        sample = data[:4096]
        try:
            text = sample.decode("utf-8")
            control = sum(1 for ch in text if ord(ch) < 32 and ch not in "\r\n\t")
            if control <= max(1, len(text) // 100):
                return MagicInfo("text", ".txt", "text/plain", 0.8)
        except Exception:
            pass
    return None


def magic_dict(data: bytes, name: str = "") -> Dict[str, Any]:
    m = detect_magic(data, name=name)
    if not m:
        return {"kind": "unknown", "ext": ".bin", "mime": "application/octet-stream", "confidence": 0.0, "note": ""}
    return {"kind": m.kind, "ext": m.ext, "mime": m.mime, "confidence": m.confidence, "note": m.note}


def get_domain(name: str, data: bytes) -> str:
    m = detect_magic(data, name=name)
    if not m:
        return "unknown"
    if m.kind in {"png", "jpeg", "webp", "gif", "bmp"}:
        return "image"
    if m.kind in {"wav", "mp3", "flac", "ogg"}:
        return "audio"
    if m.kind in {"mp4"}:
        return "video"
    if m.kind in {"pdf", "docx", "pptx", "xlsx"}:
        return "document"
    if m.kind in {"zip", "gzip", "xz", "bz2"}:
        return "archive"
    if m.kind == "text":
        return "text"
    return "other"


def validate_file_bytes(data: bytes, name: str = "") -> Tuple[bool, str, Dict[str, Any]]:
    data = bytes(data or b"")
    md = magic_dict(data, name=name)
    kind = md.get("kind", "unknown")
    try:
        if kind in {"png", "jpeg", "webp", "gif", "bmp"}:
            if Image is None:
                return True, "Image signature detected; Pillow is unavailable for deeper validation.", md
            img = Image.open(io.BytesIO(data))
            img.verify()
            return True, "Image opened successfully.", md
        if kind in {"zip", "docx", "pptx", "xlsx"}:
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                bad = zf.testzip()
                if bad is not None:
                    return False, f"ZIP internal check failed at {bad}.", md
            return True, "ZIP container opened successfully.", md
        if kind == "gzip":
            gzip.decompress(data)
            return True, "GZIP decompressed successfully.", md
        if kind == "pdf":
            return data.startswith(b"%PDF"), "PDF signature detected." if data.startswith(b"%PDF") else "PDF signature missing.", md
        if kind == "text":
            if _can_decode_as_text(data):
                return True, "Text file decoded successfully.", md
            return False, "Text extension detected, but content looks binary/corrupted.", md
        if kind in {"wav", "mp3", "flac", "ogg", "mp4"}:
            return True, "Container signature accepted.", md
        return False, "No recognizable container signature.", md
    except Exception as exc:
        return False, str(exc), md


def write_temp_file(data: bytes, preferred_name: str = "restored", ext: str = ".bin") -> str:
    out_dir = WORK_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ext.startswith("."):
        ext = "." + ext
    path = out_dir / f"{preferred_name}_{hashlib.sha1(bytes(data or b'')[:4096]).hexdigest()[:10]}{ext}"
    path.write_bytes(bytes(data or b""))
    return str(path)


_PREVIEW_WIDGET_COUNTER = 0


def preview_file_streamlit(st, path: str, title: str = "Preview", *, key_suffix: str = "") -> None:
    """Preview a file with a collision-safe Streamlit widget key.

    The same file can be previewed in Panel 1, Panel 2, Panel 5, and Panel 6,
    and both tabs can render similar widgets.  Text preview uses st.text_area,
    so the key must be globally unique.
    """
    global _PREVIEW_WIDGET_COUNTER
    _PREVIEW_WIDGET_COUNTER += 1

    st.markdown(f"#### {title}")
    if not path or not os.path.exists(path):
        st.info("Preview is not available.")
        return
    data = Path(path).read_bytes()
    md = magic_dict(data, name=path)
    kind = md.get("kind", "unknown")
    ext = Path(path).suffix.lower()
    key_base = f"{path}|{title}|{key_suffix}|{len(data)}|{_PREVIEW_WIDGET_COUNTER}"
    preview_key = "preview_" + hashlib.sha1(key_base.encode("utf-8", errors="ignore")).hexdigest()[:16]
    try:
        if kind in {"png", "jpeg", "webp", "gif", "bmp"}:
            st.image(path, width=220)
        elif kind in {"wav", "mp3", "flac", "ogg"}:
            st.audio(path)
        elif kind == "mp4":
            st.video(path)
        elif kind == "text" or ext in TEXT_EXTENSIONS or _can_decode_as_text(data):
            text = bytes_to_preview_text(data, limit=20000)
            st.text_area(
                "Text preview",
                text if text else "",
                height=260,
                label_visibility="collapsed",
                key=preview_key,
            )
        elif kind == "pdf":
            st.info("PDF preview is not rendered inline. Use download to inspect the file.")
        else:
            st.info("Preview is not available for this file type.")
    except Exception as exc:
        st.warning("Preview failed.")



def _metric_row(group: str, metric: str, original: Any, decoded: Any, value: Any, note: str = "") -> Dict[str, Any]:
    return {
        "Group": group,
        "Metric": metric,
        "Original": original,
        "Decoded / recovered": decoded,
        "Value": value,
        "Note": note,
    }


def _psnr_from_mse(mse: float, peak: float = 255.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(float(peak) / math.sqrt(float(mse))))


def _global_ssim_array(a, b, data_range: float = 255.0) -> float:
    """Dependency-light global SSIM approximation, averaged over channels."""
    if np is None:
        return float("nan")
    x = np.asarray(a).astype("float64")
    y = np.asarray(b).astype("float64")
    if x.shape != y.shape:
        raise ValueError("SSIM arrays must have the same shape")
    if x.ndim == 2:
        x = x[:, :, None]
        y = y[:, :, None]
    vals = []
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    for ch in range(x.shape[2]):
        xx = x[:, :, ch].reshape(-1)
        yy = y[:, :, ch].reshape(-1)
        mux = float(xx.mean())
        muy = float(yy.mean())
        vx = float(xx.var())
        vy = float(yy.var())
        cov = float(((xx - mux) * (yy - muy)).mean())
        vals.append(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2)))
    return float(np.mean(vals)) if vals else float("nan")


def image_quality_rows(original: bytes, decoded: bytes) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if Image is None or np is None:
        return [_metric_row("Image quality", "Image metrics", "—", "—", "Not available", "Install Pillow and numpy.")]
    try:
        img_a = Image.open(io.BytesIO(bytes(original or b""))).convert("RGB")
        img_b = Image.open(io.BytesIO(bytes(decoded or b""))).convert("RGB")
        original_size = f"{img_a.width}×{img_a.height}"
        decoded_size = f"{img_b.width}×{img_b.height}"
        note = ""
        if img_a.size != img_b.size:
            img_b = img_b.resize(img_a.size)
            note = "Decoded image resized to original size for metric calculation."
        arr_a = np.asarray(img_a).astype("float32")
        arr_b = np.asarray(img_b).astype("float32")
        diff = arr_a - arr_b
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))
        psnr = _psnr_from_mse(mse)
        ssim = _global_ssim_array(arr_a, arr_b, data_range=255.0)
        pixel_exact = float(np.mean(np.all(arr_a.astype("uint8") == arr_b.astype("uint8"), axis=2)))
        channel_accuracy = float(np.mean(arr_a.astype("uint8") == arr_b.astype("uint8")))
        rows.extend([
            _metric_row("Image quality", "Image size", original_size, decoded_size, "Match" if original_size == decoded_size else "Different", note),
            _metric_row("Image quality", "MSE", "0", "—", f"{mse:.6f}", "Lower is better."),
            _metric_row("Image quality", "MAE", "0", "—", f"{mae:.6f}", "Lower is better."),
            _metric_row("Image quality", "PSNR", "∞", "—", f"{psnr:.3f} dB", "Higher is better."),
            _metric_row("Image quality", "SSIM", "1.000", "—", f"{ssim:.6f}", "Global SSIM approximation; higher is better."),
            _metric_row("Image quality", "Exact pixel accuracy", "1.000", "—", f"{pixel_exact:.6f}", "All RGB channels must match."),
            _metric_row("Image quality", "Channel accuracy", "1.000", "—", f"{channel_accuracy:.6f}", "Per-channel equality."),
        ])
    except Exception as exc:
        rows.append(_metric_row("Image quality", "Image metrics", "—", "—", "Not available", str(exc)))
    return rows


def _words(text: str) -> List[str]:
    return [w for w in str(text or "").replace("\r", " ").replace("\n", " ").split(" ") if w]


def text_quality_rows(original: bytes, decoded: bytes) -> List[Dict[str, Any]]:
    o = bytes_to_preview_text(original, limit=max(len(bytes(original or b"")), 1_000_000))
    d = bytes_to_preview_text(decoded, limit=max(len(bytes(decoded or b"")), 1_000_000))
    char_dist = hamming_distance_str(o, d)
    char_acc = string_accuracy(o, d)
    ow = _words(o)
    dw = _words(d)
    n = min(len(ow), len(dw))
    word_dist = sum(1 for i in range(n) if ow[i] != dw[i]) + abs(len(ow) - len(dw))
    word_den = max(len(ow), len(dw), 1)
    word_acc = 1.0 - word_dist / word_den
    return [
        _metric_row("Text quality", "Characters", len(o), len(d), "Match" if len(o) == len(d) else "Different"),
        _metric_row("Text quality", "Character accuracy", "1.000", "—", f"{char_acc:.6f}"),
        _metric_row("Text quality", "Character differences", 0, "—", char_dist),
        _metric_row("Text quality", "Words", len(ow), len(dw), "Match" if len(ow) == len(dw) else "Different"),
        _metric_row("Text quality", "Word accuracy", "1.000", "—", f"{word_acc:.6f}", "Position-wise word comparison."),
        _metric_row("Text quality", "Exact text match", "Yes", "—", "Yes" if o == d else "No"),
    ]


def _read_wav_info(data: bytes) -> Tuple[Dict[str, Any], bytes]:
    with wave.open(io.BytesIO(bytes(data or b"")), "rb") as w:
        info = {
            "channels": w.getnchannels(),
            "sample_width": w.getsampwidth(),
            "frame_rate": w.getframerate(),
            "frames": w.getnframes(),
            "duration": w.getnframes() / float(w.getframerate() or 1),
        }
        frames = w.readframes(w.getnframes())
    return info, frames


def audio_quality_rows(original: bytes, decoded: bytes, name: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    omd = magic_dict(original, name=name)
    dmd = magic_dict(decoded, name=name)
    rows.append(_metric_row("Audio quality", "Container", omd.get("kind", "unknown"), dmd.get("kind", "unknown"), "Match" if omd.get("kind") == dmd.get("kind") else "Different"))
    if omd.get("kind") == "wav" and dmd.get("kind") == "wav":
        try:
            if np is None:
                raise RuntimeError("numpy is required for waveform metrics")
            oi, oframes = _read_wav_info(original)
            di, dframes = _read_wav_info(decoded)
            rows.extend([
                _metric_row("Audio quality", "Duration", f"{oi['duration']:.3f} s", f"{di['duration']:.3f} s", f"{abs(oi['duration']-di['duration']):.6f} s difference"),
                _metric_row("Audio quality", "Sample rate", oi["frame_rate"], di["frame_rate"], "Match" if oi["frame_rate"] == di["frame_rate"] else "Different"),
                _metric_row("Audio quality", "Channels", oi["channels"], di["channels"], "Match" if oi["channels"] == di["channels"] else "Different"),
            ])
            if oi["sample_width"] == di["sample_width"] and oi["channels"] == di["channels"]:
                dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
                dtype = dtype_map.get(int(oi["sample_width"]))
                if dtype is not None:
                    a = np.frombuffer(oframes, dtype=dtype).astype("float64")
                    b = np.frombuffer(dframes, dtype=dtype).astype("float64")
                    n = min(len(a), len(b))
                    if n > 0:
                        a = a[:n]
                        b = b[:n]
                        err = a - b
                        mse = float(np.mean(err ** 2))
                        rmse = float(math.sqrt(mse))
                        peak = float((2 ** (8 * oi["sample_width"] - 1)) - 1) if oi["sample_width"] > 1 else 255.0
                        psnr = _psnr_from_mse(mse, peak=peak)
                        signal_power = float(np.mean(a ** 2))
                        noise_power = mse
                        snr = 99.0 if noise_power <= 1e-12 else float(10 * math.log10(max(signal_power, 1e-12) / noise_power))
                        rows.extend([
                            _metric_row("Audio quality", "Waveform RMSE", "0", "—", f"{rmse:.6f}", "WAV PCM only."),
                            _metric_row("Audio quality", "Waveform PSNR", "∞", "—", f"{psnr:.3f} dB", "WAV PCM only."),
                            _metric_row("Audio quality", "Waveform SNR", "∞", "—", f"{snr:.3f} dB", "WAV PCM only."),
                        ])
        except Exception as exc:
            rows.append(_metric_row("Audio quality", "Waveform metrics", "—", "—", "Not available", str(exc)))
    else:
        rows.append(_metric_row("Audio quality", "Waveform metrics", "—", "—", "Not available", "Detailed audio PSNR/SNR is currently calculated for WAV PCM only. MP3/FLAC/OGG preview still uses container validation and byte accuracy."))
    return rows


def video_quality_rows(original_path: str, decoded_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if cv2 is None or np is None:
        return [_metric_row("Video quality", "Frame PSNR/SSIM", "—", "—", "Not available", "Install opencv-python-headless and numpy for sampled-frame video metrics.")]
    try:
        cap_a = cv2.VideoCapture(str(original_path))
        cap_b = cv2.VideoCapture(str(decoded_path))
        if not cap_a.isOpened() or not cap_b.isOpened():
            raise RuntimeError("Could not open one of the videos with OpenCV.")
        fps_a = cap_a.get(cv2.CAP_PROP_FPS) or 0.0
        fps_b = cap_b.get(cv2.CAP_PROP_FPS) or 0.0
        frames_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        width_b = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height_b = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        rows.extend([
            _metric_row("Video quality", "Resolution", f"{width_a}×{height_a}", f"{width_b}×{height_b}", "Match" if (width_a, height_a) == (width_b, height_b) else "Different"),
            _metric_row("Video quality", "FPS", f"{fps_a:.3f}", f"{fps_b:.3f}", "Match" if abs(fps_a - fps_b) < 1e-3 else "Different"),
            _metric_row("Video quality", "Frame count", frames_a, frames_b, "Match" if frames_a == frames_b else "Different"),
        ])
        if frames_a <= 0 or frames_b <= 0:
            raise RuntimeError("Frame count unavailable.")
        sample_count = min(10, frames_a, frames_b)
        positions = np.linspace(0, min(frames_a, frames_b) - 1, sample_count, dtype=int)
        psnrs = []
        ssims = []
        for pos in positions:
            cap_a.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            cap_b.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ok_a, fa = cap_a.read()
            ok_b, fb = cap_b.read()
            if not ok_a or not ok_b:
                continue
            if fa.shape != fb.shape:
                fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]))
            fa = cv2.cvtColor(fa, cv2.COLOR_BGR2RGB).astype("float32")
            fb = cv2.cvtColor(fb, cv2.COLOR_BGR2RGB).astype("float32")
            mse = float(np.mean((fa - fb) ** 2))
            psnrs.append(_psnr_from_mse(mse, peak=255.0))
            ssims.append(_global_ssim_array(fa, fb, data_range=255.0))
        cap_a.release()
        cap_b.release()
        if psnrs:
            rows.extend([
                _metric_row("Video quality", "Sampled-frame PSNR", "∞", "—", f"{float(np.mean(psnrs)):.3f} dB", f"Average over {len(psnrs)} sampled frames."),
                _metric_row("Video quality", "Sampled-frame SSIM", "1.000", "—", f"{float(np.mean(ssims)):.6f}", f"Global SSIM approximation over {len(ssims)} sampled frames."),
            ])
        else:
            rows.append(_metric_row("Video quality", "Sampled-frame metrics", "—", "—", "Not available", "No comparable frames could be read."))
    except Exception as exc:
        rows.append(_metric_row("Video quality", "Frame PSNR/SSIM", "—", "—", "Not available", str(exc)))
    return rows


def quality_metric_rows(original: bytes, decoded: bytes, *, input_name: str = "", input_path: str = "", decoded_path: str = "") -> List[Dict[str, Any]]:
    """Return domain-aware quality/recovery metrics for Panel 6 summarization."""
    original = bytes(original or b"")
    decoded = bytes(decoded or b"")
    md_o = magic_dict(original, name=input_name)
    md_d = magic_dict(decoded, name=input_name)
    domain = get_domain(input_name, original)
    rows: List[Dict[str, Any]] = [
        _metric_row("File recovery", "Detected domain", domain, get_domain(input_name, decoded), "Match" if get_domain(input_name, original) == get_domain(input_name, decoded) else "Different"),
        _metric_row("File recovery", "Original file type", md_o.get("kind", "unknown"), md_d.get("kind", "unknown"), "Match" if md_o.get("kind") == md_d.get("kind") else "Different"),
        _metric_row("File recovery", "File size", fmt_bytes(len(original)), fmt_bytes(len(decoded)), "Match" if len(original) == len(decoded) else "Different"),
        _metric_row("File recovery", "Byte accuracy", "1.000", "—", f"{byte_accuracy(original, decoded):.6f}"),
        _metric_row("File recovery", "Byte mismatches", 0, "—", byte_distance(original, decoded)),
        _metric_row("File recovery", "SHA256 match", "Yes", "—", "Yes" if original and decoded and sha256_bytes(original) == sha256_bytes(decoded) else "No"),
    ]
    if domain == "image":
        rows.extend(image_quality_rows(original, decoded))
    elif domain == "text":
        rows.extend(text_quality_rows(original, decoded))
    elif domain == "audio":
        rows.extend(audio_quality_rows(original, decoded, name=input_name))
    elif domain == "video":
        rows.extend(video_quality_rows(input_path, decoded_path))
    else:
        rows.append(_metric_row("Domain quality", "Domain-specific metrics", "—", "—", "Not available", "This file type uses container validation, byte accuracy, SHA256, and preview/download."))
    return rows

def hamming_distance_str(a: str, b: str) -> int:
    a = str(a or "")
    b = str(b or "")
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def string_accuracy(a: str, b: str) -> float:
    denom = max(len(str(a or "")), len(str(b or "")))
    if denom == 0:
        return 1.0
    return 1.0 - hamming_distance_str(a, b) / denom


def byte_distance(a: bytes, b: bytes) -> int:
    a = bytes(a or b"")
    b = bytes(b or b"")
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def byte_accuracy(a: bytes, b: bytes) -> float:
    denom = max(len(bytes(a or b"")), len(bytes(b or b"")))
    if denom == 0:
        return 1.0
    return 1.0 - byte_distance(a, b) / denom
