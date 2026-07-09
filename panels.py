from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from dna_design import (
    MAPPING_OPTIONS,
    clean_dna,
    display_mapping,
    encode_bytes_to_dna,
    decode_dna_to_bytes,
    gc_content,
    homopolymer_stats,
)
from error_model import mutate_dna
from rs_binary_ecc import decode_rs_bytes, encode_rs_bytes
from utils_core import (
    WORK_ROOT,
    byte_accuracy,
    byte_distance,
    bytes_to_bitstring,
    bytes_to_preview_text,
    detect_magic,
    fmt_bytes,
    get_domain,
    hamming_distance_str,
    magic_dict,
    preview_file_streamlit,
    quality_metric_rows,
    safe_basename,
    sha256_bytes,
    string_accuracy,
    validate_file_bytes,
    write_temp_file,
)

APP_STEPS = [
    (1, "Input"),
    (2, "Compression"),
    (3, "Encoding"),
    (4, "Strand Design"),
    (5, "Decoding"),
    (6, "Summarization"),
]

PANEL_TITLES = {
    "input": "Input",
    "data_encoding": "Compression",
    "dna_encoding": "Encoding",
    "strand_preparation": "Strand Design",
    "file_decoding": "Decoding",
    "validation": "Summarization",
}

BUTTONS = {
    "run_data_encoding": "Auto Prepare Container",
    "run_dna_encoding": "Run Encoding",
    "run_add_errors": "Run Add Errors",
    "run_decode": "Run Decode",
    "download_input_binary": "Download input binary",
    "download_stored_data": "Download stored data",
    "download_stored_binary": "Download stored binary",
    "download_encoded_dna": "Download encoded DNA",
    "download_noisy_dna": "Download noisy DNA",
    "download_decoded_file": "Download decoded output",
    "download_decoded_binary": "Download decoded binary",
    "download_summary": "Download summary CSV",
    "run_strand_preparation": "Run Strand Preparation",
}


def apply_app_style() -> None:
    st.markdown(
        """
<style>
:root {
  --bg:#F6F8FB;
  --surface:#FFFFFF;
  --surface-soft:#EEF4F8;
  --border:#D6E0EA;
  --text:#102033;
  --muted:#5F6F82;

  --primary:#0B5CAD;
  --primary-soft:#DCEEFF;

  --success:#0E9F6E;
  --success-soft:#DDF7EE;

  --warning:#C27803;
  --warning-soft:#FFF4D6;

  --danger:#D92D20;
  --danger-soft:#FEE4E2;
}
.stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 1.2rem; max-width: 1300px; }
.hero-card {
  background: linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 100%);
  border: 1px solid var(--border); border-radius: 18px; padding: 1.1rem 1.25rem;
  margin-bottom: 1rem; box-shadow: 0 10px 30px rgba(15,23,42,0.045);
}
.hero-title { font-size: 24px; font-weight: 780; letter-spacing: -0.02em; }
.hero-subtitle { color: var(--muted); font-size: 15px; margin-top: 0.25rem; }
.step-heading { display:flex; align-items:center; gap:0.65rem; margin: 0.1rem 0 0.8rem 0; }
.step-badge {
  width: 30px; height: 30px; border-radius: 999px; background: var(--primary); color:white;
  display:inline-flex; align-items:center; justify-content:center; font-weight:760;
}
.step-title { font-size: 20px; font-weight: 760; }
.pipeline-steps { display:grid; grid-template-columns: repeat(6, 1fr); gap:0.5rem; margin-bottom:1rem; }
.pipeline-step { border:1px solid var(--border); background:#fff; border-radius:14px; padding:0.65rem; }
.pipeline-step.done { background:#DCFCE7; border-color:#86EFAC; }
.pipeline-step.current { background:#DBEAFE; border-color:#93C5FD; }
.step-num { font-weight:760; margin-right:0.35rem; color:#1E3A8A; }
.step-name { font-weight:650; font-size:13px; }
.step-state { font-size:12px; color:var(--muted); margin-top:0.15rem; }
.region-tag {
  display:inline-block; padding:0.35rem 0.55rem; border-radius:12px; margin:0.1rem 0.2rem 0.1rem 0;
  font-family: Consolas, monospace; font-size:12px; line-height:1.6; word-break:break-all;
}
.error-base { background:#FECACA; color:#7F1D1D; font-weight:800; padding:0 1px; border-radius:3px; }
.small-note { color:#64748B; font-size:13px; }
</style>
""",
        unsafe_allow_html=True,
    )


def step_header(number: int, title: str) -> None:
    st.markdown(
        f"""
<div class="step-heading">
  <span class="step-badge">{number}</span>
  <span class="step-title">{title}</span>
</div>
""",
        unsafe_allow_html=True,
    )


def _key(prefix: str, name: str) -> str:
    return f"{prefix}_{name}"


def _content_key(prefix: str, name: str, value: Any) -> str:
    """Widget key that changes whenever the preview content changes."""
    if isinstance(value, bytes):
        raw = value
    else:
        raw = str(value or "").encode("utf-8", errors="ignore")
    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"{_key(prefix, name)}_{digest}"


def _download_text_button(label: str, text: str, file_name: str, *, key: str) -> None:
    st.download_button(
        label,
        data=str(text or "").encode("utf-8"),
        file_name=file_name,
        mime="text/plain",
        use_container_width=True,
        key=key,
    )


def _download_bytes_button(label: str, data: bytes, file_name: str, *, key: str) -> None:
    st.download_button(
        label,
        data=bytes(data or b""),
        file_name=file_name,
        mime="application/octet-stream",
        use_container_width=True,
        key=key,
    )


def _clear_downstream(prefix: str, start: str = "data") -> None:
    groups = {
        "data": ["stored_bytes", "stored_path", "stored_meta", "stored_signature"],
        "dna": ["payload_bytes", "protected_bytes", "ecc_meta", "dna", "bits", "codec_meta"],
        "error": ["noisy_dna", "error_events", "error_metrics"],
        "decode": ["decoded_payload_bytes", "decoded_before_repair_bytes", "decoded_data", "decoded_bits", "decoded_meta", "rs_report", "restored_path", "decoded_valid", "decoded_note", "decoded_magic"],
    }
    order = ["data", "dna", "error", "decode"]
    start_idx = order.index(start)
    for group in order[start_idx:]:
        for name in groups[group]:
            st.session_state.pop(_key(prefix, name), None)


def _input_available() -> bool:
    return bool(st.session_state.get("input_bytes"))


def _store_upload(uploaded) -> None:
    data = uploaded.getvalue()
    name = safe_basename(uploaded.name or "upload.bin")
    sig = f"{name}|{len(data)}|{sha256_bytes(data)}"
    if st.session_state.get("input_signature") == sig:
        return
    upload_dir = WORK_ROOT / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / name
    path.write_bytes(data)
    st.session_state.update({
        "input_signature": sig,
        "input_name": name,
        "input_path": str(path),
        "input_bytes": data,
    })
    # A new input invalidates both tabs.
    for p in ["base", "ecc"]:
        _clear_downstream(p, "data")


def _ensure_container_payload(prefix: str) -> bool:
    """Automatically prepare the file-container payload for Panel 2.

    App 1 currently has no active compressor. Therefore the compression-stage
    payload is the original file-container byte stream. This helper makes that
    assignment automatic after upload and prevents stale downstream state when
    the uploaded file changes.
    """
    data = st.session_state.get("input_bytes", b"") or b""
    path = st.session_state.get("input_path", "")
    name = st.session_state.get("input_name", "")
    if not data or not path:
        return False

    sig = f"container_preserving|{name}|{len(data)}|{sha256_bytes(data)}"
    if st.session_state.get(_key(prefix, "stored_signature")) == sig:
        return True

    _clear_downstream(prefix, "data")
    st.session_state[_key(prefix, "stored_bytes")] = data
    st.session_state[_key(prefix, "stored_path")] = path
    st.session_state[_key(prefix, "stored_signature")] = sig
    st.session_state[_key(prefix, "stored_meta")] = {
        "storage_method": "No compression",
        "source": "original_file_container_bytes",
        "input_name": name,
        "sha256": sha256_bytes(data),
        "auto_prepared": True,
    }
    return True


def _metrics_row(items: List[tuple[str, Any]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def _preview_seq(seq: str, n: int = 600) -> str:
    seq = clean_dna(seq)
    return seq[:n] + ("..." if len(seq) > n else "")


def _event_df(events: List[Dict[str, Any]], max_rows: int = 1000) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=["Original position", "Read position", "Operation", "Original base", "New/inserted base"])
    return pd.DataFrame(events[:max_rows])




DEFAULT_FBR = "ACACGACGCTCTTCCGATCT"
DEFAULT_RBR = "AGATCGGAAGAGCACACGTCT"
REGION_COLORS = {
    "FBR": ("#DCEEFF", "#0B5CAD"),
    "Index": ("#EAE6FF", "#4B2E83"),
    "Payload": ("#DDF7EE", "#0E6B4F"),
    "Filler": ("#EEF4F8", "#5F6F82"),
    "RBR": ("#FFF4D6", "#8A5A00"),
}


def _base4_index(n: int, length: int) -> str:
    """Deterministic A/C/G/T strand index with fixed length."""
    n = max(0, int(n))
    length = max(0, int(length))
    chars = []
    for _ in range(length):
        chars.append("ACGT"[n & 0b11])
        n >>= 2
    return "".join(reversed(chars))


def _make_filler(seed: int, length: int) -> str:
    if length <= 0:
        return ""
    bases = "ACGT"
    return "".join(bases[(seed + i) % 4] for i in range(length))


def _make_prepared_strands(
    dna: str,
    *,
    total_len: int = 125,
    index_len: int = 12,
    fbr: str = DEFAULT_FBR,
    rbr: str = DEFAULT_RBR,
) -> List[Dict[str, Any]]:
    """Prepare display/experiment strands as FBR + Index + Payload + Filler + RBR."""
    dna = clean_dna(dna)
    fbr = clean_dna(fbr)
    rbr = clean_dna(rbr)
    fixed = len(fbr) + int(index_len) + len(rbr)
    payload_capacity = int(total_len) - fixed
    if payload_capacity <= 0:
        raise ValueError(
            f"Total strand length is too short: total={total_len}, FBR={len(fbr)}, "
            f"Index={index_len}, RBR={len(rbr)}."
        )
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(dna), payload_capacity):
        no = len(rows) + 1
        payload = dna[start:start + payload_capacity]
        filler = _make_filler(no, payload_capacity - len(payload))
        index = _base4_index(no, int(index_len))
        full = fbr + index + payload + filler + rbr
        hp = homopolymer_stats(full)
        rows.append({
            "No.": no,
            "Type": "Prepared strand",
            "FBR": fbr,
            "Index": index,
            "Payload": payload,
            "Filler": filler,
            "RBR": rbr,
            "Full strand": full,
            "FBR length": len(fbr),
            "Index length": len(index),
            "Payload length": len(payload),
            "Filler length": len(filler),
            "RBR length": len(rbr),
            "Payload capacity": payload_capacity,
            "Total length": len(full),
            "Payload start in full": len(fbr) + len(index) + 1,
            "Payload global start": start + 1,
            "GC content": f"{gc_content(full):.3f}",
            "Longest homopolymer": hp.get("longest", 0),
        })
    return rows


def _row_regions(row: Dict[str, Any]) -> List[tuple[str, str]]:
    return [
        ("FBR", clean_dna(row.get("FBR", ""))),
        ("Index", clean_dna(row.get("Index", ""))),
        ("Payload", clean_dna(row.get("Payload", ""))),
        ("Filler", clean_dna(row.get("Filler", ""))),
        ("RBR", clean_dna(row.get("RBR", ""))),
    ]


def _region_html(name: str, seq: str, error_positions: set[int] | None = None, start_pos: int = 1) -> str:
    bg, fg = REGION_COLORS.get(name, ("#F8FAFC", "#0F172A"))
    error_positions = error_positions or set()
    chars = []
    for i, ch in enumerate(clean_dna(seq), start=start_pos):
        if i in error_positions:
            chars.append(f'<span class="error-base">{ch}</span>')
        else:
            chars.append(ch)
    body = "".join(chars) if chars else "—"
    return f'<span class="region-tag" style="background:{bg};color:{fg};"><b>{name}</b>: {body}</span>'


def _render_segmented_strand(row: Dict[str, Any], title: str, error_positions: set[int] | None = None) -> None:
    parts = []
    cursor = 1
    for name, seq in _row_regions(row):
        parts.append(_region_html(name, seq, error_positions, cursor))
        cursor += len(clean_dna(seq))
    st.markdown(f"**{title}**", unsafe_allow_html=True)
    st.markdown("".join(parts), unsafe_allow_html=True)


def _strand_summary(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    keep = ["No.", "Type", "Index length", "Payload length", "Filler length", "Total length", "GC content", "Longest homopolymer"]
    return pd.DataFrame([{k: r.get(k, "—") for k in keep} for r in rows])


def _concat_payloads(rows: List[Dict[str, Any]]) -> str:
    return "".join(clean_dna(r.get("Payload", "")) for r in rows)


def _mutate_prepared_rows(
    rows: List[Dict[str, Any]],
    *,
    scope: str,
    substitution_rate: float,
    insertion_rate: float,
    deletion_rate: float,
    seed: int,
    allow_indels: bool,
) -> tuple[List[Dict[str, Any]], str, List[Dict[str, Any]], Dict[str, Any]]:
    """Mutate prepared strands and return rows, noisy payload DNA, event rows, and metrics."""
    out_rows: List[Dict[str, Any]] = []
    all_events: List[Dict[str, Any]] = []
    total = {"substitutions": 0, "insertions": 0, "deletions": 0, "total_errors": 0}
    noisy_payloads: List[str] = []

    for row in rows:
        no = int(row.get("No.", 0))
        row_seed = int(seed) + no * 1000003
        fbr = clean_dna(row.get("FBR", ""))
        idx = clean_dna(row.get("Index", ""))
        payload = clean_dna(row.get("Payload", ""))
        filler = clean_dna(row.get("Filler", ""))
        rbr = clean_dna(row.get("RBR", ""))
        payload_full_start = len(fbr) + len(idx) + 1
        payload_global_start = int(row.get("Payload global start", 1))

        if scope == "Payload only":
            noisy_payload, evs, m = mutate_dna(
                payload,
                substitution_rate=substitution_rate,
                insertion_rate=insertion_rate,
                deletion_rate=deletion_rate,
                seed=row_seed,
                allow_indels=allow_indels,
            )
            new = dict(row)
            new["Payload"] = noisy_payload
            new["Full strand"] = fbr + idx + noisy_payload + filler + rbr
            for ev in evs:
                local_pos = int(ev.get("Original position", ev.get("Read position", 1)))
                ev2 = dict(ev)
                ev2.update({
                    "Strand": no,
                    "Region": "Payload",
                    "Full-strand position": payload_full_start + local_pos - 1,
                    "DNA payload position": payload_global_start + local_pos - 1,
                })
                all_events.append(ev2)
            noisy_payloads.append(noisy_payload)
        else:
            full = fbr + idx + payload + filler + rbr
            noisy_full, evs, m = mutate_dna(
                full,
                substitution_rate=substitution_rate,
                insertion_rate=insertion_rate,
                deletion_rate=deletion_rate,
                seed=row_seed,
                allow_indels=allow_indels,
            )
            # For substitution-only, boundaries are stable. With indels, this is only a best-effort slice.
            p0 = len(fbr) + len(idx)
            noisy_payload = noisy_full[p0:p0 + len(payload)]
            new = dict(row)
            new["Full strand"] = noisy_full
            new["Payload"] = noisy_payload
            for ev in evs:
                pos = int(ev.get("Original position", ev.get("Read position", 1)))
                region = "Payload" if p0 < pos <= p0 + len(payload) else "Non-payload"
                ev2 = dict(ev)
                ev2.update({
                    "Strand": no,
                    "Region": region,
                    "Full-strand position": pos,
                    "DNA payload position": payload_global_start + (pos - p0) - 1 if region == "Payload" else "—",
                })
                all_events.append(ev2)
            noisy_payloads.append(noisy_payload)

        for k in total:
            total[k] += int(m.get(k, 0))
        out_rows.append(new)

    return out_rows, "".join(noisy_payloads), all_events, total


def _strand_rows(dna: str, chunk_size: int = 120) -> List[Dict[str, Any]]:
    dna = clean_dna(dna)
    rows = []
    for i in range(0, len(dna), chunk_size):
        payload = dna[i:i + chunk_size]
        hp = homopolymer_stats(payload)
        rows.append({
            "No.": len(rows) + 1,
            "Type": "DNA chunk",
            "Payload length": len(payload),
            "Total length": len(payload),
            "GC content": f"{gc_content(payload):.3f}",
            "Longest homopolymer": hp.get("longest", 0),
            "Payload": payload,
            "Full strand": payload,
        })
    return rows


def _render_chunk(row: Dict[str, Any], title: str, error_positions: set[int] | None = None, offset: int = 0) -> None:
    seq = clean_dna(row.get("Full strand", ""))
    error_positions = error_positions or set()
    chars = []
    for local_i, ch in enumerate(seq, start=1):
        global_pos = offset + local_i
        if global_pos in error_positions:
            chars.append(f'<span class="error-base">{ch}</span>')
        else:
            chars.append(ch)
    body = "".join(chars) or "—"
    st.markdown(f"**{title}**", unsafe_allow_html=True)
    st.markdown(
        f'<span class="region-tag" style="background:#DCFCE7;color:#14532D;"><b>Payload</b>: {body}</span>',
        unsafe_allow_html=True,
    )


def _step_checks(prefix: str) -> Dict[int, bool]:
    return {
        1: _input_available(),
        2: bool(st.session_state.get(_key(prefix, "stored_bytes"))),
        3: bool(st.session_state.get(_key(prefix, "dna"))),
        4: bool(st.session_state.get(_key(prefix, "noisy_dna"))),
        5: st.session_state.get(_key(prefix, "decoded_data")) is not None,
        6: st.session_state.get(_key(prefix, "decoded_data")) is not None,
    }


def render_stepper(prefix: str) -> None:
    checks = _step_checks(prefix)
    parts = ['<div class="pipeline-steps">']
    for n, label in APP_STEPS:
        css = "done" if checks.get(n) else ("current" if all(checks.get(i) for i in range(1, n)) else "")
        state = "Done" if checks.get(n) else ("Next" if css == "current" else "Waiting")
        parts.append(
            f'<div class="pipeline-step {css}"><div><span class="step-num">{n}</span>'
            f'<span class="step-name">{label}</span></div><div class="step-state">{state}</div></div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_panel_1_upload(prefix: str) -> None:
    with st.container(border=True):
        step_header(1, PANEL_TITLES["input"])
        left, right = st.columns(2, gap="large")
        with left:
            uploaded = st.file_uploader("", type=None, key=_key(prefix, "upload"))
            if uploaded is not None:
                _store_upload(uploaded)
            st.caption("Upload any file to start the pipeline.")
        with right:
            data = st.session_state.get("input_bytes")
            path = st.session_state.get("input_path")
            name = st.session_state.get("input_name", "")
            if not data or not path:
                st.info("Upload a file to start.")
                return
            md = magic_dict(data, name=name)
            _metrics_row([
                ("Type", get_domain(name, data)),
                ("File extension", md.get("kind", "unknown")),
                ("Size", fmt_bytes(len(data))),
            ])
            preview_file_streamlit(st, path, "Input preview", key_suffix=_key(prefix, "input_preview"))
            bit_text = bytes_to_bitstring(data)
            with st.expander("Input binary", expanded=False):
                st.text_area("Binary bitstream", bit_text[:3000] + ("..." if len(bit_text) > 3000 else ""), height=120, key=_content_key(prefix, "input_bits_preview", bit_text))
                _download_text_button(BUTTONS["download_input_binary"], bit_text, "input_binary.txt", key=_key(prefix, "download_input_binary"))


def render_panel_2_data_encoding(prefix: str, ecc_enabled: bool) -> None:
    with st.container(border=True):
        step_header(2, PANEL_TITLES["data_encoding"])
        data = st.session_state.get("input_bytes")
        path = st.session_state.get("input_path")
        name = st.session_state.get("input_name", "")
        if not data or not path:
            st.info("Upload a file first.")
            return

        prepared = _ensure_container_payload(prefix)
        if not prepared:
            st.info("Upload a file first.")
            return

        stored = st.session_state.get(_key(prefix, "stored_bytes"), b"") or b""
        stored_path = st.session_state.get(_key(prefix, "stored_path"), path)
        md = magic_dict(stored, name=name)

        # st.markdown("#### Container bytes prepared automatically")
        # st.caption(
        #     "No active compression is applied in this app. The original file-container bytes "
        #     "are automatically used as the storage payload after upload."
        # )
        # if ecc_enabled:
        #     st.info("Reed–Solomon is applied after this automatic preparation step and before DNA mapping.")
        # else:
        #     st.info("No ECC is applied. DNA errors may corrupt the decoded bytes, so exact recovery is evaluated by SHA256 and byte accuracy.")

        _metrics_row([
            ("Compression mode", "No compression"),
            ("Prepared payload", fmt_bytes(len(stored))),
            ("Payload type", md.get("kind", "unknown")),
        ])
        preview_file_streamlit(st, bytes_to_bitstring(stored), "Prepared payload preview", key_suffix=_key(prefix, "stored_preview"))
        d1, d2 = st.columns(2)
        with d1:
            _download_bytes_button(BUTTONS["download_stored_data"], stored, f"stored_data{md.get('ext', '.bin')}", key=_key(prefix, "download_stored_data"))
        with d2:
            _download_text_button(BUTTONS["download_stored_binary"], bytes_to_bitstring(stored), "stored_binary.txt", key=_key(prefix, "download_stored_binary"))


def _encode_current_payload(prefix: str, ecc_enabled: bool, stored: bytes, mapping: str, block_size: int, parity: int) -> None:
    """Encode current settings and update session state."""
    _clear_downstream(prefix, "dna")
    if ecc_enabled:
        rs = encode_rs_bytes(stored, data_block_size=int(block_size), parity_bytes=int(parity))
        payload = rs.protected_bytes
        st.session_state[_key(prefix, "protected_bytes")] = payload
        st.session_state[_key(prefix, "ecc_meta")] = rs.meta
    else:
        payload = stored
        st.session_state[_key(prefix, "ecc_meta")] = {}
    dna, bits, meta = encode_bytes_to_dna(payload, mapping)
    meta.update({
        "ecc_enabled": bool(ecc_enabled),
        "source_bytes_len": len(stored),
        "payload_bytes_len": len(payload),
    })
    st.session_state[_key(prefix, "payload_bytes")] = payload
    st.session_state[_key(prefix, "dna")] = dna
    st.session_state[_key(prefix, "bits")] = bits
    st.session_state[_key(prefix, "codec_meta")] = meta
    st.session_state[_key(prefix, "last_encode_signature")] = f"{sha256_bytes(stored)}|{mapping}|{int(block_size)}|{int(parity)}|{ecc_enabled}"


def render_panel_3_dna_encoding(prefix: str, ecc_enabled: bool) -> None:
    with st.container(border=True):
        step_header(3, PANEL_TITLES["dna_encoding"])
        stored = st.session_state.get(_key(prefix, "stored_bytes"))
        if not stored:
            _ensure_container_payload(prefix)
            stored = st.session_state.get(_key(prefix, "stored_bytes"))
        if not stored:
            st.info("Upload a file first.")
            return
        mapping = st.selectbox(
            "DNA design rule",
            MAPPING_OPTIONS,
            index=0,
            format_func=display_mapping,
            key=_key(prefix, "mapping"),
        )

        # Case 2 design: SM/R∞ remain the only DNA mappings. Reed–Solomon is
        # an ECC option at the byte/binary layer before DNA mapping, not a third
        # DNA mapping option.
        ecc_options = ["Reed–Solomon"] if ecc_enabled else ["None"]
        ecc_method = st.selectbox(
            "ECC option",
            ecc_options,
            index=0,
            key=_key(prefix, "ecc_option"),
            help="Reed–Solomon protects file bytes before SM/R∞ DNA mapping. It is not a separate DNA mapping.",
        )
        st.caption(
            f"Current pipeline: file container bytes → "
            f"{('RS-protected bytes → ' if ecc_enabled else '')}"
            f"{display_mapping(mapping)} DNA → strand design → errors → decode"
            f"{(' → RS repair' if ecc_enabled else '')}."
        )

        if ecc_enabled:
            st.markdown("#### Reed–Solomon ECC settings")
            c1, c2 = st.columns(2)
            block_size = c1.number_input("RS data block size (bytes)", min_value=8, max_value=223, value=64, step=8, key=_key(prefix, "rs_block_size"))
            parity = c2.number_input("RS parity bytes/block", min_value=4, max_value=128, value=64, step=4, key=_key(prefix, "rs_parity"))
            if int(block_size) + int(parity) > 255:
                st.error("GF(2^8) RS requires data block size + parity bytes ≤ 255.")
                return
            max_unknown = int(parity) // 2
            st.caption(
                f"Rule of thumb: this setting can correct up to {max_unknown} unknown byte errors per "
                f"{int(block_size) + int(parity)}-byte codeword. For 3% DNA substitution, use ≥64 parity bytes/block first."
            )
        else:
            block_size = 0
            parity = 0

        desired_sig = f"{sha256_bytes(stored)}|{mapping}|{int(block_size)}|{int(parity)}|{ecc_enabled}"
        existing_sig = st.session_state.get(_key(prefix, "last_encode_signature"))
        has_previous_encoding = bool(st.session_state.get(_key(prefix, "dna")))

        if st.button(BUTTONS["run_dna_encoding"], key=_key(prefix, "run_dna_encoding")):
            _encode_current_payload(prefix, ecc_enabled, stored, mapping, int(block_size), int(parity))
            existing_sig = desired_sig
        elif has_previous_encoding and existing_sig != desired_sig:
            # Avoid the confusing stale preview problem: when SM/R∞ or RS parameters
            # change, immediately regenerate DNA so the preview and metrics match.
            _encode_current_payload(prefix, ecc_enabled, stored, mapping, int(block_size), int(parity))
            existing_sig = desired_sig
            st.caption("DNA preview was updated automatically because the DNA design settings changed.")

        dna = st.session_state.get(_key(prefix, "dna"), "")
        if not dna:
            st.info("Run DNA Encoding to continue.")
            return
        payload = st.session_state.get(_key(prefix, "payload_bytes"), b"") or b""
        ecc_meta = st.session_state.get(_key(prefix, "ecc_meta"), {}) or {}
        codec_meta = st.session_state.get(_key(prefix, "codec_meta"), {}) or {}
        encoded_mapping = codec_meta.get("mapping", mapping)
        hp = homopolymer_stats(dna)
        baseline_nt = max(1, len(stored) * 4)
        if ecc_enabled:
            _metrics_row([
                ("DNA design rule", display_mapping(encoded_mapping)),
                ("ECC option", "Reed–Solomon"),
                ("Original size", fmt_bytes(len(stored))),
                ("RS protected size", fmt_bytes(len(payload))),
                ("DNA length", f"{len(dna):,} nt"),
            ])
            _metrics_row([
                ("ECC overhead", f"{ecc_meta.get('ecc_overhead_ratio', 1.0):.2f}×"),
                ("Max byte errors/block", ecc_meta.get("max_unknown_byte_errors_per_block", "—")),
                ("GC content", f"{gc_content(dna):.3f}"),
                ("Longest HP", hp.get("longest", 0)),
                ("DNA expansion vs original", f"{len(dna) / baseline_nt:.2f}×"),
            ])
        else:
            _metrics_row([
                ("DNA design rule", display_mapping(encoded_mapping)),
                ("ECC option", "None"),
                ("Binary length", f"{len(st.session_state.get(_key(prefix, 'bits'), '')):,} bits"),
                ("DNA length", f"{len(dna):,} nt"),
                ("GC content", f"{gc_content(dna):.3f}"),
            ])
        st.text_area("Base string", _preview_seq(dna, 900), height=150, key=_content_key(prefix, "dna_preview", dna))
        d1, d2 = st.columns(2)
        with d1:
            _download_text_button(BUTTONS["download_encoded_dna"], dna, "encoded_dna.txt", key=_key(prefix, "download_encoded_dna"))
        with d2:
            _download_text_button("Download encoded binary", st.session_state.get(_key(prefix, "bits"), ""), "encoded_binary.txt", key=_key(prefix, "download_encoded_binary"))

def render_panel_4_errors(prefix: str, ecc_enabled: bool) -> None:
    with st.container(border=True):
        step_header(4, PANEL_TITLES["strand_preparation"])
        dna = st.session_state.get(_key(prefix, "dna"), "")
        if not dna:
            st.info("Run DNA Encoding first.")
            return

        st.markdown("#### Strand Design")
        if ecc_enabled:
            st.caption(
                "In this app, Reed–Solomon is applied at the binary/byte layer before SM/R∞. "
                "Therefore the prepared DNA strands below still use FBR + Index + Payload + Filler + RBR."
            )
        else:
            st.caption("Prepared strands use the same visual structure as the earlier panel.py UI.")

        with st.expander("Strand design settings", expanded=not bool(st.session_state.get(_key(prefix, "strand_rows")))):
            a, b = st.columns(2)
            total_len = a.number_input("Total strand length", min_value=80, max_value=250, value=125, step=1, key=_key(prefix, "strand_total_len"))
            index_len = b.number_input("Index length", min_value=0, max_value=24, value=12, step=1, key=_key(prefix, "strand_index_len"))
            fbr = st.text_input("FBR", value=DEFAULT_FBR, key=_key(prefix, "strand_fbr"))
            rbr = st.text_input("RBR", value=DEFAULT_RBR, key=_key(prefix, "strand_rbr"))
            build_clicked = st.button(BUTTONS["run_strand_preparation"], key=_key(prefix, "run_strand_preparation"))

        strand_sig = f"{hashlib.sha256(clean_dna(dna).encode()).hexdigest()}|{int(total_len)}|{int(index_len)}|{clean_dna(fbr)}|{clean_dna(rbr)}"
        if build_clicked or st.session_state.get(_key(prefix, "strand_signature")) != strand_sig:
            try:
                rows = _make_prepared_strands(dna, total_len=int(total_len), index_len=int(index_len), fbr=fbr, rbr=rbr)
            except Exception as exc:
                st.error(str(exc))
                return
            st.session_state[_key(prefix, "strand_rows")] = rows
            st.session_state[_key(prefix, "strand_signature")] = strand_sig
            # Strand design changes invalidate downstream errors/decode.
            st.session_state.pop(_key(prefix, "noisy_dna"), None)
            st.session_state.pop(_key(prefix, "noisy_strand_rows"), None)
            st.session_state.pop(_key(prefix, "error_events"), None)
            st.session_state.pop(_key(prefix, "error_metrics"), None)
            st.session_state.pop(_key(prefix, "decoded_data"), None)

        rows: List[Dict[str, Any]] = st.session_state.get(_key(prefix, "strand_rows"), []) or []
        if not rows:
            st.info("Run Strand Preparation to continue.")
            return

        total_full_len = sum(len(clean_dna(r.get("Full strand", ""))) for r in rows)
        dna_len = len(clean_dna(dna))
        _metrics_row([
            ("Designed strands", len(rows)),
            ("Total strand length", f"{total_full_len:,} nt"),
            ("Strand Design length increase", f"{total_full_len / max(1, dna_len):.2f}×"),
            ("DNA design rule", display_mapping(st.session_state.get(_key(prefix, "mapping"), ""))),
        ])
        st.dataframe(_strand_summary(rows), use_container_width=True, hide_index=True)
        selected = st.selectbox("Inspect designed strand", [str(r["No."]) for r in rows], key=_key(prefix, "inspect_chunk"))
        row = rows[int(selected) - 1]
        _render_segmented_strand(row, "Designed strand")

        st.markdown("---")
        st.markdown("#### Add DNA errors")
        a, b, c, d = st.columns(4)
        scope = a.selectbox("Error target", ["Payload only", "Full strand"], index=0, key=_key(prefix, "error_scope"))
        sub = b.number_input("Substitution", min_value=0.0, max_value=0.2, value=0.001, step=0.001, format="%.4f", key=_key(prefix, "sub_rate"))
        seed = c.number_input("Seed", min_value=1, max_value=999999, value=7, step=1, key=_key(prefix, "error_seed"))
        allow_indels = d.checkbox("Allow indels", value=False, key=_key(prefix, "allow_indels"))
        e, f = st.columns(2)
        ins = e.number_input("Insertion", min_value=0.0, max_value=0.2, value=0.0, step=0.001, format="%.4f", key=_key(prefix, "ins_rate"), disabled=not allow_indels)
        dele = f.number_input("Deletion", min_value=0.0, max_value=0.2, value=0.0, step=0.001, format="%.4f", key=_key(prefix, "del_rate"), disabled=not allow_indels)
        if allow_indels:
            st.warning("Insertion/deletion changes DNA length and can break SM/R∞ framing. Use substitution-only first for clean ECC comparison.")
        if ecc_enabled and float(sub) >= 0.03:
            st.warning(
                "3% DNA substitution is a severe stress test. With SM/R∞, ~11.5% of bytes are expected to be touched. "
                "Use RS parity ≥64 bytes/block or reduce the error rate to 0.5–1% for stable recovery."
            )

        if st.button(BUTTONS["run_add_errors"], key=_key(prefix, "run_add_errors")):
            _clear_downstream(prefix, "error")
            err_rows, noisy_payload_dna, events, metrics = _mutate_prepared_rows(
                rows,
                scope=scope,
                substitution_rate=float(sub),
                insertion_rate=float(ins),
                deletion_rate=float(dele),
                seed=int(seed),
                allow_indels=bool(allow_indels),
            )
            st.session_state[_key(prefix, "noisy_strand_rows")] = err_rows
            st.session_state[_key(prefix, "noisy_dna")] = noisy_payload_dna
            st.session_state[_key(prefix, "error_events")] = events
            st.session_state[_key(prefix, "error_metrics")] = metrics

        noisy = st.session_state.get(_key(prefix, "noisy_dna"), "")
        if not noisy:
            st.info("Run Add Errors to continue.")
            return
        events = st.session_state.get(_key(prefix, "error_events"), []) or []
        em = st.session_state.get(_key(prefix, "error_metrics"), {}) or {}
        _metrics_row([
            ("Added errors", em.get("total_errors", 0)),
            ("Substitutions", em.get("substitutions", 0)),
            ("Insertions", em.get("insertions", 0)),
            ("Deletions", em.get("deletions", 0)),
            ("Payload DNA accuracy", f"{string_accuracy(dna, noisy):.4f}"),
        ])

        err_positions = {
            int(ev.get("Full-strand position"))
            for ev in events
            if str(ev.get("Strand")) == str(selected)
            and ev.get("Operation") in {"substitution", "deletion"}
            and str(ev.get("Full-strand position", "")).isdigit()
        }
        _render_segmented_strand(row, "Clean strand", error_positions=err_positions)
        noisy_rows = st.session_state.get(_key(prefix, "noisy_strand_rows"), []) or []
        if int(selected) - 1 < len(noisy_rows):
            _render_segmented_strand(noisy_rows[int(selected) - 1], "Error strand", error_positions=err_positions)
        st.text_area("Recovered payload DNA for decoding", _preview_seq(noisy, 900), height=150, key=_content_key(prefix, "noisy_dna_preview", noisy))
        df = _event_df(events)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        _download_text_button(BUTTONS["download_noisy_dna"], noisy, "noisy_dna.txt", key=_key(prefix, "download_noisy_dna"))

def render_panel_5_decoding(prefix: str, ecc_enabled: bool) -> None:
    with st.container(border=True):
        step_header(5, PANEL_TITLES["file_decoding"])
        dna = st.session_state.get(_key(prefix, "dna"), "")
        noisy = st.session_state.get(_key(prefix, "noisy_dna"), "")
        if not dna:
            st.info("Run DNA Encoding first.")
            return
        source_options = ["Without errors"]
        if noisy:
            source_options.append("With errors")
        default = 1 if noisy else 0
        source = st.radio("", source_options, index=default, horizontal=True, key=_key(prefix, "decode_source"))
        input_dna = noisy if source == "With errors" else dna
        mapping = st.session_state.get(_key(prefix, "mapping"), "Simple Mapping")
        _metrics_row([
            ("DNA mapping", display_mapping(mapping)),
            ("Input DNA", source),
            ("ECC", "Reed–Solomon" if ecc_enabled else "None"),
        ])
        st.text_area("Input DNA preview", _preview_seq(input_dna, 700), height=120, key=_content_key(prefix, "decode_input_preview", input_dna))

        if st.button(BUTTONS["run_decode"], key=_key(prefix, "run_decode")):
            _clear_downstream(prefix, "decode")
            decoded_payload, decoded_bits, decoded_meta = decode_dna_to_bytes(input_dna, mapping)
            st.session_state[_key(prefix, "decoded_payload_bytes")] = decoded_payload
            st.session_state[_key(prefix, "decoded_before_repair_bytes")] = decoded_payload
            st.session_state[_key(prefix, "decoded_bits")] = decoded_bits
            st.session_state[_key(prefix, "decoded_meta")] = decoded_meta

            rs_report: Dict[str, Any] = {}
            if ecc_enabled:
                ecc_meta = st.session_state.get(_key(prefix, "ecc_meta"), {}) or {}
                final_data, rs_report = decode_rs_bytes(decoded_payload, ecc_meta)
            else:
                final_data = decoded_payload
            input_name = st.session_state.get("input_name", "")
            valid, note, md = validate_file_bytes(final_data, name=input_name)
            ext = md.get("ext", ".bin")
            if md.get("kind") == "unknown" and input_name:
                # Keep the user's original file extension so text-like restored files
                # can still be previewed/downloaded with the expected type.
                ext = Path(str(input_name)).suffix or ext
            restored_path = write_temp_file(final_data, preferred_name=f"{prefix}_decoded", ext=ext)
            st.session_state[_key(prefix, "decoded_data")] = final_data
            st.session_state[_key(prefix, "decoded_valid")] = valid
            st.session_state[_key(prefix, "decoded_note")] = note
            st.session_state[_key(prefix, "decoded_magic")] = md
            st.session_state[_key(prefix, "restored_path")] = restored_path
            st.session_state[_key(prefix, "rs_report")] = rs_report

        data = st.session_state.get(_key(prefix, "decoded_data"))
        if data is None:
            st.info("Run Decode to continue.")
            return
        md = st.session_state.get(_key(prefix, "decoded_magic"), {}) or {}
        valid = bool(st.session_state.get(_key(prefix, "decoded_valid")))
        note = st.session_state.get(_key(prefix, "decoded_note"), "")
        decoded_payload = st.session_state.get(_key(prefix, "decoded_payload_bytes"), b"") or b""
        original = st.session_state.get(_key(prefix, "stored_bytes"), b"") or b""
        exact_recovery = bool(original) and bool(data) and sha256_bytes(original) == sha256_bytes(data)
        recovery_status = "Recovered exactly" if exact_recovery else ("Corrupted but readable" if valid else "Corrupted / not readable")
        if ecc_enabled:
            rr = st.session_state.get(_key(prefix, "rs_report"), {}) or {}
            _metrics_row([
                ("Decoded protected size", fmt_bytes(len(decoded_payload))),
                ("Decoded output size", fmt_bytes(len(data))),
                ("RS corrected symbols", rr.get("corrected_symbols", 0)),
                ("Failed RS blocks", rr.get("failed_blocks", 0)),
                ("Exact recovery", "Yes" if exact_recovery else "No"),
            ])
        else:
            _metrics_row([
                ("Decoded size", fmt_bytes(len(data))),
                ("Decoded type", md.get("kind", "unknown")),
                ("Readable / valid", "Yes" if valid else "No"),
                ("Exact recovery", "Yes" if exact_recovery else "No"),
            ])
        st.caption(f"{recovery_status}. {note}" if note else recovery_status)
        restored_path = st.session_state.get(_key(prefix, "restored_path"))
        if restored_path:
            preview_file_streamlit(st, restored_path, "Decoded output preview", key_suffix=_key(prefix, "decoded_preview_panel5"))
            if not valid:
                st.warning("The decoded bytes were saved, but container validation failed. The preview above is best-effort only.")
        else:
            st.info("Decoded bytes were written, but a file preview path is not available.")
        d1, d2 = st.columns(2)
        with d1:
            _download_bytes_button(BUTTONS["download_decoded_file"], data, f"decoded{md.get('ext', '.bin')}", key=_key(prefix, "download_decoded_file"))
        with d2:
            _download_text_button(BUTTONS["download_decoded_binary"], bytes_to_bitstring(data), "decoded_binary.txt", key=_key(prefix, "download_decoded_binary"))




def _render_property_table(rows: List[Dict[str, Any]]) -> None:
    clean_rows = []
    for row in rows or []:
        clean_rows.append({
            "Property": row.get("Property", row.get("Metric", "—")),
            "Value": row.get("Value", "—") if row.get("Value", "—") not in (None, "") else "—",
        })
    st.dataframe(pd.DataFrame(clean_rows), use_container_width=True, hide_index=True)


def _basic_file_rows(path: str, data: bytes, *, label: str = "File") -> List[Dict[str, Any]]:
    data = bytes(data or b"")
    md = magic_dict(data, name=path or st.session_state.get("input_name", ""))
    return [
        {"Property": "Data", "Value": label},
        {"Property": "Type", "Value": md.get("kind", "unknown")},
        {"Property": "Size", "Value": fmt_bytes(len(data))},
        {"Property": "SHA256", "Value": sha256_bytes(data)[:16] + "..." if data else "—"},
    ]


def _encoded_summary_rows(prefix: str, ecc_enabled: bool) -> List[Dict[str, Any]]:
    stored = st.session_state.get(_key(prefix, "stored_bytes"), b"") or b""
    payload = st.session_state.get(_key(prefix, "payload_bytes"), b"") or b""
    dna = st.session_state.get(_key(prefix, "dna"), "") or ""
    mapping = st.session_state.get(_key(prefix, "mapping"), "—")
    ecc_meta = st.session_state.get(_key(prefix, "ecc_meta"), {}) or {}
    rows = [
        {"Property": "Compression method", "Value": "No compression"},
        {"Property": "Stored size", "Value": fmt_bytes(len(stored))},
        {"Property": "Encoding payload size", "Value": fmt_bytes(len(payload))},
        {"Property": "DNA design rule", "Value": display_mapping(mapping)},
        {"Property": "ECC option", "Value": "Reed–Solomon" if ecc_enabled else "None"},
        {"Property": "DNA length", "Value": f"{len(clean_dna(dna)):,} nt" if dna else "—"},
    ]
    if ecc_enabled:
        rows.append({"Property": "ECC overhead", "Value": f"{ecc_meta.get('ecc_overhead_ratio', 1.0):.3f}×"})
    return rows


def _decoded_summary_rows(prefix: str, ecc_enabled: bool) -> List[Dict[str, Any]]:
    original = st.session_state.get(_key(prefix, "stored_bytes"), b"") or b""
    decoded = st.session_state.get(_key(prefix, "decoded_data"), b"") or b""
    md = st.session_state.get(_key(prefix, "decoded_magic"), {}) or {}
    rr = st.session_state.get(_key(prefix, "rs_report"), {}) or {}
    rows = [
        {"Property": "Decoded type", "Value": md.get("kind", "unknown")},
        {"Property": "Decoded size", "Value": fmt_bytes(len(decoded))},
        {"Property": "Byte accuracy", "Value": f"{byte_accuracy(original, decoded):.6f}" if decoded else "—"},
        {"Property": "SHA256 match", "Value": "Yes" if original and decoded and sha256_bytes(original) == sha256_bytes(decoded) else "No"},
        {"Property": "Readable / valid", "Value": "Yes" if st.session_state.get(_key(prefix, "decoded_valid")) else "No"},
    ]
    if ecc_enabled:
        rows.insert(3, {"Property": "RS failed blocks", "Value": rr.get("failed_blocks", "—")})
        rows.insert(3, {"Property": "RS corrected symbols", "Value": rr.get("corrected_symbols", 0)})
    return rows


def _compression_analysis_rows(prefix: str, ecc_enabled: bool) -> List[Dict[str, Any]]:
    original = st.session_state.get(_key(prefix, "stored_bytes"), b"") or b""
    payload = st.session_state.get(_key(prefix, "payload_bytes"), b"") or b""
    return [
        {"Property": "Compression mode", "Value": "No compression"},
        {"Property": "Original container size", "Value": fmt_bytes(len(original))},
        {"Property": "Compressed/encoded payload size", "Value": fmt_bytes(len(payload))},
        {"Property": "Size ratio", "Value": f"{len(original) / max(1, len(payload)):.3f}×" if payload else "—"},
        {"Property": "Note", "Value": "Container bytes are preserved; no raw conversion is used."},
    ]


def _encode_decode_analysis_rows(prefix: str, ecc_enabled: bool) -> List[Dict[str, Any]]:
    original = st.session_state.get(_key(prefix, "stored_bytes"), b"") or b""
    decoded = st.session_state.get(_key(prefix, "decoded_data"), b"") or b""
    dna = st.session_state.get(_key(prefix, "dna"), "") or ""
    strand_rows = st.session_state.get(_key(prefix, "strand_rows"), []) or []
    mapping = st.session_state.get(_key(prefix, "mapping"), "—")
    hp = homopolymer_stats(dna)
    return [
        {"Property": "DNA design rule", "Value": display_mapping(mapping)},
        {"Property": "DNA length", "Value": f"{len(clean_dna(dna)):,} nt" if dna else "—"},
        {"Property": "GC content", "Value": f"{gc_content(dna):.6f}" if dna else "—"},
        {"Property": "Longest homopolymer", "Value": hp.get("longest", "—") if dna else "—"},
        {"Property": "Designed strands", "Value": f"{len(strand_rows):,}"},
        {"Property": "Byte accuracy", "Value": f"{byte_accuracy(original, decoded):.6f}" if decoded else "—"},
        {"Property": "Byte mismatches", "Value": byte_distance(original, decoded) if decoded else "—"},
    ]


def _summary_rows(prefix: str, ecc_enabled: bool) -> List[Dict[str, Any]]:
    original = st.session_state.get(_key(prefix, "stored_bytes"), b"") or b""
    payload = st.session_state.get(_key(prefix, "payload_bytes"), b"") or b""
    decoded_payload = st.session_state.get(_key(prefix, "decoded_payload_bytes"), b"") or b""
    decoded = st.session_state.get(_key(prefix, "decoded_data"), b"") or b""
    dna = st.session_state.get(_key(prefix, "dna"), "") or ""
    noisy = st.session_state.get(_key(prefix, "noisy_dna"), "") or ""
    mapping = st.session_state.get(_key(prefix, "mapping"), "—")
    valid = bool(st.session_state.get(_key(prefix, "decoded_valid")))
    rr = st.session_state.get(_key(prefix, "rs_report"), {}) or {}
    em = st.session_state.get(_key(prefix, "error_metrics"), {}) or {}
    ecc_meta = st.session_state.get(_key(prefix, "ecc_meta"), {}) or {}

    rows: List[Dict[str, Any]] = [
        {"Property": "Mode", "Value": "Reed–Solomon ECC" if ecc_enabled else "No ECC baseline"},
        {"Property": "DNA design rule", "Value": display_mapping(mapping)},
        {"Property": "ECC option", "Value": "Reed–Solomon" if ecc_enabled else "None"},
        {"Property": "Original file size", "Value": fmt_bytes(len(original))},
        {"Property": "Encoding payload size", "Value": fmt_bytes(len(payload))},
        {"Property": "DNA length", "Value": f"{len(dna):,} nt"},
        {"Property": "Added DNA errors", "Value": em.get("total_errors", 0)},
        {"Property": "DNA accuracy after errors", "Value": f"{string_accuracy(dna, noisy):.6f}" if noisy else "—"},
    ]
    if ecc_enabled:
        rows.extend([
            {"Property": "ECC overhead", "Value": f"{ecc_meta.get('ecc_overhead_ratio', 1.0):.3f}×"},
            {"Property": "RS corrected symbols", "Value": rr.get("corrected_symbols", 0)},
            {"Property": "RS failed blocks", "Value": rr.get("failed_blocks", "—")},
            {"Property": "Protected byte accuracy before RS", "Value": f"{byte_accuracy(payload, decoded_payload):.6f}" if decoded_payload else "—"},
            {"Property": "Recovered byte accuracy after RS", "Value": f"{byte_accuracy(original, decoded):.6f}" if decoded else "—"},
        ])
    else:
        rows.extend([
            {"Property": "Binary accuracy", "Value": f"{string_accuracy(bytes_to_bitstring(original), bytes_to_bitstring(decoded)):.6f}" if decoded else "—"},
            {"Property": "Byte accuracy", "Value": f"{byte_accuracy(original, decoded):.6f}" if decoded else "—"},
            {"Property": "Byte mismatches", "Value": byte_distance(original, decoded) if decoded else "—"},
        ])
    rows.extend([
        {"Property": "SHA256 match", "Value": "Yes" if original and decoded and sha256_bytes(original) == sha256_bytes(decoded) else "No"},
        {"Property": "Readable / valid", "Value": "Yes" if valid else "No"},
    ])
    return rows



def _encoding_statistics_rows(prefix: str, ecc_enabled: bool) -> List[Dict[str, Any]]:
    original = st.session_state.get(_key(prefix, "stored_bytes"), b"") or b""
    payload = st.session_state.get(_key(prefix, "payload_bytes"), b"") or b""
    dna = st.session_state.get(_key(prefix, "dna"), "") or ""
    mapping = st.session_state.get(_key(prefix, "mapping"), "—")
    ecc_meta = st.session_state.get(_key(prefix, "ecc_meta"), {}) or {}
    hp = homopolymer_stats(dna)
    rows = [
        {"Group": "Encoding statistics", "Property": "Storage method", "Value": "No compression"},
        {"Group": "Encoding statistics", "Property": "Original file size", "Value": fmt_bytes(len(original))},
        {"Group": "Encoding statistics", "Property": "DNA design rule", "Value": display_mapping(mapping)},
        {"Group": "Encoding statistics", "Property": "ECC option", "Value": "Reed–Solomon" if ecc_enabled else "None"},
        {"Group": "Encoding statistics", "Property": "Encoding payload size", "Value": fmt_bytes(len(payload))},
        {"Group": "Encoding statistics", "Property": "DNA length", "Value": f"{len(dna):,} nt"},
        {"Group": "Encoding statistics", "Property": "DNA expansion vs original", "Value": f"{len(dna) / max(1, len(original) * 4):.3f}×"},
        {"Group": "Encoding statistics", "Property": "GC content", "Value": f"{gc_content(dna):.6f}"},
        {"Group": "Encoding statistics", "Property": "Longest homopolymer", "Value": hp.get("longest", 0)},
        {"Group": "Encoding statistics", "Property": "Homopolymer segments ≥2", "Value": hp.get("count_ge2", 0)},
    ]
    if ecc_enabled:
        rows.extend([
            {"Group": "Encoding statistics", "Property": "RS data block size", "Value": ecc_meta.get("data_block_size", "—")},
            {"Group": "Encoding statistics", "Property": "RS parity bytes/block", "Value": ecc_meta.get("parity_bytes", "—")},
            {"Group": "Encoding statistics", "Property": "ECC overhead", "Value": f"{ecc_meta.get('ecc_overhead_ratio', 1.0):.3f}×"},
            {"Group": "Encoding statistics", "Property": "Max unknown byte errors/block", "Value": ecc_meta.get("max_unknown_byte_errors_per_block", "—")},
        ])
    return rows


def _error_statistics_rows(prefix: str) -> List[Dict[str, Any]]:
    dna = st.session_state.get(_key(prefix, "dna"), "") or ""
    noisy = st.session_state.get(_key(prefix, "noisy_dna"), "") or ""
    em = st.session_state.get(_key(prefix, "error_metrics"), {}) or {}
    return [
        {"Group": "Error Adding Report", "Property": "Added DNA errors", "Value": em.get("total_errors", 0)},
        {"Group": "Error Adding Report", "Property": "Substitutions", "Value": em.get("substitutions", 0)},
        {"Group": "Error Adding Report", "Property": "Insertions", "Value": em.get("insertions", 0)},
        {"Group": "Error Adding Report", "Property": "Deletions", "Value": em.get("deletions", 0)},
        {"Group": "Error Adding Report", "Property": "Original DNA length", "Value": f"{len(dna):,} nt"},
        {"Group": "Error Adding Report", "Property": "Noisy DNA length", "Value": f"{len(noisy):,} nt" if noisy else "—"},
        {"Group": "Error Adding Report", "Property": "DNA accuracy after errors", "Value": f"{string_accuracy(dna, noisy):.6f}" if noisy else "—"},
        {"Group": "Error Adding Report", "Property": "DNA mismatches / distance", "Value": hamming_distance_str(dna, noisy) if noisy else "—"},
    ]


def _decode_statistics_rows(prefix: str, ecc_enabled: bool) -> List[Dict[str, Any]]:
    original = st.session_state.get(_key(prefix, "stored_bytes"), b"") or b""
    payload = st.session_state.get(_key(prefix, "payload_bytes"), b"") or b""
    decoded_payload = st.session_state.get(_key(prefix, "decoded_payload_bytes"), b"") or b""
    decoded = st.session_state.get(_key(prefix, "decoded_data"), b"") or b""
    valid = bool(st.session_state.get(_key(prefix, "decoded_valid")))
    note = st.session_state.get(_key(prefix, "decoded_note"), "")
    rr = st.session_state.get(_key(prefix, "rs_report"), {}) or {}
    rows = [
        {"Group": "Decode / Recovery statistics", "Property": "Decoded payload size", "Value": fmt_bytes(len(decoded_payload)) if decoded_payload else "—"},
        {"Group": "Decode / Recovery statistics", "Property": "Decoded output size", "Value": fmt_bytes(len(decoded)) if decoded else "—"},
        {"Group": "Decode / Recovery statistics", "Property": "Readable / valid", "Value": "Yes" if valid else "No"},
        {"Group": "Decode / Recovery statistics", "Property": "Validation note", "Value": note or "—"},
    ]
    if ecc_enabled:
        rows.extend([
            {"Group": "Decode / Recovery statistics", "Property": "RS corrected symbols", "Value": rr.get("corrected_symbols", 0)},
            {"Group": "Decode / Recovery statistics", "Property": "RS failed blocks", "Value": rr.get("failed_blocks", "—")},
            {"Group": "Decode / Recovery statistics", "Property": "Protected byte accuracy before RS", "Value": f"{byte_accuracy(payload, decoded_payload):.6f}" if decoded_payload else "—"},
            {"Group": "Decode / Recovery statistics", "Property": "Recovered byte accuracy after RS", "Value": f"{byte_accuracy(original, decoded):.6f}" if decoded else "—"},
        ])
    else:
        rows.extend([
            {"Group": "Decode / Recovery statistics", "Property": "Binary accuracy", "Value": f"{string_accuracy(bytes_to_bitstring(original), bytes_to_bitstring(decoded)):.6f}" if decoded else "—"},
            {"Group": "Decode / Recovery statistics", "Property": "Byte accuracy", "Value": f"{byte_accuracy(original, decoded):.6f}" if decoded else "—"},
            {"Group": "Decode / Recovery statistics", "Property": "Byte mismatches", "Value": byte_distance(original, decoded) if decoded else "—"},
        ])
    rows.append({"Group": "Decode / Recovery statistics", "Property": "SHA256 match", "Value": "Yes" if original and decoded and sha256_bytes(original) == sha256_bytes(decoded) else "No"})
    return rows


def render_panel_6_validation(prefix: str, ecc_enabled: bool) -> None:
    with st.container(border=True):
        step_header(6, PANEL_TITLES["validation"])
        decoded = st.session_state.get(_key(prefix, "decoded_data"))
        if decoded is None:
            st.info("Run Decode first.")
            return

        original = st.session_state.get(_key(prefix, "stored_bytes"), b"") or b""
        stored_path = st.session_state.get(_key(prefix, "stored_path"), "")
        input_bytes = st.session_state.get("input_bytes", b"") or b""
        input_path = st.session_state.get("input_path", "")
        input_name = st.session_state.get("input_name", "")
        restored_path = st.session_state.get(_key(prefix, "restored_path"), "")
        dna = st.session_state.get(_key(prefix, "dna"), "") or ""

        st.markdown("#### 📊 Summary")
        original_col, encoded_col, decoded_col = st.columns(3, gap="large")
        with original_col:
            # st.markdown("##### Original")
            if input_path and input_bytes:
                preview_file_streamlit(st, input_path, "Original preview", key_suffix=_key(prefix, "summary_original"))
                _render_property_table(_basic_file_rows(input_path, input_bytes, label="Original file"))
            else:
                st.info("Upload a file first.")
        with encoded_col:
            # st.markdown("##### Compressed / Encoded")
            if stored_path and original:
                preview_file_streamlit(st, stored_path, "Encoded preview", key_suffix=_key(prefix, "summary_encoded_file"))
            elif dna:
                st.text_area(
                    "Encoded DNA preview",
                    _preview_seq(dna, 900),
                    height=220,
                    key=_content_key(prefix, "summary_encoded_dna_preview", dna),
                )
            _render_property_table(_encoded_summary_rows(prefix, ecc_enabled))
        with decoded_col:
            # st.markdown("##### Decoded")
            if restored_path:
                preview_file_streamlit(st, restored_path, "Decoded preview", key_suffix=_key(prefix, "summary_decoded"))
            _render_property_table(_decoded_summary_rows(prefix, ecc_enabled))

        st.markdown("#### 🧾 Compression analysis")
        _render_property_table(_compression_analysis_rows(prefix, ecc_enabled))

        st.markdown("#### 🧬 Encode-decode analysis")
        _render_property_table(_encode_decode_analysis_rows(prefix, ecc_enabled))

        st.markdown("#### ⚠️ Error Adding Report")
        err_df = pd.DataFrame(_error_statistics_rows(prefix))
        st.dataframe(err_df[["Property", "Value"]], use_container_width=True, hide_index=True)

        st.markdown("#### 🔁 Decode / Recovery Report")
        dec_df = pd.DataFrame(_decode_statistics_rows(prefix, ecc_enabled))
        st.dataframe(dec_df[["Property", "Value"]], use_container_width=True, hide_index=True)

        st.markdown("#### ✅ Recovery Quality Report")
        quality_rows = quality_metric_rows(
            original,
            decoded,
            input_name=input_name,
            input_path=input_path,
            decoded_path=restored_path,
        )
        qdf = pd.DataFrame(quality_rows)
        # Keep the report visible, but compact enough for the final panel.
        st.dataframe(qdf, use_container_width=True, hide_index=True)

        st.markdown("#### 🧾 Final summary")
        summary = pd.DataFrame(_summary_rows(prefix, ecc_enabled))
        st.dataframe(summary, use_container_width=True, hide_index=True)

        combined_rows = []
        for rows in [
            _compression_analysis_rows(prefix, ecc_enabled),
            _encode_decode_analysis_rows(prefix, ecc_enabled),
            _error_statistics_rows(prefix),
            _decode_statistics_rows(prefix, ecc_enabled),
            _summary_rows(prefix, ecc_enabled),
        ]:
            combined_rows.extend(rows)
        combined_csv = pd.DataFrame(combined_rows).to_csv(index=False).encode("utf-8")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                BUTTONS["download_summary"],
                data=combined_csv,
                file_name=f"{prefix}_summary.csv",
                mime="text/csv",
                use_container_width=True,
                key=_key(prefix, "download_summary"),
            )
        with col2:
            st.download_button(
                "Download quality report CSV",
                data=qdf.to_csv(index=False).encode("utf-8"),
                file_name=f"{prefix}_quality_report.csv",
                mime="text/csv",
                use_container_width=True,
                key=_key(prefix, "download_quality_report"),
            )

def render_pipeline(prefix: str, ecc_enabled: bool) -> None:
    render_stepper(prefix)
    render_panel_1_upload(prefix)
    render_panel_2_data_encoding(prefix, ecc_enabled=ecc_enabled)
    render_panel_3_dna_encoding(prefix, ecc_enabled=ecc_enabled)
    render_panel_4_errors(prefix, ecc_enabled=ecc_enabled)
    render_panel_5_decoding(prefix, ecc_enabled=ecc_enabled)
    render_panel_6_validation(prefix, ecc_enabled=ecc_enabled)
