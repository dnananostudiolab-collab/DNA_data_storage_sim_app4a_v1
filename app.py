from __future__ import annotations

import streamlit as st

from panels import apply_app_style, render_pipeline

APP_TITLE = "DNA Error Simulation and Reed–Solomon ECC Recovery Pipeline"


def _render_hero() -> None:
    st.markdown(
        """
<div class="hero-card">
  <div class="hero-title">🧬 DNA Error Simulation and Reed–Solomon ECC Recovery Pipeline</div>
  <div class="hero-subtitle">Mode: File-container byte storage</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_app() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")
    apply_app_style()
    _render_hero()

    
    tab1, tab2 = st.tabs([
    "No ECC — File-container",
    "RS-ECC Recovery — File-container",
])
    with tab1:
        # st.info(
        #     "No compression + SM/R∞ mapping + DNA errors + decode. "
        #     "File may not open, but binary/DNA/byte accuracy is still evaluated."
        # )
        render_pipeline(prefix="base", ecc_enabled=False)
    with tab2:
        # st.info(
        #     "No compression + Reed–Solomon byte-level ECC + SM/R∞ mapping + DNA errors + decode + RS repair. "
        #     "This branch tests whether the original file container can be recovered."
        # )
        render_pipeline(prefix="ecc", ecc_enabled=True)


if __name__ == "__main__":
    render_app()
