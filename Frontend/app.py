"""Project dashboard entry point.

Run from inside the Frontend folder (so .streamlit/config.toml, the theme, is loaded):
    cd Frontend
    streamlit run app.py

This file only registers pages. Each component's page lives in
Frontend/components/<component>/page.py and is owned by that component's member.
"""

from __future__ import annotations

import streamlit as st

from components.c1_encoding import page as c1
from components.c2_global_snn import page as c2
from components.c3_adapter import page as c3
from components.c4_xai import page as c4
from shared import home
from shared.registry import component
from shared.style import apply_style

st.set_page_config(page_title="J26-DS-348 SNN Depression", page_icon=":material/neurology:", layout="wide")
apply_style()
if not st.get_option("theme.sidebar.backgroundColor"):
    # Streamlit reads .streamlit/config.toml from the launch folder only.
    st.sidebar.warning("Theme not loaded: run `cd Frontend` then `streamlit run app.py`.")


def _page(module, key: str) -> st.Page:
    info = component(key)
    return st.Page(module.render, title=f"{key.upper()} · {info.title}", icon=info.icon, url_path=info.folder)


navigation = st.navigation(
    {
        "Project": [st.Page(home.render, title="Overview", icon=":material/home:", url_path="overview", default=True)],
        "Components": [_page(c1, "c1"), _page(c2, "c2"), _page(c3, "c3"), _page(c4, "c4")],
    }
)
navigation.run()
