"""Smoke tests: the dashboard and every component page render without errors."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

FRONTEND = Path(__file__).resolve().parents[1] / "Frontend"


def _render_page(frontend: str, module: str) -> None:
    import importlib
    import sys

    sys.path.insert(0, frontend)
    importlib.import_module(module).render()


def test_app_starts_on_the_overview_page():
    app = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=60).run()

    assert not app.exception
    assert app.title and app.title[0].value.strip()


@pytest.mark.parametrize(
    "module",
    [
        "components.c1_encoding.page",
        "components.c2_global_snn.page",
        "components.c3_adapter.page",
        "components.c4_xai.page",
    ],
)
def test_every_component_page_renders(module):
    app = AppTest.from_function(_render_page, args=(str(FRONTEND), module), default_timeout=60).run()

    assert not app.exception
    assert len(app.tabs) >= 4  # the four standard tabs; a page may add its own
