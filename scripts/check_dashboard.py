# SPDX-License-Identifier: AGPL-3.0-only
"""Render Streamlit against the actual local API and assert stored synthetic events are visible."""

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

from roadsense.config import Settings

if __name__ == "__main__":
    cfg = Settings()
    os.environ["ROADSENSE_API_KEY"] = cfg.api_key
    os.environ["ROADSENSE_API_URL"] = cfg.api_url
    app = AppTest.from_file(str(Path("src/roadsense/dashboard/app.py").resolve()), default_timeout=30).run()
    assert not app.exception, app.exception
    assert not app.error, app.error
    app.sidebar.checkbox[0].check().run()
    assert not app.exception, app.exception
    assert not app.error, app.error
    assert app.dataframe, "No stored event/trajectory tables rendered"
    assert any("API and database connected" in item.value for item in app.success)
    print("Dashboard rendered actual API health, stored synthetic event table and local artifacts.")
