# Third-party notices

RoadSense-U's authored source uses **AGPL-3.0-only** (see LICENSE). Third-party dependencies retain their licenses; this file does not relicense them. This is an inventory for a research source release, not certification of every possible binary distribution. Model weights, videos and other datasets are not bundled in the public source or source distributions.

## Principal direct dependencies

| Component | Upstream license information | Usage |
|---|---|---|
| [Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) | AGPL-3.0 / separate commercial terms; see [vendor terms](https://www.ultralytics.com/license) | Optional pretrained detector and ByteTrack integration. Public source availability does not grant a separate enterprise license. |
| [Original ByteTrack](https://github.com/ifzhang/ByteTrack/blob/main/LICENSE) | MIT in the original repository | Tracking research reference. This project imports the Ultralytics implementation, whose terms also apply. No original ByteTrack source is vendored. |
| [OpenCV](https://github.com/opencv/opencv/blob/4.x/LICENSE) | Apache-2.0 for current OpenCV; Python wheel distributions carry additional third-party notices | Video I/O and geometry. Review wheel codec/FFmpeg notices before redistributing binaries. |
| [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt), [SciPy](https://github.com/scipy/scipy/blob/main/LICENSE.txt), [pandas](https://github.com/pandas-dev/pandas/blob/main/LICENSE), [scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | Primarily BSD-style, with bundled component notices | Numerical processing and offline evaluation. |
| [FastAPI](https://github.com/fastapi/fastapi/blob/master/LICENSE), [Pydantic](https://github.com/pydantic/pydantic/blob/main/LICENSE), [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy/blob/main/LICENSE), [HTTPX](https://github.com/encode/httpx/blob/master/LICENSE.md) | MIT/BSD-style project licenses; see each upstream | API, validation, persistence and HTTP client. |
| [Psycopg](https://www.psycopg.org/psycopg3/docs/basic/install.html) | LGPL-3.0 project licensing; binary distributions include separately licensed libraries | PostgreSQL driver; review included libpq/OpenSSL and redistribution requirements. |
| [Streamlit](https://github.com/streamlit/streamlit/blob/develop/LICENSE) | Apache-2.0 | Optional local dashboard. |
| [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) / [torchvision](https://github.com/pytorch/vision/blob/main/LICENSE) | BSD-style project licenses plus third-party notices | Optional CPU/GPU detection runtime. |
| [PyYAML](https://github.com/yaml/pyyaml/blob/main/LICENSE), [joblib](https://github.com/joblib/joblib/blob/main/LICENSE.txt), [psutil](https://github.com/giampaolo/psutil/blob/master/LICENSE), [Uvicorn](https://github.com/encode/uvicorn/blob/main/LICENSE.md), [lap](https://github.com/gatagat/lap/blob/master/LICENSE) | MIT/BSD-style project licenses; preserve package notices | Configuration, local artifacts, measurements, serving and optional association dependency. |

These links identify upstream projects; exact installed package metadata and bundled notices govern the selected version. A package advisory scan is separate from a license review. Installed transitive dependencies and container base images must also be inventoried when distributing a binary or image; this source inventory does not establish all obligations for that deployment.

## Documentation and optional test tools

The README gallery consists of actual browser screenshots of this application's generated synthetic fixture and computed results. The captures include no third-party traffic footage or downloaded detector example images. Their capture provenance and hashes are in `docs/assets/provenance.json`.

[Altair](https://github.com/vega/altair/blob/main/LICENSE) supplies dashboard charts under BSD-3-Clause terms. Optional [Playwright for Python](https://github.com/microsoft/playwright-python/blob/main/LICENSE) supplies browser automation under Apache-2.0. Browser binaries are installed separately under their upstream terms and are not redistributed in the source release.

## External artifacts and research

The explicit weight downloader retrieves one official YOLO11n artifact, records its source and verifies the pinned SHA256 from the initially reviewed upstream download. The pin is not an independent author signature. Upstream licensing applies to weights. The detector smoke helper retrieves an official example image only on request; the generated repeated-image fixture is not a motion benchmark and is not included in release artifacts.

Dataset access and publication do not imply redistribution permission. Nexar, TU-DAT, TUMTraf-V2X, BrnoCompSpeed and DoTA have separate terms and/or unresolved footage permissions: see docs/DATASETS.md. Literature citations identify prior work and do not imply this project has reproduced it or obtained rights to redistribute its assets.

The Bellevue regression fixture contains numeric bounding-box measurements and a single-reviewer object-distinctness assessment from local testing, with a source checksum. It includes no frames, clips, model weights, or independently validated conflict labels. Source-video redistribution rights have not been verified; the video and review stills remain outside the release.

Optional contact-sheet export uses Matplotlib, distributed under its own PSF-derived Matplotlib license (see the installed distribution's license text). It is an optional review dependency, not a license for source videos.
