# conda env: starch_vit
"""Shared path config.

This repo (F:/starch) holds the ML pipeline only. The raw images and
per-granule measurement CSV live in a separate data repo (F:/datasets/starch)
that this pipeline treats as read-only. Override with STARCH_DATA_ROOT if
that data ever moves.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("STARCH_DATA_ROOT", "F:/datasets/starch")
PHOTOS_ROOT = os.path.join(DATA_ROOT, "MS_NTP_Photos", "MS_NTP_Photos")
