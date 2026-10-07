"""Shared pytest setup.

The model_service and webhook_forwarder apps use bare imports (``import main``,
``import database``, ``import app``) because inside their Docker containers each
app's own directory is the working directory. To import them from the repo root
during testing, put those directories on ``sys.path``.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for path in (
    os.path.join(ROOT, "model_service"),
    os.path.join(ROOT, "tools", "webhook_forwarder"),
):
    if path not in sys.path:
        sys.path.insert(0, path)
