"""Global pytest configuration and shared fixtures.

Adds the project root to sys.path so that absolute imports (e.g.
``from core.coordinates import ...``) work correctly without installing
the package.
"""

import sys
import os

# Make the project root importable
sys.path.insert(0, os.path.dirname(__file__))
