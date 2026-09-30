"""Make the module root importable (the evaluators use flat top-level imports
such as `from simple_scorer import SimpleScorer`) regardless of where pytest
is launched from."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
