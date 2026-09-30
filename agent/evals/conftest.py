"""Put each layer's directory on sys.path so the tests can import the scripts
by module name (the scripts themselves import siblings the same way)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for sub in ("harness", "behaviour", "benchmark", "loop"):
    p = str(HERE / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
