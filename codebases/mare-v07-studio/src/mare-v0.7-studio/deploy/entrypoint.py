"""Read optional Docker secret files without including their values in process arguments."""
import os
from pathlib import Path
import sys

for name in ("MARE_AUTH_TOKEN", "OPENAI_API_KEY", "MARE_DATABASE_URL"):
    source = os.environ.get(name + "_FILE")
    if source:
        os.environ[name] = Path(source).read_text().strip()
os.execvp(sys.argv[1], sys.argv[1:])
