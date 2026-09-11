"""Regenerate the checked-in API contract before `npm run schemas --prefix web`."""
import json
from pathlib import Path
from mare_web.api import create_app
from mare_web.config import Settings

schema = create_app(Settings(environment="test", _env_file=None)).openapi()
Path("docs/openapi.json").write_text(json.dumps(schema, indent=2) + "\n")
