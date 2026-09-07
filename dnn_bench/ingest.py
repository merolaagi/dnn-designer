"""Thin re-export of the self-contained paper_to_spec module. See discover.py."""
from .paper_to_spec import (      # noqa: F401
    Paper, Passage, ingest, blank_spec, SPEC_TEMPLATE,
)
