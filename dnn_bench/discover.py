"""Thin re-export of the self-contained paper_to_spec module.

The implementation lives in paper_to_spec.py, which has no package imports and
no torch dependency, so it can be copied into another project as a single file.
Keeping one implementation means the standalone copy cannot drift from the one
the bench uses.
"""
from .paper_to_spec import (      # noqa: F401
    Hit, discover, search_epmc, search_arxiv, references_of, enrich,
    full_text, full_text_alternates, score_fit, build_queries, key_terms, key_phrases, topical,
    CAVEAT, UA, EPMC, HARD, SOFT, OBSERVATIONAL,
)
