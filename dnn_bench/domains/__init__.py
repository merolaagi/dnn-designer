"""Domain plugins.

Importing this module registers the coded domains below, then every *.json in
the top-level specs/ directory. A declared domain is preferred to a coded one:
its Jacobian is derived from its residual and so cannot contradict it.
"""
import os

from . import crn, contraction, convex          # noqa: F401
from ..spec import load_spec_dir

SPEC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'specs')
loaded_specs = load_spec_dir(SPEC_DIR)
