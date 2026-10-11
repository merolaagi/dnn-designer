"""Ready-made MCP servers for the Agent lab: the command that starts each, and the tools to offer.

A server can list far more tools than a model should see at once (REA lists 139, about
600 KB of schemas): each preset offers a focused subset. The Inspector's Discover button
lists the rest.
"""

import json

# REA: Reverse Engineer Anything (github.com/morluto/rea, MIT), rea-agents 6.3.0.
# Native binaries need Ghidra, Hopper or IDA on this computer; JavaScript/Electron apps,
# .NET assemblies, packages and plists need nothing else.
REA_COMMAND = "npx -y rea-agents@latest mcp"
REA_TOOLS = json.loads(r"""[
 {
  "name": "open_binary",
  "description": "Open a local executable, application bundle, archive, JavaScript, source map, plist, or analysis database after validation. format=dos-com explicitly interprets 1..65280 headerless bytes as a DOS COM analysis image; omission preserves header-based detection. provider_id selects one deep provider or deterministic auto selection; the binding remains stable until close or an explicit switch, with no failure fallback. Reopening the same target and profile retains its active database and metadata edits. An optional analysis snapshot is imported atomically and must match the binary identity, concrete provider, and canonical analysis profile exactly; importing into a session with metadata mutations requires closing and reopening it first.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "path": {
     "$ref": "#/$defs/__schema0",
     "description": "Absolute local filesystem path for the analysis target; relative paths are rejected."
    },
    "format": {
     "$ref": "#/$defs/__schema2",
     "description": "Explicit headerless DOS COM interpretation; omission preserves header-based detection"
    },
    "provider_id": {
     "$ref": "#/$defs/__schema4",
     "description": "Exact deep-analysis provider ID, or automatic selection when omitted."
    },
    "snapshot_path": {
     "$ref": "#/$defs/__schema5",
     "description": "Absolute local filesystem path for the analysis snapshot; relative paths are rejected."
    }
   },
   "required": [
    "path"
   ],
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "type": "string",
     "minLength": 1,
     "pattern": "^[^\\x00]*$",
     "description": "Absolute local filesystem path for the analysis target; relative paths are rejected.",
     "$ref": "#/$defs/__schema1"
    },
    "__schema1": {
     "type": "string",
     "minLength": 1,
     "pattern": "^[^\\x00]*$"
    },
    "__schema2": {
     "description": "Explicit headerless DOS COM interpretation; omission preserves header-based detection",
     "$ref": "#/$defs/__schema3"
    },
    "__schema3": {
     "type": "string",
     "const": "dos-com"
    },
    "__schema4": {
     "anyOf": [
      {
       "type": "string",
       "const": "auto"
      },
      {
       "type": "string",
       "minLength": 1,
       "pattern": "^[a-z][a-z0-9\\x2d]*$"
      }
     ]
    },
    "__schema5": {
     "type": "string",
     "minLength": 1,
     "pattern": "^[^\\x00]*$",
     "description": "Absolute local filesystem path for the analysis snapshot; relative paths are rejected.",
     "$ref": "#/$defs/__schema1"
    }
   },
   "examples": [
    {
     "path": "/tmp/fixture"
    }
   ]
  }
 },
 {
  "name": "close_binary",
  "description": "Drain earlier provider requests, optionally write a provider-neutral analysis snapshot atomically to the caller-supplied path, then close the active target and every provider resource started for it. Later provider requests wait until this lifecycle operation finishes. Existing files require explicit overwrite; a failed save leaves the session open so cached analysis is not lost. Metadata edits prevent immutable snapshot saves until the session is recreated; export_evidence_bundle preserves mutable observations before closing without a snapshot.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "snapshot_path": {
     "$ref": "#/$defs/__schema0",
     "description": "Absolute local filesystem path for the analysis snapshot; relative paths are rejected."
    },
    "overwrite": {
     "$ref": "#/$defs/__schema1",
     "description": "Whether an existing destination may be replaced."
    }
   },
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "type": "string",
     "minLength": 1,
     "pattern": "^[^\\x00]*$",
     "description": "Absolute local filesystem path for the analysis snapshot; relative paths are rejected."
    },
    "__schema1": {
     "default": false,
     "type": "boolean"
    }
   },
   "examples": [
    {
     "overwrite": false
    }
   ]
  }
 },
 {
  "name": "binary_session",
  "description": "Report the complete current target, provider, capability availability, client-feature, analysis, and server-identity status without starting analysis.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "expected_package_version": {
     "$ref": "#/$defs/__schema0",
     "description": "Expected package version used to reject stale or mismatched input."
    },
    "expected_catalog_digest": {
     "$ref": "#/$defs/__schema1",
     "description": "Expected catalog digest used to reject stale or mismatched input."
    },
    "expected_server_path": {
     "$ref": "#/$defs/__schema2",
     "description": "Absolute local filesystem path expected for the running server; relative paths are rejected."
    }
   },
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "type": "string",
     "minLength": 1
    },
    "__schema1": {
     "type": "string",
     "pattern": "^[a-f0-9]{64}$"
    },
    "__schema2": {
     "type": "string",
     "minLength": 1,
     "pattern": "^[^\\x00]*$",
     "description": "Absolute local filesystem path expected for the running server; relative paths are rejected."
    }
   },
   "examples": [
    {}
   ]
  }
 },
 {
  "name": "binary_overview",
  "description": "Return native-binary metadata, every segment with its length, and exhaustive procedure/string counts. Use search_procedures, list_procedures, or analyze_function directly when you already know which procedure to inspect.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {},
   "additionalProperties": false,
   "examples": [
    {}
   ]
  }
 },
 {
  "name": "search_strings",
  "description": "Search every analyzed string object using literal matching by default or regex when requested. Results are deterministic and complete over the native inventory. Hopper retains typed byte decoding, native display, and explicit decoding unknowns. Adjacent long-literal fragments match independently. Hopper regex mode uses ECMAScript Unicode syntax in a cancellable worker; a matching deadline leaves its API usable.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "pattern": {
     "$ref": "#/$defs/__schema0",
     "description": "The literal text or regex pattern to search for"
    },
    "mode": {
     "$ref": "#/$defs/__schema1",
     "description": "Operation mode that selects the requested behavior."
    },
    "case_sensitive": {
     "$ref": "#/$defs/__schema2",
     "description": "Whether to match case"
    },
    "document": {
     "$ref": "#/$defs/__schema4",
     "description": "The document name"
    }
   },
   "required": [
    "pattern"
   ],
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "type": "string",
     "minLength": 1,
     "description": "The literal text or regex pattern to search for"
    },
    "__schema1": {
     "default": "literal",
     "type": "string",
     "enum": [
      "literal",
      "regex"
     ]
    },
    "__schema2": {
     "default": false,
     "description": "Whether to match case",
     "$ref": "#/$defs/__schema3"
    },
    "__schema3": {
     "type": "boolean"
    },
    "__schema4": {
     "description": "The document name",
     "$ref": "#/$defs/__schema5"
    },
    "__schema5": {
     "type": "string"
    }
   },
   "examples": [
    {
     "pattern": "authorization failed",
     "mode": "literal",
     "case_sensitive": false
    }
   ]
  }
 },
 {
  "name": "search_procedures",
  "description": "Search every analyzed procedure name using literal matching by default or regex when requested. Results are deterministic and complete. Hopper regex mode uses ECMAScript Unicode syntax in a cancellable worker; a matching deadline leaves its API usable.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "pattern": {
     "$ref": "#/$defs/__schema0",
     "description": "The literal text or regex pattern to search for"
    },
    "mode": {
     "$ref": "#/$defs/__schema1",
     "description": "Operation mode that selects the requested behavior."
    },
    "case_sensitive": {
     "$ref": "#/$defs/__schema2",
     "description": "Whether to match case"
    },
    "document": {
     "$ref": "#/$defs/__schema4",
     "description": "The document name"
    }
   },
   "required": [
    "pattern"
   ],
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "type": "string",
     "minLength": 1,
     "description": "The literal text or regex pattern to search for"
    },
    "__schema1": {
     "default": "literal",
     "type": "string",
     "enum": [
      "literal",
      "regex"
     ]
    },
    "__schema2": {
     "default": false,
     "description": "Whether to match case",
     "$ref": "#/$defs/__schema3"
    },
    "__schema3": {
     "type": "boolean"
    },
    "__schema4": {
     "description": "The document name",
     "$ref": "#/$defs/__schema5"
    },
    "__schema5": {
     "type": "string"
    }
   },
   "examples": [
    {
     "pattern": "main",
     "mode": "literal",
     "case_sensitive": false
    }
   ]
  }
 },
 {
  "name": "procedure_pseudo_code",
  "description": "Decompile one analyzed procedure by symbol name or provider-normalized address. Returns provider-specific pseudocode, never original source or cross-provider text equivalence, and may return null; request procedure_assembly when instruction precision matters.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "procedure": {
     "$ref": "#/$defs/__schema0",
     "description": "The procedure name or address"
    },
    "document": {
     "$ref": "#/$defs/__schema1",
     "description": "The active target's provider document name; use open_binary to switch targets"
    }
   },
   "required": [
    "procedure"
   ],
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "type": "string",
     "description": "The procedure name or address"
    },
    "__schema1": {
     "description": "The active target's provider document name; use open_binary to switch targets",
     "$ref": "#/$defs/__schema2"
    },
    "__schema2": {
     "type": "string"
    }
   },
   "examples": [
    {
     "procedure": "main"
    }
   ]
  }
 },
 {
  "name": "procedure_assembly",
  "description": "Return assembly for one analyzed procedure identified by symbol or hexadecimal address. Use when pseudocode loses calling-convention or instruction-level detail; output is currently returned as one unpaginated string.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "procedure": {
     "$ref": "#/$defs/__schema0",
     "description": "The procedure name or address"
    },
    "document": {
     "$ref": "#/$defs/__schema1",
     "description": "The active target's provider document name; use open_binary to switch targets"
    }
   },
   "required": [
    "procedure"
   ],
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "type": "string",
     "description": "The procedure name or address"
    },
    "__schema1": {
     "description": "The active target's provider document name; use open_binary to switch targets",
     "$ref": "#/$defs/__schema2"
    },
    "__schema2": {
     "type": "string"
    }
   },
   "examples": [
    {
     "procedure": "main"
    }
   ]
  }
 },
 {
  "name": "procedure_callers",
  "description": "Return the provider's resolved direct callers for one procedure identified by symbol or address. Results reflect completed static analysis and may omit unresolved indirect references.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "procedure": {
     "$ref": "#/$defs/__schema0",
     "description": "The procedure name or address"
    },
    "document": {
     "$ref": "#/$defs/__schema1",
     "description": "The active target's provider document name; use open_binary to switch targets"
    }
   },
   "required": [
    "procedure"
   ],
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "type": "string",
     "description": "The procedure name or address"
    },
    "__schema1": {
     "description": "The active target's provider document name; use open_binary to switch targets",
     "$ref": "#/$defs/__schema2"
    },
    "__schema2": {
     "type": "string"
    }
   },
   "examples": [
    {
     "procedure": "main"
    }
   ]
  }
 },
 {
  "name": "procedure_callees",
  "description": "Return the provider's resolved direct callees for one procedure identified by symbol or address. Unresolved indirect calls may be absent; use analyze_function and typed references to preserve available edge uncertainty.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "procedure": {
     "$ref": "#/$defs/__schema0",
     "description": "The procedure name or address"
    },
    "document": {
     "$ref": "#/$defs/__schema1",
     "description": "The active target's provider document name; use open_binary to switch targets"
    }
   },
   "required": [
    "procedure"
   ],
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "type": "string",
     "description": "The procedure name or address"
    },
    "__schema1": {
     "description": "The active target's provider document name; use open_binary to switch targets",
     "$ref": "#/$defs/__schema2"
    },
    "__schema2": {
     "type": "string"
    }
   },
   "examples": [
    {
     "procedure": "main"
    }
   ]
  }
 },
 {
  "name": "xrefs",
  "description": "Return analyzed references to a code or data address. Hopper may default to its current cursor; headless providers such as Ghidra require an explicit address. Use to connect strings, globals, selectors, and functions; bare addresses are untyped and indirect references may be incomplete.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "document": {
     "$ref": "#/$defs/__schema0",
     "description": "The active target's provider document name; use open_binary to switch targets"
    },
    "address": {
     "$ref": "#/$defs/__schema2",
     "description": "A provider-normalized address; default memory uses 0x-prefixed hexadecimal"
    }
   },
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "description": "The active target's provider document name; use open_binary to switch targets",
     "$ref": "#/$defs/__schema1"
    },
    "__schema1": {
     "type": "string"
    },
    "__schema2": {
     "type": "string",
     "description": "A provider-normalized address; default memory uses 0x-prefixed hexadecimal"
    }
   },
   "examples": [
    {}
   ]
  }
 },
 {
  "name": "trace_feature",
  "description": "Trace the complete literal query, preserving spaces, through every matching string and procedure, their xrefs, and truthful containing-procedure resolution. Returns observations, operation count, and residual unknowns without inferring reference kinds.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "query": {
     "$ref": "#/$defs/__schema0",
     "description": "Non-empty feature or text query to investigate."
    },
    "case_sensitive": {
     "$ref": "#/$defs/__schema1",
     "description": "Whether text matching distinguishes uppercase and lowercase."
    }
   },
   "required": [
    "query"
   ],
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "type": "string",
     "minLength": 1
    },
    "__schema1": {
     "default": false,
     "type": "boolean"
    }
   },
   "examples": [
    {
     "query": "license",
     "case_sensitive": false
    }
   ]
  }
 },
 {
  "name": "analyze_function",
  "description": "Build a dossier for one native function identified by symbol or provider-returned address. Returns identity with complete inclusive body ranges or explicit unknown, provider-specific pseudocode and assembly, comments, calls, references, referenced strings/names, local CFG blocks, and available native API boundary observations.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "procedure": {
     "$ref": "#/$defs/__schema0",
     "description": "A procedure name or address"
    }
   },
   "required": [
    "procedure"
   ],
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "type": "string",
     "description": "A procedure name or address"
    }
   },
   "examples": [
    {
     "procedure": "main"
    }
   ]
  }
 },
 {
  "name": "inspect_artifact",
  "description": "Inspect an active archive or application package in one call. Returns the complete content-addressed artifact graph, source Evidence, observations, relationships, integrity contradictions, hypotheses, unexplored branches, limitations, and next probes inline. Read-only DMG mounting is used automatically on supported macOS hosts. Integrity mismatches fail by default; record-and-continue keeps mismatches explicitly untrusted. It does not extract files; use extract_artifact to materialize its regular contents.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "integrity_policy": {
     "$ref": "#/$defs/__schema0",
     "description": "Value for integrity policy."
    }
   },
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "default": "fail",
     "type": "string",
     "enum": [
      "fail",
      "record-and-continue"
     ]
    }
   },
   "examples": [
    {}
   ]
  }
 },
 {
  "name": "analyze_javascript_application",
  "description": "Reconstruct one local ASAR or extracted JavaScript application as an inline application graph without executing it. Returns recovered graph nodes, edges, semantic relations, integrity contradictions, limitations, and coverage. Integrity mismatches fail by default; record-and-continue retains observed bytes as untrusted and marks graph coverage partial.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "input_path": {
     "$ref": "#/$defs/__schema0",
     "description": "Local filesystem path for input."
    },
    "format": {
     "$ref": "#/$defs/__schema1",
     "description": "Declared input artifact format."
    },
    "integrity_policy": {
     "$ref": "#/$defs/__schema2",
     "description": "Value for integrity policy."
    }
   },
   "required": [
    "input_path"
   ],
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "type": "string",
     "minLength": 1,
     "pattern": "^[^\\x00]*$"
    },
    "__schema1": {
     "default": "auto",
     "type": "string",
     "enum": [
      "auto",
      "asar",
      "directory"
     ]
    },
    "__schema2": {
     "default": "fail",
     "type": "string",
     "enum": [
      "fail",
      "record-and-continue"
     ]
    }
   },
   "examples": [
    {
     "input_path": "/Applications/Example.app/Contents/Resources/app.asar",
     "format": "auto",
     "integrity_policy": "fail"
    }
   ]
  }
 },
 {
  "name": "inspect_managed_artifact",
  "description": "Open and classify an explicit managed PE/CLI path, or inspect the active managed target, then inventory exact assembly/module identity, target framework evidence, references, resources, and custom attributes without loading or executing target code. Returns complete inline inventories when metadata is admitted and explicit partial or malformed coverage otherwise.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "path": {
     "$ref": "#/$defs/__schema0",
     "description": "Explicit absolute managed PE/CLI path; omit to reuse the previously selected managed target"
    }
   },
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "type": "string",
     "minLength": 1,
     "pattern": "^[^\\x00]*$",
     "description": "Explicit absolute managed PE/CLI path; omit to reuse the previously selected managed target"
    }
   },
   "examples": [
    {
     "path": "/tmp/Example.dll"
    }
   ]
  }
 },
 {
  "name": "inspect_android_package",
  "description": "Inspect a local APK's package, SDK and permission declarations, decoded manifest and class/resource counts. Does not execute the APK or verify signatures. Requires a caller-supplied JADX MCP JAR and JDK; uses one worker in an isolated temporary workspace.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "path": {
     "$ref": "#/$defs/__schema0",
     "description": "Local APK path; the target is never executed"
    }
   },
   "required": [
    "path"
   ],
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "type": "string",
     "minLength": 1,
     "pattern": "^[^\\x00]*$",
     "description": "Local APK path; the target is never executed"
    }
   },
   "examples": [
    {
     "path": "/tmp/Example.apk"
    }
   ]
  }
 },
 {
  "name": "inspect_plist",
  "description": "Parse Info.plist from the active artifact by default, or pass any local plist path. Returns normalized JSON rather than plutil text.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "path": {
     "$ref": "#/$defs/__schema0",
     "description": "Local filesystem path used by this operation."
    }
   },
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "type": "string",
     "minLength": 1,
     "pattern": "^[^\\x00]*$"
    }
   },
   "examples": [
    {
     "path": "/Applications/Example.app/Contents/Info.plist"
    }
   ]
  }
 },
 {
  "name": "list_unknowns",
  "description": "List every current residual-unknown head in deterministic ID order, with optional exact status, severity, and domain filters. Results are complete and inline. This is read-only; unresolved, contradicted, and non-truth dispositions remain distinct.",
  "inputSchema": {
   "type": "object",
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "properties": {
    "status": {
     "$ref": "#/$defs/__schema0",
     "description": "Current lifecycle or verification status."
    },
    "severity": {
     "$ref": "#/$defs/__schema1",
     "description": "Value for severity."
    },
    "domain": {
     "$ref": "#/$defs/__schema2",
     "description": "Value for domain."
    }
   },
   "additionalProperties": false,
   "$defs": {
    "__schema0": {
     "type": "string",
     "enum": [
      "open",
      "investigating",
      "blocked",
      "contradicted",
      "resolved"
     ]
    },
    "__schema1": {
     "type": "string",
     "enum": [
      "low",
      "medium",
      "high",
      "critical"
     ]
    },
    "__schema2": {
     "type": "string",
     "minLength": 1
    }
   },
   "examples": [
    {}
   ]
  }
 }
]""")
