"""SQLite run store, shared by the web bench and the CLI."""
import json, os, sqlite3, time, uuid

DB_PATH = os.environ.get('CRN_DB', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs.db'))

_SCHEMA = ("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created REAL,"
           " status TEXT, note TEXT, config TEXT, results TEXT, summary TEXT,"
           " error TEXT)")


def _db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute(_SCHEMA)
    # added in 0.7.0; older databases are migrated in place so run history
    # from earlier versions keeps working
    cols = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    if 'verdict' not in cols:
        con.execute("ALTER TABLE runs ADD COLUMN verdict TEXT")
        con.commit()
    return con


def save(run_id, status, cfg, results=None, summary=None, note='', error=None,
         verdict=None):
    con = _db()
    con.execute("INSERT OR REPLACE INTO runs (id,created,status,note,config,"
                "results,summary,error,verdict) VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, time.time(), status, note, json.dumps(cfg),
                 json.dumps(results or []), json.dumps(summary or []), error,
                 json.dumps(verdict) if verdict else None))
    con.commit(); con.close()


def listing(limit=100):
    con = _db()
    rows = con.execute("SELECT id,created,status,note,config,summary,error FROM"
                       " runs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [{'id': r[0], 'created': r[1], 'status': r[2], 'note': r[3],
             'config': json.loads(r[4]), 'summary': json.loads(r[5] or '[]'),
             'error': r[6]} for r in rows]


def get(run_id):
    con = _db()
    r = con.execute("SELECT id,created,status,note,config,results,summary,"
                    "error,verdict FROM runs WHERE id=?", (run_id,)).fetchone()
    con.close()
    if not r:
        return None
    return {'id': r[0], 'created': r[1], 'status': r[2], 'note': r[3],
            'config': json.loads(r[4]), 'results': json.loads(r[5] or '[]'),
            'summary': json.loads(r[6] or '[]'), 'error': r[7],
            'verdict': json.loads(r[8]) if r[8] else None}


def delete(run_id):
    con = _db(); con.execute("DELETE FROM runs WHERE id=?", (run_id,))
    con.commit(); con.close()


def reap_orphans():
    """A run still marked 'running' means the server died mid-sweep."""
    con = _db()
    con.execute("UPDATE runs SET status='interrupted', error='server restarted"
                " mid-sweep' WHERE status='running'")
    con.commit(); con.close()


def new_id():
    return uuid.uuid4().hex[:12]
