"""Idempotent persistence; SQLite for local use, Supabase for deployment."""
import csv
import io
import json
import sqlite3
import time
from pathlib import Path

TABLES = ('sessions', 'trials', 'questionnaires')
KEYS = {'sessions': ['session_id'], 'trials': ['session_id', 'trial_idx'], 'questionnaires': ['session_id']}
JSON_FIELDS = {'trial_plan', 'ai_probs'}
BOOL_FIELDS = {'is_pilot', 'is_attention_check', 'attention_passed', 'ai_correct_frozen',
               'correct', 'switched', 'followed_ai_top1', 'chose_ai_top2'}
# Same field names/types as production; generated IDs use SQLite's native syntax.
SQLITE_SCHEMA = Path(__file__).with_name('schema.sql').read_text().split('create index sessions_participant')[0]
SQLITE_SCHEMA = SQLITE_SCHEMA.replace('create table ', 'create table if not exists ')
SQLITE_SCHEMA = SQLITE_SCHEMA.replace('bigint generated always as identity primary key', 'integer primary key autoincrement')
SQLITE_SCHEMA = SQLITE_SCHEMA.replace('default now()', 'default CURRENT_TIMESTAMP')


def encode(row):
    return {k: json.dumps(v) if k in JSON_FIELDS and v is not None else v for k,v in row.items()}


def decode(row):
    return {k: json.loads(v) if k in JSON_FIELDS and v is not None else
            bool(v) if k in BOOL_FIELDS and v is not None else v for k,v in dict(row).items()}


class SQLiteStore:
    backend = 'sqlite'

    def __init__(self, path):
        self.path = str(path)
        with self.connect() as conn:
            conn.execute('pragma journal_mode=WAL')
            conn.executescript(SQLITE_SCHEMA)

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute('pragma foreign_keys=ON')
        return conn

    def rows(self, table, **filters):
        assert table in TABLES
        assert set(filters) <= set(self.columns(table))
        where = ' and '.join(f'{k}=?' for k in filters)
        with self.connect() as conn:
            return [decode(r) for r in conn.execute(f'select * from {table}'+(' where '+where if where else ''), list(filters.values()))]

    def columns(self, table):
        assert table in TABLES
        with self.connect() as conn:
            return [r['name'] for r in conn.execute(f'pragma table_info({table})')]

    def health(self):
        with self.connect() as conn:
            conn.execute('select 1 from sessions limit 1').fetchall()

    def upsert(self, table, row):
        assert table in TABLES
        assert set(row) <= set(self.columns(table))
        keys = KEYS[table]
        fields = list(row)
        update = ','.join(f'{k}=excluded.{k}' for k in fields if k not in keys)
        query = f"insert into {table} ({','.join(fields)}) values ({','.join('?' for _ in fields)}) on conflict ({','.join(keys)}) do update set {update}"
        with self.connect() as conn:
            conn.execute(query, list(encode(row).values()))

    def start(self, row):
        # Transaction protects duplicate completion checks and simultaneous ID starts.
        with self.connect() as conn:
            conn.execute('begin immediate')
            if conn.execute('select 1 from sessions where session_id=?', (row['session_id'],)).fetchone():
                return
            history = [decode(r) for r in conn.execute('select * from sessions where participant_id=?', (row['participant_id'],))]
            if not row['is_pilot'] and any(s['condition']==row['condition'] and s['status']=='completed' and not s['is_pilot'] for s in history):
                raise ValueError('already_completed')
            used = {r[0] for r in conn.execute('select image_id from trials where participant_id=?', (row['participant_id'],))}
            for session in history:
                if session['status']=='in_progress' and session['condition']!=row['condition']:
                    used.update(p['image_id'] for p in session['trial_plan'])
            if used.intersection(p['image_id'] for p in row['trial_plan']):
                raise ValueError('stale_plan')
            conn.execute("update sessions set status='abandoned' where participant_id=? and condition=? and status='in_progress'", (row['participant_id'],row['condition']))
            conn.execute(f"insert into sessions ({','.join(row)}) values ({','.join('?' for _ in row)})", list(encode(row).values()))


class SupabaseStore:
    backend = 'supabase'

    def __init__(self, url, key):
        from supabase import create_client, ClientOptions
        self.client = create_client(url, key, options=ClientOptions(postgrest_client_timeout=8))

    def health(self):
        self.client.table('sessions').select('session_id').limit(1).execute()

    def rows(self, table, **filters):
        assert table in TABLES
        result, offset = [], 0
        # Explicit pagination avoids the default 1,000-row API cap in exports/history.
        while True:
            q = self.client.table(table).select('*')
            for key, value in filters.items():
                q = q.eq(key, value)
            page = q.order('id' if table=='trials' else 'session_id').range(offset, offset+499).execute().data
            result.extend(page)
            if len(page) < 500:
                return result
            offset += 500

    def upsert(self, table, row):
        assert table in TABLES
        self.client.table(table).upsert(row, on_conflict=','.join(KEYS[table])).execute()

    def start(self, row):
        # A lost response after a committed RPC can be safely retried with the same UUID.
        if self.rows('sessions', session_id=row['session_id']):
            return
        self.client.rpc('start_study_session', {'payload': row}).execute()


def retry(action, attempts=3, delay=.15):
    for attempt in range(attempts):
        try:
            return action()
        except Exception:
            if attempt == attempts-1:
                raise
            time.sleep(delay * (2**attempt))


def flush(store, queue):
    """Preserve write order: exposures → answers → questionnaire → completed."""
    while queue:
        table, row = queue[0]
        try:
            retry(lambda: store.upsert(table, row))
        except Exception:
            return False
        queue.pop(0)
    return True


def table_columns(table):
    assert table in TABLES
    with sqlite3.connect(':memory:') as conn:
        conn.executescript(SQLITE_SCHEMA)
        return [r[1] for r in conn.execute(f'pragma table_info({table})')]


def csv_text(rows, columns=None):
    out = io.StringIO()
    fields = columns or (list(rows[0]) if rows else [])
    if fields:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            # Protect human-entered IDs/comments/user agents in spreadsheet exports.
            safe = {}
            for k,v in row.items():
                if isinstance(v, (list,dict)):
                    v = json.dumps(v, separators=(',', ':'))
                if isinstance(v,str) and v.lstrip().startswith(('=', '+', '-', '@')):
                    v = "'"+v
                safe[k] = v
            writer.writerow(safe)
    return out.getvalue()
