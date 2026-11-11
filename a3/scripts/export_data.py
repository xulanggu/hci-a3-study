"""Export local or production data. Credentials read from environment/secrets file."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import tomllib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import ROOT
from storage import SQLiteStore, SupabaseStore, TABLES, csv_text, table_columns


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-pilot', action='store_true')
    parser.add_argument('--backend', choices=['sqlite','supabase'], default='sqlite')
    parser.add_argument('--db', type=Path, default=ROOT/'local.db')
    parser.add_argument('--secrets', type=Path, default=ROOT/'.streamlit/secrets.toml')
    parser.add_argument('--output', type=Path, default=ROOT/'exports')
    args=parser.parse_args()
    settings=tomllib.loads(args.secrets.read_text()) if args.secrets.exists() else {}
    if args.backend=='sqlite':
        if not args.db.exists():
            parser.error('Database does not exist; export will not create an empty study.')
        store=SQLiteStore(args.db)
    else:
        store=SupabaseStore(os.environ.get('SUPABASE_URL',settings.get('SUPABASE_URL','')),
                            os.environ.get('SUPABASE_SERVICE_KEY',settings.get('SUPABASE_SERVICE_KEY','')))
    args.output.mkdir(parents=True,exist_ok=True)
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    for table in TABLES:
        rows=store.rows(table, **({} if args.include_pilot else {'is_pilot':False}))
        # Header-only exports still carry the complete data dictionary schema.
        columns=table_columns(table)
        path=args.output/f'{table}_{timestamp}.csv'
        path.write_text(csv_text(rows, columns))
        print(f'{table}: {len(rows)} rows → {path}')


if __name__=='__main__':
    main()
