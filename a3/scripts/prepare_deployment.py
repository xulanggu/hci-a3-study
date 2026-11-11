"""Create a clean deployable repo tree. Never copies secrets, response DBs or exports."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
# Explicit allowlist keeps future local files out of the deployment by default.
FILES=['app.py','admin.py','config.py','inference.py','storage.py','study.py',
       'requirements.txt','requirements-dev.txt','schema.sql','README.md','DATA_USE.txt','.gitignore']
DIRECTORIES=['assets/images','models','data','vendor','tests','scripts']
SUFFIXES={'.py','.sh','.csv','.json','.png','.onnx','.bib','.txt'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    destination=args.destination.resolve()
    if destination.exists():
        parser.error('Destination already exists; choose a fresh directory to avoid overwriting work.')
    app=destination/'a3';app.mkdir(parents=True)
    paths=[ROOT/f for f in FILES]
    for directory in DIRECTORIES:
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in SUFFIXES and '__pycache__' not in p.parts)
    paths += [ROOT/'.streamlit/secrets.toml.example',ROOT/'data/dataset_README']
    hashes={}
    for source in sorted(set(paths)):
        target=app/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        hashes[str(target.relative_to(destination))]=hashlib.sha256(target.read_bytes()).hexdigest()
    workflows=destination/'.github/workflows';workflows.mkdir(parents=True)
    workflow_source=ROOT/'.github/workflows/validate.yml'
    if not workflow_source.exists():
        workflow_source=ROOT.parent/'.github/workflows/validate.yml'
    shutil.copy2(workflow_source,workflows/'validate.yml')
    (destination/'.gitignore').write_text('**/.venv/\n**/__pycache__/\n**/.pytest_cache/\n**/.streamlit/secrets.toml\n.env\n.env.*\n*.db\n*.db-shm\n*.db-wal\na3/exports/\n.DS_Store\n')
    (destination/'README.md').write_text('# CSE 594 A3 study\n\nDeploy `a3/app.py` on Streamlit Community Cloud using Python 3.11.\nSee [the study README](a3/README.md) for Supabase setup, pilot URLs, and validation limits.\nKeep this repository private because it includes the course study assets.\n')
    (destination/'.gitattributes').write_text('a3/data/*.csv -text\n*.onnx binary\n*.png binary\n')
    (destination/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(f'Prepared {len(hashes)} app files in {destination}; no secrets or participant responses copied.')


if __name__=='__main__':main()
