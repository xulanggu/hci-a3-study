"""Run both orders and attention termination using actual Streamlit AppTest forms.

Creates an isolated temporary SQLite database by default; never uses study storage.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from test_flow_p0 import launch, start, finish, click
from storage import SQLiteStore


def run(path):
    if path.exists():
        raise ValueError('Simulation refuses to overwrite an existing database; choose a fresh path.')
    os.environ['A3_STORAGE_BACKEND']='sqlite'
    os.environ['A3_SQLITE_PATH']=str(path)
    for pid,order in [('sim_base_first',('baseline','ai')),('sim_ai_first',('ai','baseline'))]:
        for condition in order:
            at=finish(start(launch(condition),pid))
            assert at.session_state['page']=='done'
    failed=finish(start(launch('ai'),'sim_attention'),True)
    assert failed.session_state['page']=='terminated'
    click(failed,'Restart');start(failed,'sim_attention')
    assert failed.session_state['page']=='trials'
    store=SQLiteStore(path)
    sessions,trials,questions=[store.rows(t) for t in ('sessions','trials','questionnaires')]
    assert len(sessions)==6 and len(questions)==4
    assert sum(s['status']=='completed' for s in sessions)==4
    assert sum(s['status']=='terminated' for s in sessions)==1
    for pid in ('sim_base_first','sim_ai_first'):
        rows=[r for r in trials if r['participant_id']==pid]
        assert len(rows)==44 and len({r['image_id'] for r in rows})==44
        assert sorted(s['block_order'] for s in sessions if s['participant_id']==pid)==[1,2]
    assert all(r['submitted_at'] for r in trials if r['participant_id']!='sim_attention')
    print(json.dumps({'simulation_only':True,'sessions':len(sessions),'completed':4,'terminated':1,
                      'restart_in_progress':1,'trial_rows':len(trials),'questionnaires':len(questions),
                      'database':str(path),'assertions':'passed'},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,help='Retain simulation data in this NEW SQLite file.')
    args=parser.parse_args()
    if args.db:
        run(args.db)
    else:
        with tempfile.TemporaryDirectory(prefix='a3-simulation-') as directory:
            run(Path(directory)/'simulation.db')


if __name__=='__main__':main()
