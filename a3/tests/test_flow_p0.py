"""P0 acceptance gate; uses real forms and an isolated SQLite database."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from storage import SQLiteStore
from config import LABELS

APP=Path(__file__).resolve().parents[1]/'app.py'


def launch(condition='baseline', pilot=False):
    at=AppTest.from_file(str(APP),default_timeout=30)
    at.query_params['condition']=condition
    if pilot:at.query_params['pilot']='1'
    at.run();assert not at.exception
    return at


def click(at,label):
    next(b for b in at.button if b.label==label).click().run()
    assert not at.exception


def start(at,pid):
    click(at,'Continue')
    at.text_input[0].set_value(pid)
    click(at,'Continue to instructions')
    if at.session_state['page']=='instructions':click(at,'Start labeling')
    return at


def finish(at,fail_attention=False):
    guard=0
    while at.session_state['page']=='trials':
        guard+=1;assert guard<60
        i=at.session_state['idx'];item=at.session_state['plan'][i]
        choice=item['attention_target'] if item['is_attention_check'] else 'fear'
        if item['is_attention_check'] and fail_attention:
            choice=next(x for x in LABELS if x!=choice)
        assert all(r.value is None for r in at.radio)
        at.radio[0].set_value(choice.title())
        if len(at.radio)>1:at.radio[1].set_value(4)
        at.button[-1].click().run();assert not at.exception
    if at.session_state['page']=='questionnaire':
        click(at,'Finish');assert at.error and at.session_state['page']=='questionnaire'
        for radio in at.radio:radio.set_value(3)
        click(at,'Finish')
    return at


@pytest.fixture
def db(tmp_path,monkeypatch):
    path=tmp_path/'study.db'
    monkeypatch.setenv('A3_STORAGE_BACKEND','sqlite')
    monkeypatch.setenv('A3_SQLITE_PATH',str(path))
    return path


@pytest.mark.parametrize('conditions',[('baseline','ai'),('ai','baseline')])
def test_full_both_orders(db,conditions):
    sessions=[]
    for condition in conditions:
        at=start(launch(condition),'  Pair_123  ')
        i=at.session_state['idx'];t=at.session_state['trial_states'][i]['t_initial']
        at.button[-1].click().run()
        assert at.error and at.session_state['idx']==i
        assert at.session_state['trial_states'][i]['t_initial']==t
        finish(at)
        assert at.session_state['page']=='done' and not at.session_state['queue']
        sessions.append(at.session_state['session'])
    assert [s['block_order'] for s in sessions]==[1,2]
    store=SQLiteStore(db);rows=store.rows('trials')
    assert len(rows)==44 and len({r['image_id'] for r in rows})==44
    assert len(store.rows('questionnaires'))==2
    assert all(s['status']=='completed' for s in store.rows('sessions'))
    for row in rows:
        assert row['participant_id']=='pair_123'
        assert row['shown_at'] and row['submitted_at']
        assert row['rt_total_ms']>=0
        assert row['rt_total_ms']==row['rt_initial_ms']+(row['rt_final_ms'] or 0)
        if row['is_attention_check']:
            assert row['attention_passed'] and row['confidence'] is None
            assert row['correct'] is None and row['ai_source'] is None
        else:
            assert row['correct']==(row['final_label']==row['true_label'])
            assert row['confidence']==4
            if row['condition']=='baseline':
                for field in ['initial_label','ai_top1','ai_top2','ai_probs','ai_source','ai_inference_ms','switched','followed_ai_top1','chose_ai_top2']:
                    assert row[field] is None
            else:
                assert row['ai_source'] in ('live','cached_fallback')
                assert len(row['ai_probs'])==8
                assert row['switched']==(row['initial_label']!=row['final_label'])
                assert row['followed_ai_top1']==(row['final_label']==row['ai_top1'])
                assert row['chose_ai_top2']==(row['final_label']==row['ai_top2'])
    duplicate=start(launch(conditions[0]),'pair_123')
    assert duplicate.session_state['page']=='duplicate'


def test_attention_failure_and_restart(db):
    at=finish(start(launch('ai'),'attention_123'),True)
    assert at.session_state['page']=='terminated'
    store=SQLiteStore(db)
    previous={r['image_id'] for r in store.rows('trials')}
    assert store.rows('sessions')[0]['status']=='terminated'
    assert store.rows('sessions')[0]['attention_failures']==2
    click(at,'Restart');start(at,'attention_123')
    assert not previous & {p['image_id'] for p in at.session_state['plan']}
    assert len(store.rows('sessions'))==2


def test_refresh_excludes_unanswered_exposure(db):
    first=start(launch(),'refresh_123')
    old=first.session_state['session']['session_id']
    image=first.session_state['plan'][0]['image_id']
    second=start(launch(),'refresh_123')
    store=SQLiteStore(db)
    assert store.rows('sessions',session_id=old)[0]['status']=='abandoned'
    assert image not in {p['image_id'] for p in second.session_state['plan']}
    assert store.rows('trials',session_id=old)[0]['final_label'] is None


def test_no_storage_still_renders(monkeypatch):
    monkeypatch.delenv('A3_STORAGE_BACKEND',raising=False)
    at=launch()
    assert at.warning and at.title
    click(at,'Continue');at.text_input[0].set_value('offline_123');click(at,'Continue to instructions')
    assert at.error and at.session_state['page']=='id'


def test_invalid_condition(db):
    at=launch('bad')
    assert at.info and not at.button and not at.radio


def test_one_stage(db,monkeypatch):
    import config
    monkeypatch.setattr(config,'AI_WORKFLOW','one_stage')
    at=finish(start(launch('ai'),'single_123'))
    assert at.session_state['page']=='done'
    for row in SQLiteStore(db).rows('trials'):
        assert row['initial_label'] is None and row['switched'] is None and row['rt_final_ms'] is None


def test_pilot_repeats_and_tags(db):
    finish(start(launch(),'pilot_123'))
    at=finish(start(launch(pilot=True),'pilot_123'))
    assert at.session_state['page']=='done'
    store=SQLiteStore(db)
    sid=at.session_state['session']['session_id']
    for table in ('sessions','trials','questionnaires'):
        assert all(r['is_pilot'] for r in store.rows(table,session_id=sid))
