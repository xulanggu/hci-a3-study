from streamlit.testing.v1 import AppTest
import pytest
from test_flow_p0 import launch, start, finish, click, db, APP
from storage import SQLiteStore
from inference import advice
from study import load_data


def test_fallback(monkeypatch):
    import inference
    def broken(_):raise RuntimeError('model unavailable')
    monkeypatch.setattr(inference,'predict',broken)
    row=load_data()[0][0]
    output=advice(row)
    assert output['ai_source']=='cached_fallback'
    assert output['ai_top1']==row['model_prediction']
    assert output['ai_top2']==row['model_second_choice']
    assert output['ai_probs']==row['probs']


def test_live_advice_logged_before_final(db,monkeypatch):
    import inference
    calls=[]
    def predict(image_id):
        calls.append(image_id)
        return 'surprise','fear',[.1,.1,.3,.1,.1,.1,.1,.1],12.5
    monkeypatch.setattr(inference,'predict',predict)
    at=start(launch('ai'),'live_123')
    assert not calls  # Two-stage: no early advice.
    at.radio[0].set_value('Fear');click(at,'Submit & see AI suggestion')
    at.run();assert len(calls)==1
    row=SQLiteStore(db).rows('trials')[0]
    assert row['ai_source']=='live' and row['ai_inference_ms']==12.5 and row['final_label'] is None
    assert row['initial_label']=='fear'
    assert all(r.value is None for r in at.radio)
    at.radio[0].set_value('Surprise');click(at,'Submit final answer')
    assert at.error and at.session_state['idx']==0  # Missing confidence.
    at.radio[1].set_value(5);click(at,'Submit final answer')
    row=SQLiteStore(db).rows('trials')[0]
    assert row['switched'] and row['followed_ai_top1'] and not row['chose_ai_top2']
    assert len(calls)==1


def test_write_outage_and_recovery(db,monkeypatch):
    at=start(launch(),'outage_123')
    original=SQLiteStore.upsert
    def broken(self,*args):raise ConnectionError('offline')
    monkeypatch.setattr(SQLiteStore,'upsert',broken)
    at.radio[0].set_value('Fear');at.radio[1].set_value(3);click(at,'Submit')
    assert at.session_state['idx']==1 and at.session_state['queue']
    assert not at.radio  # Unlogged new face must not be revealed.
    assert any(b.label=='Retry this trial' for b in at.button)
    monkeypatch.setattr(SQLiteStore,'upsert',original)
    click(at,'Retry this trial')
    assert not at.session_state['queue'] and len(at.radio)==2
    assert len(SQLiteStore(db).rows('trials'))==2


def test_admin_auth_and_exports(db):
    finish(start(launch(),'admin_data'))
    denied=AppTest.from_file(str(APP));denied.query_params['admin']='bad';denied.run()
    assert denied.error and not denied.dataframe
    at=AppTest.from_file(str(APP),default_timeout=30)
    at.secrets['ADMIN_TOKEN']='test-secret'
    at.query_params['admin']='test-secret';at.run()
    assert not at.exception and len(at.dataframe)==2
    assert len(at.get('download_button'))==3
    assert at.metric[0].value=='0'
    click(at,'Refresh')
    assert not at.exception


def test_advice_outage_waits_for_durable_reveal(db,monkeypatch):
    at=start(launch('ai'),'advice_123')
    original=SQLiteStore.upsert
    def broken_advice(self,table,row):
        if 'ai_source' in row:raise ConnectionError('offline')
        return original(self,table,row)
    monkeypatch.setattr(SQLiteStore,'upsert',broken_advice)
    at.radio[0].set_value('Fear');click(at,'Submit & see AI suggestion')
    assert not at.radio and at.session_state['queue']
    assert not any('AI suggestion:' in m.value for m in at.markdown)
    monkeypatch.setattr(SQLiteStore,'upsert',original)
    click(at,'Retry suggestion')
    assert len(at.radio)==2 and not at.session_state['queue']
    assert any('AI suggestion:' in m.value for m in at.markdown)


def test_unreachable_configured_storage_renders(db,monkeypatch):
    def broken_health(self):raise ConnectionError('offline')
    monkeypatch.setattr(SQLiteStore,'health',broken_health)
    at=launch()
    assert at.warning and not at.exception
