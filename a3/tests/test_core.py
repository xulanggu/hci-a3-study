import uuid
from concurrent.futures import ThreadPoolExecutor
import pytest
from study import load_data, sample_plan, derived, normalize_id, now
from storage import SQLiteStore, flush, retry, csv_text


@pytest.fixture
def dataset():
    return load_data()


def session(pid='tester', condition='baseline', plan=None):
    return dict(session_id=str(uuid.uuid4()),participant_id=pid,condition=condition,block_order=1,
                ai_workflow='two_stage',is_pilot=False,app_version='test',seed=42,trial_plan=plan or [],
                status='in_progress',attention_failures=0,user_agent='pytest',started_at=now(),completed_at=None)


def test_sampler(dataset):
    study,dev=dataset
    lookup={r['image_id']:r for r in study}
    for seed in range(100):
        plan,warnings=sample_plan(study,dev,set(),seed)
        assert not warnings
        assert plan==sample_plan(study,dev,set(),seed)[0]
        main=[p for p in plan if not p['is_attention_check']]
        assert len(main)==20
        assert sum(lookup[p['image_id']]['ai_correct_frozen'] for p in main)==16
        indices=[i for i,p in enumerate(plan) if p['is_attention_check']]
        assert len(indices)==2 and min(indices)>=3 and indices[1]-indices[0]>1
        assert len({p['image_id'] for p in plan})==22
        assert all(p['image_id'] in {r['image_id'] for r in dev} for p in plan if p['is_attention_check'])
        next_plan,_=sample_plan(study,dev,{p['image_id'] for p in plan},seed)
        assert not {p['image_id'] for p in plan}&{p['image_id'] for p in next_plan}


def test_exhausted_sampler(dataset):
    study,dev=dataset
    plan,warnings=sample_plan(study,dev,{r['image_id'] for r in study+dev},1)
    assert plan==[] and warnings
    for n in range(1,5):
        plan,_=sample_plan(study[:n],dev,set(),2)
        checks=[i for i,p in enumerate(plan) if p['is_attention_check']]
        assert all(i>=3 for i in checks)
        assert all(b-a>1 for a,b in zip(checks,checks[1:]))


@pytest.mark.parametrize('initial,final,correct,switched,follow,alternative',[
    ('fear','surprise',True,True,True,False),
    ('fear','fear',False,False,False,True),
    ('anger','sadness',False,True,False,False),
])
def test_derived(initial,final,correct,switched,follow,alternative):
    r=derived(final,'surprise',{'ai_top1':'surprise','ai_top2':'fear'},initial)
    assert (r['correct'],r['switched'],r['followed_ai_top1'],r['chose_ai_top2'])==(correct,switched,follow,alternative)
    assert derived(final,'surprise')['switched'] is None
    assert derived(final,'surprise')['followed_ai_top1'] is None
    assert derived(final,'surprise',attention_target=final)['attention_passed'] is True
    assert derived(final,'surprise',attention_target='neutral')['correct'] is None


def test_ids():
    assert normalize_id('  Abc_1 ')== 'abc_1'
    for bad in ['ab','a b','<script>','x'*21,'张三','abc-123']:
        with pytest.raises(ValueError): normalize_id(bad)


def test_storage_roundtrip_and_idempotence(tmp_path):
    store=SQLiteStore(tmp_path/'study.db'); s=session()
    store.start(s)
    assert store.rows('sessions')[0]==s
    exposure=dict(session_id=s['session_id'],trial_idx=0,participant_id='tester',image_id='face',shown_at=now())
    store.upsert('trials',exposure)
    result=dict(exposure,final_label='fear',ai_probs=[.125]*8,correct=True)
    store.upsert('trials',result);store.upsert('trials',result)
    rows=store.rows('trials'); assert len(rows)==1
    assert rows[0]['correct'] is True and rows[0]['ai_probs']==[.125]*8
    store.upsert('questionnaires',dict(session_id=s['session_id'],participant_id='tester',tlx_mental=3))
    assert store.rows('questionnaires')[0]['tlx_mental']==3
    store.upsert('sessions',dict(s,status='completed'))
    with pytest.raises(ValueError,match='already_completed'):store.start(session())
    pilot=session();pilot['is_pilot']=True;store.start(pilot)
    assert len(store.rows('sessions'))==2


def test_abandon_and_cross_condition_reservation(tmp_path):
    store=SQLiteStore(tmp_path/'study.db')
    first=session(plan=[{'image_id':'reserved'}]);store.start(first)
    with pytest.raises(ValueError,match='stale_plan'):
        store.start(session(condition='ai',plan=[{'image_id':'reserved'}]))
    second=session();store.start(second)
    assert store.rows('sessions',session_id=first['session_id'])[0]['status']=='abandoned'
    store.upsert('trials',dict(session_id=second['session_id'],trial_idx=0,participant_id='tester',image_id='seen'))
    with pytest.raises(ValueError,match='stale_plan'):
        store.start(session(plan=[{'image_id':'seen'}]))


def test_queue_order_retry_and_recovery(tmp_path):
    store=SQLiteStore(tmp_path/'study.db');s=session()
    class Unreachable:
        calls=0
        def upsert(self,*args):
            self.calls+=1
            raise ConnectionError('offline')
    broken=Unreachable()
    q=[('sessions',s),('trials',dict(session_id=s['session_id'],trial_idx=0,final_label='fear')),
       ('questionnaires',dict(session_id=s['session_id'],tlx_mental=3)),('sessions',dict(s,status='completed'))]
    assert not flush(broken,q)
    assert broken.calls==3 and len(q)==4
    assert flush(store,q) and not q
    assert store.rows('sessions')[0]['status']=='completed'
    assert len(store.rows('questionnaires'))==1


def test_sqlite_30_writers(tmp_path):
    store=SQLiteStore(tmp_path/'study.db')
    def write(i):
        s=session(pid=f'person_{i}');store.start(s)
        store.upsert('trials',dict(session_id=s['session_id'],participant_id=s['participant_id'],trial_idx=0,image_id='shared_face'))
    with ThreadPoolExecutor(max_workers=30) as pool:list(pool.map(write,range(30)))
    assert len(store.rows('sessions'))==len(store.rows('trials'))==30


def test_csv_escaping():
    result=csv_text([{'comment':'=1+1','ai_probs':[.1,.9]}])
    assert "'=1+1" in result
