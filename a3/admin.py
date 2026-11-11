"""Authenticated study monitoring; imported only for an admin request."""
import hmac
import streamlit as st
from storage import TABLES, csv_text, table_columns
from inference import checkpoint_status


def summary(sessions, trials):
    sessions=[s for s in sessions if not s['is_pilot']]
    counts=[];valid={}
    for condition in ('baseline','ai'):
        group=[s for s in sessions if s['condition']==condition]
        # Conservative validity: every scheduled check passed and at least two checks.
        valid[condition]=set()
        for s in group:
            checks=[r for r in trials if r['session_id']==s['session_id'] and r['is_attention_check']]
            expected=sum(p['is_attention_check'] for p in s['trial_plan'])
            if s['status']=='completed' and s['attention_failures']==0 and expected>=2 and len(checks)==expected and all(r['attention_passed'] is True for r in checks):
                valid[condition].add(s['participant_id'])
        counts.append(dict(condition=condition, **{status:sum(s['status']==status for s in group)
                      for status in ('completed','terminated','in_progress','abandoned')}, valid_participants=len(valid[condition])))
    completed={c:{s['participant_id'] for s in sessions if s['condition']==c and s['status']=='completed'} for c in ('baseline','ai')}
    return counts, len(completed['baseline'] & completed['ai']), len(valid['baseline'] & valid['ai'])


def render(store, backend, provided, expected):
    if not expected or not hmac.compare_digest(str(provided).encode(),str(expected).encode()):
        st.error('Admin access denied.')
        st.stop()
    st.title('Study monitor')
    if st.button('Refresh'):
        st.rerun()
    st.write(f'Storage backend: {backend}')
    st.write('Checkpoint SHA-256 check')
    st.json(checkpoint_status())
    try:
        if store is None:raise ConnectionError('unavailable')
        tables={name:store.rows(name) for name in TABLES}
    except Exception:
        st.error('Storage connection unavailable. Check the server configuration and retry.')
        st.stop()
    st.success('Storage connected')
    counts,both,valid_both=summary(tables['sessions'],tables['trials'])
    st.dataframe(counts,hide_index=True)
    st.metric('IDs completed both conditions',both)
    st.metric('IDs valid in both conditions',valid_both)
    st.caption('Valid = completed, non-pilot, at least two attention checks, all passed. A completed block with one failed check is counted as completed but excluded from valid counts.')
    sources=[r['ai_source'] for r in tables['trials'] if not r['is_pilot'] and r['ai_source'] is not None]
    share=sources.count('cached_fallback')/len(sources) if sources else 0
    st.metric('Cached fallback share (non-pilot advice exposures)',f'{share:.1%}')
    st.subheader('Last 20 sessions')
    recent=sorted(tables['sessions'],key=lambda s:s['started_at'],reverse=True)[:20]
    st.dataframe([{k:s[k] for k in ('started_at','participant_id','condition','block_order','status','attention_failures','is_pilot')} for s in recent],hide_index=True)
    include=st.toggle('Include pilot rows in downloads',value=False)
    for name,rows in tables.items():
        selected=[r for r in rows if include or not r['is_pilot']]
        st.download_button(f'Download {name}.csv',csv_text(selected,table_columns(name)),file_name=name+'.csv',mime='text/csv')
    st.caption('Unsubmitted trial rows are exposure records, not answers. Filter submitted_at before analyzing accuracy or response time.')
