"""CSE 594 A3: run with `streamlit run a3/app.py`."""
import io
import logging
import os
from pathlib import Path
import secrets
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st
from PIL import Image
import config as C
from inference import data, advice, checkpoint_status
from storage import SQLiteStore, SupabaseStore, retry, flush
from study import now, normalize_id, sample_plan, derived

st.set_page_config(page_title='Facial-expression labeling · CSE 594', page_icon='◐', layout='centered')
S = st.session_state


def setting(key, default=None):
    try:
        return st.secrets.get(key, default)
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return default


@st.cache_resource
def get_store(backend, path, url, key):
    if backend == 'sqlite':
        return SQLiteStore(path)
    if backend == 'supabase':
        if not url or not key:
            raise ValueError('Supabase configuration missing')
        return SupabaseStore(url, key)
    raise ValueError('Unknown storage backend')


@st.cache_data
def image_bytes(image_path):
    with Image.open(C.ROOT/image_path) as im:
        out = io.BytesIO()
        im.convert('L').resize((C.IMAGE_DISPLAY_PX, C.IMAGE_DISPLAY_PX), Image.Resampling.BICUBIC).save(out, format='PNG')
        return out.getvalue()


def enqueue(table, row):
    S.queue.append((table, row.copy()))
    if store is not None:
        flush(store, S.queue)


def save_session(**updates):
    S.session.update(updates)
    enqueue('sessions', S.session)


def pending_notice():
    if S.queue:
        st.warning('Some responses are waiting to save. Please keep this tab open; saving will retry on your next submission.')
        if st.button('Retry saving', key='retry_saving'):
            if store is not None:
                flush(store, S.queue)
            st.rerun()


def initialize():
    S.page = 'welcome'
    S.queue = []
    S.recorded = set()
    S.trial_states = {}
    S.idx = 0


def begin_session(participant):
    # Atomic storage validation catches history changes between read and insert.
    for _ in range(3):
        history = store.rows('sessions', participant_id=participant)
        if not pilot and any(s['condition']==condition and s['status']=='completed' and not s['is_pilot'] for s in history):
            S.page = 'duplicate'
            return
        seen = {r['image_id'] for r in store.rows('trials', participant_id=participant)}
        # Reserve active plans from the other condition for simultaneous tabs.
        for prior in history:
            if prior['status']=='in_progress' and prior['condition']!=condition:
                seen.update(p['image_id'] for p in prior['trial_plan'])
        seed = secrets.randbits(52)
        plan, warnings = sample_plan(study, dev, seen, seed)
        if not any(not p['is_attention_check'] for p in plan):
            raise ValueError('pool_exhausted')
        session = dict(session_id=str(uuid.uuid4()), participant_id=participant, condition=condition,
                       block_order=2 if any(s['condition']!=condition and s['status']=='completed' and not s['is_pilot'] for s in history) else 1,
                       ai_workflow=C.AI_WORKFLOW, is_pilot=pilot, app_version=C.APP_VERSION,
                       seed=seed, trial_plan=plan, status='in_progress', attention_failures=0,
                       user_agent=st.context.headers.get('User-Agent','') if hasattr(st, 'context') else '',
                       started_at=now(), completed_at=None)
        try:
            retry(lambda: store.start(session))
        except Exception as error:
            if 'stale_plan' in str(error):
                continue
            if 'already_completed' in str(error):
                S.page = 'duplicate'
                return
            raise
        for warning in warnings:
            logging.warning('Sampling session %s: %s', session['session_id'], warning)
        S.session, S.plan, S.seed = session, plan, seed
        S.sampling_warnings = warnings
        S.page = 'instructions'
        return
    raise ValueError('history_changed')


def render_trial():
    i = S.idx
    if i >= len(S.plan):
        S.page = 'questionnaire'
        st.rerun()
    item = S.plan[i]
    row = lookup[item['image_id']]
    attention = item['is_attention_check']
    assisted = condition == 'ai' and not attention
    two_stage = assisted and S.session['ai_workflow'] == 'two_stage'
    # Detect a new tab that abandoned this session before accepting further data.
    if store is not None:
        try:
            current = store.rows('sessions', session_id=S.session['session_id'])
            if current and current[0]['status'] != 'in_progress':
                st.info('This session is no longer active. Please return to the class sheet or your other tab.')
                st.stop()
        except Exception:
            pass
    state = S.trial_states.setdefault(i, {})
    if 'shown_at' not in state:
        state.update(shown_at=now(), t_initial=time.time(), stage=1)
        exposure = dict(session_id=S.session['session_id'], participant_id=S.session['participant_id'],
                        condition=condition, is_pilot=pilot, app_version=C.APP_VERSION,
                        trial_idx=i, image_id=item['image_id'], is_attention_check=attention,
                        attention_target=item['attention_target'], shown_at=state['shown_at'],
                        true_label=None if attention else row['reference_label'],
                        ai_correct_frozen=None if attention else row['ai_correct_frozen'])
        state['exposure'] = exposure
        enqueue('trials', exposure)
    # A new stimulus must have a durable exposure before it can be displayed.
    # Submitted responses remain queued without losing progress during transient outages.
    exposure_pending = any(t=='trials' and r.get('trial_idx')==i and r.get('session_id')==S.session['session_id']
                           for t,r in S.queue)
    if exposure_pending and not state.get('displayed'):
        st.info('Waiting to save this trial before showing the next face. Please retry saving.')
        if st.button('Retry this trial'):
            if store is not None:
                flush(store, S.queue)
            st.rerun()
        st.stop()
    if not state.get('displayed'):
        state['t_initial'] = time.time()
        state['displayed'] = True
    st.progress(i/len(S.plan), text=f'{i+1} / {len(S.plan)}')
    if attention:
        st.warning(f"Attention check — please select **{item['attention_target'].title()}** for this one, regardless of the face.")
    st.image(image_bytes(row['image_path']), width=C.IMAGE_DISPLAY_PX)
    reveal = assisted and (not two_stage or state['stage']==2)
    if reveal:
        if 'ai' not in state:
            state['ai'] = advice(row)
            enqueue('trials', dict(session_id=S.session['session_id'], trial_idx=i, **state['ai']))
        if not state.get('advice_displayed'):
            if S.queue:
                st.info('Waiting to save the suggestion. Please retry before continuing.')
                if st.button('Retry suggestion'):
                    if store is not None:
                        flush(store, S.queue)
                    st.rerun()
                st.stop()
            if two_stage:
                state['t_final'] = time.time()
            state['advice_displayed'] = True
        if two_stage:
            st.write(f"Your answer: {state['initial_label'].title()}")
        with st.container(border=True):
            st.markdown(f"AI suggestion: **{state['ai']['ai_top1'].title()}**")
            st.write(f"Alternative: {state['ai']['ai_top2'].title()}")
    initial_stage = two_stage and state['stage']==1
    suffix = f"{S.session['session_id']}_{i}_{state['stage']}"
    with st.form('trial_'+suffix):
        label = st.radio('Which facial expression do you see?' if initial_stage else 'Your final answer',
                         [x.title() for x in C.LABELS], index=None, horizontal=True, key='label_'+suffix)
        confidence = None
        if not attention and not initial_stage:
            confidence = st.radio('How confident are you in your final answer?', list(range(1,6)),
                                  index=None, horizontal=True, key='confidence_'+suffix,
                                  help='1 = not at all, 5 = very')
        submitted = st.form_submit_button('Submit & see AI suggestion' if initial_stage else
                                         'Submit final answer' if reveal else 'Submit')
        if submitted:
            if label is None or (not attention and not initial_stage and confidence is None):
                st.error('Please select an answer'+(' and confidence.' if not attention and not initial_stage else '.'))
            elif i not in S.recorded:
                submitted_time = time.time()
                label = label.lower()
                if initial_stage:
                    state.update(initial_label=label, rt_initial_ms=round((submitted_time-state['t_initial'])*1000), stage=2)
                    enqueue('trials', dict(session_id=S.session['session_id'], trial_idx=i,
                                          initial_label=label, rt_initial_ms=state['rt_initial_ms']))
                    st.rerun()
                initial_ms = state['rt_initial_ms'] if two_stage else round((submitted_time-state['t_initial'])*1000)
                final_ms = round((submitted_time-state['t_final'])*1000) if two_stage else None
                result = dict(state['exposure'], initial_label=state.get('initial_label'), final_label=label,
                              confidence=confidence, rt_initial_ms=initial_ms, rt_final_ms=final_ms,
                              rt_total_ms=initial_ms+(final_ms or 0), submitted_at=now())
                if reveal:
                    result.update(state['ai'])
                result.update(derived(label, row['reference_label'], state.get('ai'), state.get('initial_label'), item['attention_target']))
                enqueue('trials', result)
                S.recorded.add(i)
                S.idx += 1
                if attention and not result['attention_passed']:
                    failures = S.session['attention_failures']+1
                    save_session(attention_failures=failures)
                    if failures >= C.ATTENTION_FAILS_TO_TERMINATE:
                        save_session(status='terminated')
                        S.page = 'terminated'
                st.rerun()


def questionnaire():
    questions = {
        'tlx_mental': ('How mentally demanding was the task?', '1 = Very low, 7 = Very high'),
        'tlx_effort': ('How hard did you have to work to accomplish your level of performance?', '1 = Very low, 7 = Very high'),
        'tlx_frustration': ('How insecure, discouraged, irritated, stressed, and annoyed were you?', '1 = Very low, 7 = Very high'),
        'tlx_performance': ('How successful were you in accomplishing what you were asked to do?', '1 = Perfect, 7 = Failure'),
    }
    if condition=='ai':
        questions.update(trust=("I trusted the AI's suggestions.", '1 = Strongly disagree, 7 = Strongly agree'),
                         usefulness=("The AI's suggestions helped me label the faces more accurately.", '1 = Strongly disagree, 7 = Strongly agree'))
    st.subheader('A few final questions')
    with st.form('questionnaire_'+S.session['session_id']):
        answers = {}
        for key, (question, anchors) in questions.items():
            answers[key] = st.radio(question, list(range(1,8)), index=None, horizontal=True,
                                    key='q_'+key, help=anchors)
        comment = st.text_area('Comment (optional)', max_chars=4000)
        if st.form_submit_button('Finish'):
            if any(v is None for v in answers.values()):
                st.error('Please answer every rating question.')
            else:
                enqueue('questionnaires', dict(session_id=S.session['session_id'], participant_id=S.session['participant_id'],
                                              condition=condition, is_pilot=pilot, **answers, comment=comment, submitted_at=now()))
                save_session(status='completed', completed_at=now())
                S.page = 'done'
                st.rerun()


if 'page' not in S:
    initialize()
backend = setting('STORAGE_BACKEND', os.environ.get('A3_STORAGE_BACKEND', 'supabase'))
store = None
try:
    store = get_store(backend, os.environ.get('A3_SQLITE_PATH', str(C.ROOT/'local.db')),
                      setting('SUPABASE_URL',''), setting('SUPABASE_SERVICE_KEY',''))
    if S.page in ('welcome','id'):
        store.health()
    if S.queue:
        flush(store, S.queue)
except Exception:
    st.warning('Study storage is currently unavailable. You can read the welcome page; please try again before starting.')

checkpoint_status()  # Hash once at startup, independent of lazy model load.
if 'admin' in st.query_params:
    from admin import render
    render(store, backend, st.query_params['admin'], setting('ADMIN_TOKEN',''))
    st.stop()
condition = st.query_params.get('condition','')
pilot = st.query_params.get('pilot','')=='1'
if condition not in ('baseline','ai'):
    st.title('Facial-expression labeling')
    st.info('Please use the Without AI or With AI link from the class sheet. The link must include ?condition=baseline or ?condition=ai.')
    st.stop()
if 'session' in S and (S.session['condition']!=condition or S.session['is_pilot']!=pilot):
    st.info('Please open this version in a new tab to keep the two sessions separate.')
    st.stop()
try:
    study, dev = data()
    lookup = {r['image_id']:r for r in study+dev}
except Exception:
    st.error('Study assets failed validation. Please contact the study organizer.')
    st.stop()

st.caption('CSE 594 · FACIAL-EXPRESSION LABELING'+(' · PILOT' if pilot else ''))
st.title('With AI' if condition=='ai' else 'Without AI')
pending_notice()
if S.page=='welcome':
    st.write('This CSE 594 class study takes about 5 minutes. You will label face images, and we will save your answers, ratings, and response times; your ID is used only to pair the two versions. Participation is voluntary; select Continue if you agree to take part.')
    if st.button('Continue'):
        S.page='id'
        st.rerun()
elif S.page=='id':
    with st.form('participant'):
        participant = st.text_input('Participant ID', help='Use your uniqname — use the same ID for both versions.')
        if st.form_submit_button('Continue to instructions'):
            try:
                participant = normalize_id(participant)
                if store is None:
                    st.error('Storage is unavailable. Please try again shortly.')
                else:
                    begin_session(participant)
                    st.rerun()
            except ValueError as error:
                if str(error)=='pool_exhausted':
                    st.error('No unseen study images remain for this ID. Please contact the study organizer.')
                elif str(error)=='history_changed':
                    st.error('Your other session changed. Please try again.')
                else:
                    st.error(str(error) if 'Use 3' in str(error) else 'Unable to start safely. Please try again shortly.')
            except Exception:
                st.error('Unable to reach study storage. Please try again shortly.')
elif S.page=='duplicate':
    st.success("You've already completed this version — thanks!")
elif S.page=='instructions':
    st.subheader('How to label a face')
    st.write('Choose the category that best describes the visible facial expression. Work quickly but carefully.')
    for label, cue in zip(C.LABELS, C.CUES):
        st.markdown(f'**{label.title()}:** {cue}')
    st.write('There are a few attention checks. Follow the written instruction for those trials, regardless of the face.')
    if condition=='ai':
        st.write('The AI shows a suggestion and an alternative. It is an automatic classifier that is often but not always right; you may choose any of the eight categories.')
        if S.session['ai_workflow']=='two_stage':
            st.write("First give your own answer; then you'll see the AI's suggestion and can keep or change your answer.")
    if S.sampling_warnings:
        st.info('Fewer unseen images remain, so this block may be shorter than usual.')
    if st.button('Start labeling'):
        S.page='trials'
        st.rerun()
elif S.page=='trials':
    render_trial()
elif S.page=='questionnaire':
    questionnaire()
elif S.page=='terminated':
    st.info("This session ended because the attention checks weren't passed. You can restart using the same link.")
    if st.button('Restart', disabled=bool(S.queue)):
        for key in list(S.keys()):
            del S[key]
        st.rerun()
elif S.page=='done':
    if S.queue:
        st.info('Your answers are complete, but saving is still pending. Please keep this tab open and select Retry saving.')
    else:
        version = 'With AI' if condition=='ai' else 'Without AI'
        st.success(f"Thanks! You've finished the {version} version. Please return to the class sheet.")
        if S.session['block_order']==1:
            st.write('Remember to complete the other version later with the same ID.')
