-- Run once in the Supabase SQL editor. No browser/public access is granted.
create table sessions (
  session_id uuid primary key,
  participant_id text not null,
  condition text not null check (condition in ('baseline','ai')),
  block_order int, ai_workflow text, is_pilot boolean not null default false,
  app_version text, seed bigint, trial_plan jsonb,
  status text not null default 'in_progress' check (status in ('in_progress','completed','terminated','abandoned')),
  attention_failures int default 0, user_agent text,
  started_at timestamptz default now(), completed_at timestamptz
);
create table trials (
  id bigint generated always as identity primary key,
  session_id uuid references sessions(session_id),
  participant_id text, condition text, is_pilot boolean, app_version text,
  trial_idx int, image_id text,
  is_attention_check boolean, attention_target text, attention_passed boolean,
  true_label text, ai_correct_frozen boolean,
  initial_label text, ai_top1 text, ai_top2 text, ai_probs jsonb, ai_source text, ai_inference_ms real,
  final_label text, correct boolean, confidence int,
  switched boolean, followed_ai_top1 boolean, chose_ai_top2 boolean,
  rt_initial_ms int, rt_final_ms int, rt_total_ms int,
  shown_at timestamptz, submitted_at timestamptz,
  unique (session_id, trial_idx), unique (participant_id, image_id)
);
create table questionnaires (
  session_id uuid primary key references sessions(session_id),
  participant_id text, condition text, is_pilot boolean,
  tlx_mental int, tlx_effort int, tlx_frustration int, tlx_performance int,
  trust int, usefulness int, comment text, submitted_at timestamptz default now()
);
create index sessions_participant on sessions(participant_id);
create index trials_session on trials(session_id);
alter table sessions enable row level security;
alter table trials enable row level security;
alter table questionnaires enable row level security;
revoke all on sessions, trials, questionnaires from anon, authenticated;
grant all on sessions, trials, questionnaires to service_role;
grant usage, select on sequence trials_id_seq to service_role;

-- Serialize session creation for one participant, including across conditions.
-- If history changed since sampling, reject so the server can resample.
create or replace function start_study_session(payload jsonb) returns void
language plpgsql security invoker set search_path = public as $$
declare pid text := payload->>'participant_id';
begin
  perform pg_advisory_xact_lock(hashtextextended(pid, 0));
  if exists (select 1 from sessions where session_id=(payload->>'session_id')::uuid) then return; end if;
  if not (payload->>'is_pilot')::boolean and exists (
    select 1 from sessions where participant_id=pid and condition=payload->>'condition'
      and status='completed' and not is_pilot
  ) then raise exception 'already_completed'; end if;
  if exists (
    select 1 from jsonb_array_elements(payload->'trial_plan') p
    join trials t on t.participant_id=pid and t.image_id=p->>'image_id'
  ) or exists (
    select 1 from sessions s, jsonb_array_elements(s.trial_plan) oldp,
      jsonb_array_elements(payload->'trial_plan') newp
    where s.participant_id=pid and s.status='in_progress' and s.condition<>payload->>'condition'
      and oldp->>'image_id'=newp->>'image_id'
  ) then raise exception 'stale_plan'; end if;
  update sessions set status='abandoned'
    where participant_id=pid and condition=payload->>'condition' and status='in_progress';
  insert into sessions select * from jsonb_populate_record(null::sessions, payload);
end $$;
revoke all on function start_study_session(jsonb) from public, anon, authenticated;
grant execute on function start_study_session(jsonb) to service_role;
