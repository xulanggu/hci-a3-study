# A3 — Facial-expression labeling study

## Implementation plan
1. Preserve A1's consent → instructions → unselected labeling → completion structure, replacing tweet stimuli with A2's frozen FER+ images.
2. Copy A2's Predictor verbatim and freeze its 240 study predictions, 40 development images, and checkpoint. Verify both prediction ranks and pixel provenance.
3. Implement P0 sampling, exposure-safe sessions, two-stage advice, questionnaires, SQLite/Supabase persistence, retries, model verification, exports, and tests.
4. Only after P0 passes, implement P1 admin monitoring and end-to-end AppTest simulations. Defer optional P2 resume/exposure balancing to preserve the specified uniform design.

Implementation and validation details follow below.

## Connection to A1 and A2

A1's app (`../Xulang_Guan_A1/a1-code/app.py`) contributed the simple consent/labeling/completion sequence, random stimuli, session identifiers, progress feedback, and `index=None` answer controls. A1's report emphasized concrete category guidance and avoiding unnecessary interaction. A3 therefore supplies the specified one-line cues and a stable image/answer layout. A1's append-at-completion CSV storage has been replaced with immediate, idempotent database writes. A1 code and data were not modified.

A2's submitted report (`../Xulang_Guan_A2/CSE_594_Xulang_Guan_Assignment2.pdf`) motivates top-1 plus top-2 assistance and testing whether people can correct classifier errors. A3 uses the actual artifacts in `../Xulang_Guan_A2/hai_a2_package`: `results/study_predictions.csv`, `data/study_manifest.csv`, `data/pilot_manifest.csv`, `models/emotion-ferplus-8.onnx`, and the `Predictor` class from `pipeline.py`. The 40 A2 **development/pilot images** serve only as attention-check stimuli; `?pilot=1` is a separate session tag and does not change the main stimulus pool.

A2's submitted PDF predicts **human + AI > human-only > AI-only**, whereas its internal `report_notes.json` predicts **human + AI > AI-only > human-only**. These are conflicting hypotheses, not measured human results. A3 does not encode either ranking as a result or alter the requested experiment. Its logged initial/final answers permit analysis of advice following, switching, and error correction.

The A2 Predictor class is copied **verbatim** into `vendor/a2_predictor.py`. A small surrounding shim provides constants and disables runtime downloads; an explicit local checkpoint is always supplied. Model input remains grayscale → **64×64 LANCZOS**, float32, raw 0–255, NCHW. The participant image is separately resized to **240×240 BICUBIC**. No preprocessing was re-derived. `data/provenance.json` records the copied class/source hashes. The current A2 pipeline file's hash differs from the earlier run's recorded whole-file hash; the prediction verification independently checks its current Predictor against all frozen outputs.

## Run locally

From the HCI directory, using Python **3.11**:

```sh
python3.11 -m venv a3/.venv
source a3/.venv/bin/activate
python -m pip install -r a3/requirements-dev.txt
A3_STORAGE_BACKEND=sqlite streamlit run a3/app.py
```

Open `http://localhost:8501/?condition=baseline` and `http://localhost:8501/?condition=ai`. SQLite writes to `a3/local.db`; use `A3_SQLITE_PATH` to select an isolated test database. The app intentionally defaults to Supabase if neither secrets nor the local environment explicitly select SQLite. It never silently substitutes SQLite when Supabase fails.

Alternatively, run from `a3/`, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, set `STORAGE_BACKEND = "sqlite"`, and run `streamlit run app.py`. Streamlit reads secrets relative to the launch directory (or its user settings), not the script directory. To use secrets when launching from HCI, put them in HCI's `.streamlit/secrets.toml` and ensure that file is ignored by your repository. Never commit the actual secrets. The example contains placeholders only.

```sh
python -m pytest a3/tests -q
python a3/scripts/verify_model.py
python a3/scripts/simulate.py
python a3/scripts/export_data.py
python a3/scripts/export_data.py --include-pilot
python a3/scripts/export_data.py --backend supabase --secrets a3/.streamlit/secrets.toml
```

`simulate.py` uses AppTest and a temporary SQLite database, deletes it on exit, and never writes study data. `--db /path/to/NEW-simulation.db` retains synthetic responses for inspection and refuses existing paths. Simulated participants are not evidence of human-study outcomes. Export defaults to excluding pilots; all three CSVs are written under `a3/exports/` with UTC timestamps. Empty tables still have headers. Supabase reads paginate beyond 1,000 rows. Spreadsheet formula prefixes in text fields are escaped during CSV export only.

## Human setup steps

1. **Create a Supabase project** in your own account. Copy its project URL and server-side service-role key (never an anonymous/public key).
2. **Run `a3/schema.sql`** once in that project's SQL editor. It creates the three tables, constraints, indexes, RLS with no public policies, and the service-role-only `start_study_session` RPC. The RPC serializes new sessions for the same participant.
3. Put the **`a3/` directory**, including its checkpoint, all 280 images, and `requirements.txt`, in the GitHub repository used by Streamlit. Keep course assets private in accordance with the preserved `DATA_USE.txt`; do not upload your entire HCI directory, personal data, SQLite files, exports, or secrets. In **Streamlit Cloud → app settings → Secrets**, paste the example keys with real values: `STORAGE_BACKEND="supabase"`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, and a long random `ADMIN_TOKEN` (for example, generate one with `python -c "import secrets; print(secrets.token_urlsafe(32))"`).
4. **Deploy in Streamlit Community Cloud**, selecting main file **`a3/app.py`** and **Python 3.11**. The adjacent `a3/requirements.txt` pins direct dependencies. The application has no dependencies on A1/A2 directories at runtime. Do not select SQLite for hosting.
5. Open both participant links: **`https://<app>.streamlit.app/?condition=baseline`** and **`https://<app>.streamlit.app/?condition=ai`**. Put these into the class sheet in the assigned counterbalanced order. Open **`https://<app>.streamlit.app/?admin=<ADMIN_TOKEN>`** privately and confirm storage connected, checkpoint matched, and no unexpected fallback usage. Do not share the admin URL with participants.
6. **Pilot both links with `&pilot=1`**, use the same pilot ID for both, check the exported records, and confirm live advice is recorded. Repeating pilots still excludes previously seen images; use a dedicated pilot ID. Before the November 12, 2026, 10:00 am ET session, run a hosted multi-user pilot, confirm the app is awake, and verify Supabase persists responses after a browser refresh. Local tests do not establish 30-client hosted capacity.

No cloud project, credentials, deployment URL, or existing remote destination was supplied. The production adapter and SQL are provided, but actual Supabase connectivity/RLS and the hosted deployment must be verified in your account. The specified Streamlit/Supabase deployment is used; the Sites plugin's Worker hosting is not a substitute for this Python study app.

## Design decisions and assumptions

- Main entry is **HCI/a3/app.py**, matching the requested `a3/` path. `Xulang_Guan_A3/requirement.md` remains the input specification.
- Default workflow is `two_stage`. Change `AI_WORKFLOW` in `config.py` to `one_stage` before collection if needed; both are tested. Do not change workflow mid-study without a documented protocol change. Bump `APP_VERSION` for any collection-time hotfix.
- Each block ordinarily has 20 main trials (16 frozen-correct, 4 frozen-wrong) and two development-image checks. Sampling is uniform within the two strata, deterministic from the saved seed, and shuffled. This is an **80% frozen-correct experimental sample**, not the 77.92% accuracy of the complete 240-image A2 set. Analyze conditions on the actual sampled trials; do not treat these two denominators as interchangeable.
- IDs are stripped/lowercased and must match `^[a-z0-9_]{3,20}$`. The other condition's completed **non-pilot** session determines block order. Pilot completion does not change a real participant's block order.
- Every stimulus gets an exposure row **before display**, even if the participant never answers. Refresh creates a new session and abandons the prior in-progress session for that ID/condition. Submitted and unanswered exposures, pilots, terminated sessions, and both conditions all exclude images from future sampling. Development check images are also excluded, so repeat pilots can eventually exhaust that pool.
- Active plans in the other condition are temporarily reserved to avoid overlapping simultaneous tabs. This is an additional conservative exclusion beyond already-shown images. A participant should use one tab at a time. Unique `(participant_id, image_id)` and `(session_id, trial_idx)` constraints protect against duplicate exposures/writes; session creation uses a transaction/advisory lock. Unshown images in an abandoned plan become eligible again.
- Exposure bookkeeping happens just before the browser displays the image; a disconnect in that tiny interval may exclude an image that was never visibly rendered. This conservative over-exclusion is preferable to repeat exposure.
- Pool exhaustion samples only what remains **within each requested stratum**, without replacement or rebalancing. Warnings go to server logs and a shorter-block notice appears. If no main trials remain, a new block is refused. Attention checks are reduced only if unique development images or valid positions are exhausted; admin validity then excludes blocks with fewer than two checks.
- Writes make three total attempts with exponential backoff. A FIFO session-state queue preserves exposure → answer → questionnaire → completion order. Trial upserts are idempotent, and a session-state guard prevents double submission. Completed/terminated status is queued only after preceding records. A done acknowledgment is withheld until all writes succeed.
- Answer-save outages keep submitted rows queued and preserve progress. A **new face or advice reveal waits for durable logging**, with an explicit Retry button. The interface remains viewable with unavailable storage, but new data collection requires storage. This conservative behavior preserves cross-refresh exposure exclusion and advice auditability. Pending answers remain in Streamlit memory and would be lost if the user closes/refreshes the tab during an outage; the app asks them to keep it open.
- No model probabilities, reference labels, correctness feedback, or measured AI accuracy are shown to participants. Baseline advice fields are null; `ai_correct_frozen` is retained solely for sampling/analysis. Attention rows have null reference/correctness/AI/confidence fields; their target and pass flag are logged instead.
- Live inference runs once at the first actual advice request, with the model cached per process and one CPU thread per inference session. Any load/prediction exception uses the actual frozen A2 advice (`cached_fallback`). A checkpoint mismatch also forces fallback. The verifier never permits fallback.
- Timings are **server-side and include network latency**. `rt_initial_ms` measures first image rendering to the initial/single submit; `rt_final_ms` measures advice reveal to final submit for two-stage trials; total is their sum. Model inference time is separately logged; the between-stage inference wait is not added to the two-stage response-time sum. Rerenders do not reset timers; validation errors count toward response time. Single-stage final time is null.
- `tlx_performance` retains the original requested direction: **1 = Perfect, 7 = Failure**. Higher ratings indicate worse perceived performance. These four items are a raw TLX subset, not a weighted/full NASA-TLX score. Trust/usefulness are null in baseline questionnaires.
- Admin valid counts use the conservative definition: completed, non-pilot, **all** scheduled checks passed, at least two checks. A completed block with one failed check is still exported and included in completed counts, but not valid counts. Both completed-pair and valid-pair totals are shown. Fallback share uses all non-pilot advice exposures, including unanswered ones.
- P2 resume and global least-exposure sampling are intentionally deferred. The specified P0 refresh policy and uniform sampling remain in effect.
- The supplied data notice is preserved. The checkpoint and selected images add **35,534,697 bytes (33.89 MiB)**; manifests/source and application files add a small amount beyond that. All paths resolve relative to `app.py` through `config.ROOT`.

## Data dictionary

Times are UTC ISO-8601 strings in SQLite and `timestamptz` in PostgreSQL. Booleans export as True/False. Null means not applicable or not yet submitted, never an implicit negative response. `trial_idx` is **zero-based**; displayed progress is one-based. Use `submitted_at IS NOT NULL` to select answered trials, then exclude attention checks for main-task analyses. Join questionnaire/session records by `session_id`, and pair conditions by normalized `participant_id`.

### sessions

| Column | Meaning |
|---|---|
| session_id | UUID for a single attempt; primary key. |
| participant_id | Normalized pairing ID. |
| condition | `baseline` or `ai`. |
| block_order | 1 unless the other condition has a completed non-pilot attempt, otherwise 2. |
| ai_workflow | Configured `two_stage` or `one_stage`; recorded in both conditions. |
| is_pilot | Whether the URL contained `pilot=1`. |
| app_version | Version at session creation. |
| seed | 52-bit random seed used to create this plan. |
| trial_plan | Ordered JSON list of image_id, is_attention_check, attention_target; includes checks. |
| status | in_progress, completed, terminated, or abandoned. |
| attention_failures | Number of submitted incorrect attention responses. |
| user_agent | Browser User-Agent if available, otherwise empty string. |
| started_at | Session creation timestamp. |
| completed_at | Questionnaire-completion timestamp; null for unfinished/terminated attempts. |

### trials

| Column | Meaning |
|---|---|
| id | Generated database row ID. |
| session_id | Attempt UUID; foreign key to sessions. |
| participant_id | Normalized pairing ID; together with image_id prevents repeat exposure. |
| condition | Session condition. |
| is_pilot | Session pilot flag. |
| app_version | App version when the exposure was created. |
| trial_idx | Zero-based position in this session's plan; unique within session. |
| image_id | Frozen A2 trial_id (for example fer_0034675). |
| is_attention_check | True for a development-image instruction check. |
| attention_target | Random instructed label for checks; null on main trials. |
| attention_passed | Whether submitted answer matches check target; null otherwise. |
| true_label | FER+ ≥6/10 majority reference on main trials; null on checks. |
| ai_correct_frozen | Frozen A2 top-1 equals reference; both main-trial conditions, null on checks. |
| initial_label | Independent stage-1 label for two-stage AI trials; otherwise null. |
| ai_top1 | Actual displayed suggestion; null before reveal and in baseline/checks. |
| ai_top2 | Actual displayed alternative; null when advice not shown. |
| ai_probs | Eight probabilities in LABELS order, logged only; null when advice not shown. |
| ai_source | live or cached_fallback; null when advice not shown. |
| ai_inference_ms | Live call elapsed time including first-load cost, if any; null for fallback/no advice. |
| final_label | Submitted main/check answer; null on exposure-only rows. |
| correct | Final main answer matches reference; null for checks or unsubmitted rows. |
| confidence | Required 1–5 final-confidence rating; null on checks/unsubmitted rows. |
| switched | Initial and final labels differ, only for completed two-stage AI trials. |
| followed_ai_top1 | Final label equals actual displayed suggestion; null without advice/final answer. |
| chose_ai_top2 | Final label equals actual displayed alternative; null without advice/final answer. |
| rt_initial_ms | Image-to-initial/single-submit elapsed milliseconds. |
| rt_final_ms | Advice-reveal-to-final-submit milliseconds, only for two-stage AI. |
| rt_total_ms | rt_initial_ms + (rt_final_ms or 0), available after final submit. |
| shown_at | Timestamp recorded immediately before first display/exposure persistence. |
| submitted_at | Final/check answer timestamp; null for exposure-only or stage-1-only records. |

### questionnaires

| Column | Meaning |
|---|---|
| session_id | Attempt UUID; primary key and sessions foreign key. |
| participant_id | Normalized pairing ID. |
| condition | Session condition. |
| is_pilot | Session pilot flag. |
| tlx_mental | Mental demand: 1 Very low – 7 Very high. |
| tlx_effort | Effort: 1 Very low – 7 Very high. |
| tlx_frustration | Frustration: 1 Very low – 7 Very high. |
| tlx_performance | Perceived performance: 1 Perfect – 7 Failure. |
| trust | AI suggestions trusted: 1 Strongly disagree – 7 Strongly agree; AI only. |
| usefulness | AI suggestions helped accuracy: same agreement scale; AI only. |
| comment | Optional free text (up to 4,000 characters); empty string if omitted. |
| submitted_at | Questionnaire submission timestamp. |

## Implementation references

The persistence adapter uses Supabase's documented [Python upsert API](https://supabase.com/docs/reference/python/upsert) with explicit conflict columns. Browser metadata uses [Streamlit context headers](https://docs.streamlit.io/1.46.0/develop/api-reference/caching-and-state/st.context). Tests use the [Streamlit AppTest interface](https://github.com/streamlit/streamlit/blob/develop/lib/streamlit/testing/v1/app_test.py). These references support API choices; the supplied assignment determines the study protocol.

## Validation completed on November 11, 2026

- **27 pytest tests passed**. Coverage includes 100 deterministic sampler seeds, 16/4 strata, exhausted pools, derived fields, normalized IDs, SQLite round-trip/idempotent trial writes, 30 concurrent SQLite writers, retry ordering, both participant orders, two-stage/one-stage flows, missing selections/confidence/questionnaire ratings, refresh exclusion, attention termination/restart, pilot repeats, startup and mid-block outages, durable advice logging, admin authentication/downloads, and offline Supabase adapter pagination/RPC contracts.
- **Model: 240/240 top-1 AND top-2 matches**, checkpoint SHA-256 matches A2, accuracy **0.7791666667 (187 correct / 53 wrong)**, all 280 image pixel hashes verified. Per-class recall is in `data/model_verification.json`; the Predictor class is byte-for-byte identical to the current A2 class.
- **AppTest simulator passed:** four completed blocks covering both orders, one attention-terminated block, one successful restart (six sessions, four questionnaires). The retained verification run had 100 exposure/answer rows; this varies because termination positions are random. CSV export successfully produced all three tables with the expected rows and full headers. Synthetic response files remain outside this deliverable.
- **Environment limits:** package-download networking was unavailable. Cached packages supported the full UI suite on Python 3.11 with Streamlit 1.64.0, numpy 2.4.6, Pillow 12.3.0, and pandas 2.3.3. Actual ONNX verification used A2's existing Python 3.12.6 environment (onnxruntime 1.24.4, onnx 1.21.0, numpy 2.5.3, Pillow 12.3.0). UI flows exercised real frozen fallback; a controlled live-adapter test verifies reveal-time logging and timing. A **fresh complete Python 3.11 installation and live inference in the deployed process still require confirmation**. Do not interpret the passing offline adapter tests as a live Supabase validation.
- This machine's Anaconda `readline` extension crashes pytest's capture initialization. The local command used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p no:capture a3/tests -q`; no study tests were skipped. A normal clean Python installation can use the standard command above.
- Full machine-readable evidence and remaining verification items are in `data/validation.json`. No hosted deployment, real participant responses, or human pilot are claimed.

## Deployment preparation follow-up

`scripts/setup_local.sh` installs a clean Python 3.11 environment, checks dependencies, verifies the live model, and runs the tests. Run `bash a3/scripts/setup_local.sh --serve` from a network-enabled terminal to also launch the local app.

`scripts/prepare_deployment.py /path/to/NEW-directory` assembles a standalone repository tree with `a3/app.py`, all frozen assets, a source checksum manifest, and a GitHub Actions workflow. It uses an explicit file allowlist and refuses an existing destination. The prepared repository keeps CSV bytes unchanged with `.gitattributes`. Its workflow validates a fresh Python 3.11 installation, all live predictions, pytest, and the simulator on GitHub's runner; it has not run until the repository is pushed and Actions completes.

A clean local Git repository was prepared at `../Xulang_Guan_A3/deployment/` in this workspace. GitHub and Supabase integrations were discovered but were not connected at preparation time. No remote repository, Supabase project, or Streamlit deployment has been created. The local environment also blocks listening on a localhost port, so browser preview could not be launched here; AppTest validation remains available.
