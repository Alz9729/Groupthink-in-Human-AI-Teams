# IRIS v2

## Flow
1. Participant ID screen.
2. **Ask anything**: the participant describes the game situation, taps **+** to add Option A / Option B, and can use the mic to dictate.
3. Two option cards appear. The participant explains **why each option makes sense** and picks the one they'd choose.
4. Both reasons are checked for gibberish/bad faith: pattern rules first, then a quick AI check. IRIS stays locked until both pass.
5. IRIS always gives its opinion beside the cards. There are no experimental conditions: IRIS has no assigned role or valence framing and answers however it thinks best.
6. Up to 5 follow-up questions. The participant can switch A/B or edit a reason at any time (edits are re-checked), then locks in a final choice.

## Run
```
cd IRIS_v2
python -m venv venv
venv\Scripts\pip install -r requirements.txt
set OPENAI_API_KEY=sk-...        (or put the key alone in api_key.txt)
venv\Scripts\python backend.py
```
Open http://localhost:5000.

Dry run without OpenAI (canned IRIS replies, AI reason check skipped):
```
venv\Scripts\python backend.py --mock
```

## Logs
- `experiment_log.json`: index of sessions (session id → log file).
- `chat_logs/<participant>_<timestamp>.json`: one file per session. Each decision records:
  - `situation`, `option_a`, `option_b`
  - `reasons` (final text), `reason_edits` (old → new), `rejected_attempts` (every gibberish/bad-faith attempt and why)
  - `initial_choice`, `choice_history` (initial / switch / final, with how many follow-ups had happened), `final_choice`, `changed_mind`, `times_switched`
  - `iris_opinion` and `followups` (question, answer, the player's choice at that moment, and any switch/edit IRIS was told about)
  - `iris_leanings`, `iris_changed_stance`, `final_matches_iris`: which option each IRIS reply favoured (labelled by a separate background AI call after each reply, so IRIS's own replies carry no extra instructions)
- `GET /api/stats` (total sessions), `GET /api/export-all-logs` (every session in one JSON).

## Settings (top of backend.py)
`MODEL`, `GAME_DOCUMENT_PATH`, `MAX_FOLLOWUPS`, `MIN_REASON_WORDS`.
