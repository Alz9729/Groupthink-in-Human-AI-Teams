# Groupthink in Human-AI Teams

Materials, software and data for a human-subjects study of how an AI teammate's **communication style** affects people's decision-making and their tendency toward groupthink.

Participants play through decision points in the horror game *Until Dawn* (Chapter 6) alongside **IRIS**, an LLM-powered "game buddy" that gives its opinion on the choices in front of them. The study has been run in two iterations:

| Iteration | What IRIS does | Design |
|---|---|---|
| **IRIS v1** | IRIS is assigned a **role** (closed / neutral / open) and a **valence framing** (positive / negative) that shapes how it argues for Option A | Between-subjects, 7 conditions (6 role × valence cells + a control) |
| **IRIS v2** | IRIS has no assigned role or framing and answers however it thinks best. Participants first write their own reasons for each option, then discuss with IRIS | Single condition; logs track whether and why participants change their minds |

## Repository layout

```
.
├── A1–A7 …              Study materials for the IRIS v1 (Until Dawn) study
├── B1–B3 …              Study materials for the IRIS 2.0 study
├── IRIS_v1/             v1 web app, experiment log and 35 participant chat logs
└── IRIS_v2/             v2 web app (see IRIS_v2/README.md for details)
```

### Study materials

| File | Contents |
|---|---|
| `A1-Until-Dawn-Pre-survey.pdf` | Pre-study survey (v1) |
| `A2-Until-Dawn-Post-Survey.pdf` | Post-study survey (v1) |
| `A3-Informed-Consent-Form.docx` | Informed consent form |
| `A4-Experiment-Script.docx` | Script for the experimenter to follow during sessions |
| `A5-Observation-Sheet.pdf` | Observer's sheet for recording behaviour (v1) |
| `A6-Chapter6-Transcript.pdf` | Transcript of the game chapter used as the study scenario |
| `A7-Charecter-Profile.pdf` | Character profiles for the game |
| `B1-IRIS-2.0-Pre-Survey.docx` | Pre-study survey (v2) |
| `B2-IRIS-2.0-Post-Survey.docx` | Post-study survey (v2) |
| `B3-Observation-Sheet.docx` | Observer's sheet for recording behaviour (v2) |

## IRIS v1

A Flask backend plus a single-page chat frontend. When a session starts, the backend assigns the participant a condition, and the system prompt sent to the model is built from that condition.

**Conditions**

- **AI role:** *closed* (commanding, shuts down alternatives), *neutral* (one calm recommendation and one reason), or *open* (devil's advocate, probing questions).
- **Valence framing:** *positive* (mentions only the benefits of Option A) or *negative* (mentions only the downsides of Option B).
- **Control:** no role or framing.

**Data**

- `IRIS_v1/chat_logs/` holds one JSON file per participant, named `<id>_<condition>.json`. Each contains the condition, the full message history with timestamps, and any feedback submitted.
- `IRIS_v1/experiment_log.json` holds the per-condition counts and an index of sessions.
- Participant IDs run from `001` to `035`.

**Running it**

v1 reads its API key from a constant at the top of `backend.py`. Paste a key in locally and **do not commit it**. It uses the same dependencies as v2.

```bash
cd IRIS_v1
pip install -r ../IRIS_v2/requirements.txt
python backend.py
```

Open http://localhost:5000.


## IRIS v2

IRIS v2 drops the manipulated conditions. The participant describes a game situation and two options, explains why each makes sense, and picks one. Both reasons are screened for gibberish and bad faith (pattern rules, then a quick AI check). Only then does IRIS give its opinion, and the participant can ask up to 5 follow-ups, switch their choice, or revise a reason before locking in a final answer.

It logs the participant's initial and final choice, every switch, every reason edit, each IRIS reply, and which option each reply favoured, so changes of mind can be traced back to the conversation.

**Quick start (Windows)**

```
cd IRIS_v2
python -m venv venv
venv\Scripts\pip install -r requirements.txt
set OPENAI_API_KEY=sk-...
venv\Scripts\python backend.py
```

Open http://localhost:5000. To try the full flow without calling OpenAI, run `venv\Scripts\python backend.py --mock`.

See [IRIS_v2/README.md](IRIS_v2/README.md) for the participant flow, the log schema, the API endpoints and the settings.

## Requirements

- Python 3 (recent versions recommended; required by Flask 3)
- An OpenAI API key (not needed for `--mock` runs of v2)
- `flask`, `flask-cors`, `openai`, `PyPDF2` (pinned in `IRIS_v2/requirements.txt`)

Both versions use the `gpt-5-nano` model and ground IRIS in the Chapter 6 transcript PDF in their own folder.

## Data and ethics

The chat logs contain participant responses collected under the informed-consent form in this repo (`A3-Informed-Consent-Form.docx`). Logs are keyed by study ID. Do not add anything that could identify a participant, and do not commit API keys.
