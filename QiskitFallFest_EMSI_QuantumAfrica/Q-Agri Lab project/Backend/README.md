# Q-Agri Lab backend

FastAPI services for the official Moroccan crop-classification challenge.
The implementation lives under `app/`; the older top-level packages remain
small compatibility imports for existing development scripts.

## Run locally

From this directory, install the backend dependencies and start the API:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger. The API reads the repository's
`.env.example` by default, then falls back to `.env`; set `QAGRI_ENV_FILE` to
select another file. Keep credential files private.

Optional IBM Runtime dependencies are in `requirements-hardware.txt`. IBM
credentials are needed only for `/api/quantum/ibm/status` and `/api/quantum/ibm/run`.
Hardware execution submits a real job to the selected IBM backend.

## Challenge data and protocol

The backend reads the repository's supplied train, validation, test and PCA4
files. At load time it checks manifest counts, CSV schemas, finite features,
disjoint IDs, labels and alignment with the PCA view. It does not alter source
data or use test labels. The API reports the official 914 / 150 / 221 split
sizes and preserves the class order alfalfa, maize, wheat.

The RBF-SVM baseline uses the frozen settings from `configs/reference.json`
and the 13 supplied CSV features. The QNN uses the challenge's supplied
four-feature PCA view. It defaults to the official fixed 100-row starter
subset for a quick run; set `training_size` to 600 or 914 for larger runs.
Exact fidelity-kernel experiments are restricted to the 100-row starter
subset because their cost grows quadratically. A comparison that includes a
quantum model defaults all models to the same 100 training rows.

## Main API routes

- `GET /health` and `GET /api/experiments/dataset`
- `POST /api/experiments` and `POST /api/experiments/compare`
- `POST /api/quantum/experiment`, `POST /api/quantum/circuit`
- `GET /api/quantum/status`, `GET /api/quantum/ibm/status`
- `POST /api/quantum/ibm/run` (submits hardware work)
- `POST /api/agent/chat` (Responses API with bounded experiment tool calls)
- `GET /api/results/reference`, `POST /api/results/report`

## Tests

```powershell
python -m pytest -q
```

The quantum-model smoke tests run local simulations and take longer than the
classical/API checks. Automated tests do not submit hardware jobs.
