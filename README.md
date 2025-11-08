# Network Intrusion Detection – Backend (FastAPI + scikit-learn)

A FastAPI backend exposing a binary network intrusion classifier (BENIGN vs ATTACK). It accepts single-flow JSON, applies the same preprocessing pipeline used at training time (encoders + scaler + engineered features), and returns both class and probabilities.

## Key Files

- Backend API
  - `backend/app.py` – FastAPI app, CORS, `/predict/` endpoint, model loading.
- Preprocessing & Artifacts
  - `backend/model/preprocess.py` – Feature engineering, categorical encoding, scaling, feature ordering.
  - `backend/model/trained_model.pkl` – Pickled dict containing the trained scikit-learn model under key `"model"`.
  - `backend/model/scaler.pkl` – Fitted scaler used at training time.
  - `backend/model/label_encoder.pkl` – Label encoder fitted on target classes.
  - `backend/model/protocol_encoder.pkl` – Encoder for `protocol` feature (must contain all runtime categories).
  - `backend/model/state_encoder.pkl` – Encoder for `state` feature (must contain all runtime categories).
  - `backend/model/selected_features.txt` – Saved feature list (loaded but not currently used in preprocessing).

Note: The optional CSV upload endpoint in `backend/app.py` is commented out.

## Runtime Requirements

- Python 3.9+ recommended
- Packages: fastapi, uvicorn, pydantic, numpy, pandas, scikit-learn (for model artifacts compatibility), pickle (stdlib)

If you don’t use a requirements file, install quickly with:
```
pip install fastapi uvicorn pydantic numpy pandas scikit-learn
```

## Project Setup (Windows, VS Code)

1) Create and activate a virtual environment:
```
python -m venv .venv
.venv\Scripts\activate
```

2) Install dependencies:
```
pip install -r requirements.txt
```
or (if no requirements.txt):
```
pip install fastapi uvicorn pydantic numpy pandas scikit-learn
```

3) Ensure package structure (needed for `from .model.preprocess import preprocess_input`):
- Add empty `__init__.py` files if missing:
  - `backend/__init__.py`
  - `backend/model/__init__.py`

4) Verify model artifacts exist:
- `backend/model/trained_model.pkl`
- `backend/model/scaler.pkl`
- `backend/model/label_encoder.pkl`
- `backend/model/protocol_encoder.pkl`
- `backend/model/state_encoder.pkl`
- `backend/model/selected_features.txt`

## Running the API

From the project root:
```
python -m uvicorn backend.app:app --reload
```

- Open Swagger UI: http://127.0.0.1:8000/docs
- CORS allows `http://localhost:5173` (Vite/React default). Adjust in `backend/app.py` if your frontend runs elsewhere.

## API

### POST /predict/

- Purpose: Single-record inference.
- Content-Type: application/json

Request body schema (validated by Pydantic `NetworkInput`):
```json
{
  "source_bytes": 1234.0,
  "dest_bytes": 567.0,
  "source_pkts": 10.0,
  "dest_pkts": 8.0,
  "tcp_win_fwd": 5120.0,
  "tcp_win_bwd": 4096.0,
  "mean_seg_size_fwd": 200.0,
  "mean_seg_size_bwd": 180.0,
  "duration": 2.5,
  "protocol": "tcp",
  "state": "ESTABLISHED"
}
```

Successful response:
```json
{
  "predicted_prob_benign": 0.8732,
  "predicted_prob_attack": 0.1268,
  "predicted_label": 0,
  "predicted_class": "BENIGN"
}
```

- `predicted_label`: 0 = BENIGN, 1 = ATTACK.
- Probability indices follow scikit-learn’s `predict_proba` ordering.

### Example curl
```
curl -X POST "http://127.0.0.1:8000/predict/" ^
  -H "Content-Type: application/json" ^
  -d "{\"source_bytes\":1234,\"dest_bytes\":567,\"source_pkts\":10,\"dest_pkts\":8,\"tcp_win_fwd\":5120,\"tcp_win_bwd\":4096,\"mean_seg_size_fwd\":200,\"mean_seg_size_bwd\":180,\"duration\":2.5,\"protocol\":\"tcp\",\"state\":\"ESTABLISHED\"}"
```

## How It Works

### Model Loading

`backend/app.py` loads a pickle once at startup:
- `trained_model.pkl` is unpickled to `model_dict`.
- The actual model is taken from `model_dict["model"]`.

This keeps inference fast and stateless across requests.

### Preprocessing Pipeline

`backend/model/preprocess.py` is designed to replicate training-time transforms:

1) Input to DataFrame
   - Converts the single JSON payload into a one-row DataFrame.

2) Feature engineering (derived, numeric):
   - `avg_pkt_size` = (source_bytes + dest_bytes) / (source_pkts + dest_pkts + 1e-6)
   - `pkt_ratio` = source_pkts / (dest_pkts + 1)
   - `byte_ratio` = source_bytes / (dest_bytes + 1)
   - `req_resp_avg_pkt_ratio` = (source_bytes/(source_pkts+1)) / (dest_bytes/(dest_pkts+1) + 1e-6)
   - `win_payload_ratio` = (tcp_win_fwd + tcp_win_bwd) / (source_bytes + dest_bytes + 1)
   - `bytes_per_sec` = (source_bytes + dest_bytes) / (duration + 1)
   - `pkts_per_sec` = (source_pkts + dest_pkts) / (duration + 1)

3) Categorical encoding:
   - `protocol_encoded` = `protocol_encoder.transform(protocol)`
   - `state_encoded` = `state_encoder.transform(state)`
   - Original `protocol`, `state` columns are dropped after encoding.

   Important: Incoming categories must exist in the encoders’ vocabularies. Unknown categories will raise an error.

4) Feature ordering and dtype:
   - Columns arranged to the exact `feature_order` expected by the model.
   - Cast to `float32`.

5) Scaling:
   - `scaler.transform(...)` applied to the ordered features.
   - Returns a DataFrame matching `feature_order`.

Note: `selected_features.txt` is loaded but not currently used to slice features; the pipeline relies on the hard-coded `feature_order`.

### Inference

`/predict/` performs:
- `X_processed = preprocess_input(data_dict)`
- `y_pred_proba = model.predict_proba(X_processed)` → `[P(0=BENIGN), P(1=ATTACK)]`
- `y_pred = model.predict(X_processed)` → `[0|1]`
- Converts to JSON with both numeric label and string class.

### Error Handling

- All exceptions in `/predict/` return HTTP 500 with the exception message.
- Full traceback is printed to server logs for debugging.
- Common runtime errors and fixes:
  - `FileNotFoundError`: Ensure all `.pkl` and `.txt` artifacts exist at `backend/model/`.
  - `ValueError` (unknown category): Ensure `protocol`/`state` values were seen during training or extend encoders.
  - `ImportError` (relative import): Ensure `backend/` and `backend/model/` contain `__init__.py`.

## Frontend Integration

- CORS is configured to allow `http://localhost:5173`.
- Adjust `allow_origins` in `backend/app.py` for different domains or add `"*"` during development.

## Performance Notes

- Model and artifacts load once at startup.
- Preprocessing uses vectorized pandas/numpy ops; single-record latency is dominated by Python/Model inference.
- For production, consider:
  - Running behind an ASGI server with workers (e.g., `uvicorn --workers 2`).
  - Enabling structured logging and metrics.
  - Pinning scikit-learn/numpy versions consistent with training.

## Extensibility

- Batch predictions: The commented `/upload-csv/` endpoint in `backend/app.py` shows how to read a CSV, preprocess, and annotate predictions. It can be re-enabled and adapted if needed.
- Health checks: Add `/health` returning simple status for monitoring.
- Validation: Extend `NetworkInput` with ranges and regex for stricter validation.

## Troubleshooting

- VS Code Debug: Use a launch config with module `"uvicorn"` and args `["backend.app:app", "--reload"]`.
- Relative Paths: Start the server from project root so artifact paths like `backend/model/...` resolve correctly.
- Data Types: All numeric inputs are expected as numbers; strings for `protocol` and `state`.

## License

Internal/Academic use. Add a license file if distributing.