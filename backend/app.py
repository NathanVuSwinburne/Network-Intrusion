from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
from .model.preprocess import preprocess_input
import pandas as pd
import json
import traceback
import os
app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("backend/model/trained_model.pkl", "rb") as f:
    model_dict = pickle.load(f)

STATISTICS_FILE = "backend/statistics.json"

model = model_dict["model"]  # Access the actual ML model

# Pydantic schema for validation
class NetworkInput(BaseModel):
    source_bytes: float
    dest_bytes: float
    source_pkts: float
    dest_pkts: float
    tcp_win_fwd: float
    tcp_win_bwd: float
    mean_seg_size_fwd: float
    mean_seg_size_bwd: float
    duration: float
    protocol: str
    state: str

@app.post("/predict/")
async def predict(input_data: NetworkInput):
    try:
        # Convert to DataFrame
        data_dict = input_data.dict()

        # Preprocess using same training pipeline
        X_processed = preprocess_input(data_dict)

        # Get prediction probabilities
        y_pred_proba = model.predict_proba(X_processed)
        y_pred = model.predict(X_processed)
        
        prediction = int(y_pred[0])
        prob_benign = float(y_pred_proba[0][0])  # Probability of class 0 (BENIGN)
        prob_attack = float(y_pred_proba[0][1])  # Probability of class 1 (ATTACK)

        # Return detailed result
        label = "ATTACK" if prediction == 1 else "BENIGN"

        return {
            "predicted_prob_benign": prob_benign,
            "predicted_prob_attack": prob_attack,
            "predicted_label": prediction,
            "predicted_class": label
        }

    except Exception as e:
        print("Error details:", traceback.format_exc())  # Detailed error log
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    try:
        if not os.path.exists(STATISTICS_FILE):
            raise HTTPException(
                status_code=404,
                detail=f"Statistics file '{STATISTICS_FILE}' not found. Please generate statistics first."
            )

        with open(STATISTICS_FILE, 'r') as f:
            data = json.load(f)

        # Return the entire JSON object as-is
        return JSONResponse(content=data)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Error decoding JSON file. The file may be corrupted."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# =====================================================
# UPLOAD & PREDICTION ENDPOINT
# =====================================================
# @app.post("/upload-csv/")
# async def upload_csv(file: UploadFile = File(...)):
#     """
#     Upload a CSV, preprocess it using the same function as training,
#     and get predictions from the trained model.
#     """
#     try:
#         # Step 1: Read uploaded CSV
#         contents = await file.read()
#         df = pd.read_csv(io.BytesIO(contents))

#         # Step 2: Preprocess using the exact same logic from preprocess.py
#         X_processed = preprocess_input(df)

#         # Step 3: Predict using trained model
#         y_pred = model.predict(X_processed)
#         df['prediction'] = y_pred

#         # Step 4: Summarize results
#         benign = int((df['prediction'] == 0).sum())
#         attack = int((df['prediction'] == 1).sum())
#         summary = {"Benign": benign, "Attack": attack}

#         # Step 5: Return small preview and summary
#         preview = df.head(10).to_dict(orient="records")

#         return JSONResponse(content={
#             "message": "success",
#             "summary": summary,
#             "preview": preview
#         })

#     except Exception as e:
#         return JSONResponse(
#             status_code=400,
#             content={"message": f"Error processing the file: {str(e)}"}
#         )
