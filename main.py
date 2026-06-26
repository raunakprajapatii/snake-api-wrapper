from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import time

app = FastAPI()

HF = "https://rounakhii-serpentai.hf.space"


@app.post("/run/predict")
async def predict(file: UploadFile = File(...)):

    # -------------------------
    # STEP 1 Upload file
    # -------------------------

    upload = requests.post(
        f"{HF}/gradio_api/upload",
        files={
            "files": (
                file.filename,
                await file.read(),
                file.content_type
            )
        },
        timeout=60
    )

    upload.raise_for_status()

    uploaded_path = upload.json()[0]

    # -------------------------
    # STEP 2 Start prediction
    # -------------------------

    body = {
        "data": [
            {
                "path": uploaded_path,
                "meta": {
                    "_type": "gradio.FileData"
                }
            }
        ]
    }

    r = requests.post(
        f"{HF}/gradio_api/call/v2/classify",
        json=body,
        timeout=60
    )

    r.raise_for_status()

    event_id = r.json()["event_id"]

    # -------------------------
    # STEP 3 Wait
    # -------------------------

    while True:

        sse = requests.get(
            f"{HF}/gradio_api/call/classify/{event_id}",
            timeout=60
        )

        text = sse.text

        if "event: complete" in text:

            data = text.split("data:")[-1].strip()

            return {
                "data": data
            }

        time.sleep(1)