# Vidog API: Plug-and-Play Developer Integration Guide

This guide is designed for the frontend and backend developer of the web portal. 

You do **not** need to handle video frame sampling, cropping, model loading, or embedding math. You only need to make **two simple HTTP calls** to our microservice.

---

## 1. Running the Model Microservice

The microservice runs on port `8000`:

```bash
cd vidog_pure_cosine_similarity
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

* **Health Check**: `GET http://localhost:8000/health`

---

## 2. CALL 1: Registration (`POST /api/v1/register`)

Trigger this call when the user clicks **"Register"** or submits a **New Case Report**.

### Request Specifications
* **URL**: `http://localhost:8000/api/v1/register`
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`

### Form Fields (Key-Value)
| Field Key | Type | Required? | Description |
| :--- | :--- | :--- | :--- |
| `case_id` | `string` | **Yes** | Your database Case ID / UUID (e.g. `"case-00124"`). |
| `dog_name` | `string` | No | Name of the dog if known (defaults to `"Unknown"`). |
| `metadata` | `string` (JSON) | No | Any raw form fields from your website (e.g. `JSON.stringify(formData)`). |
| `nose_video` | `File` (MP4/MOV) | Optional* | Snout close-up video. Can also be named `nose_<anything>.mp4`. |
| `body_video` | `File` (MP4/MOV) | Optional* | Full body video. Can also be named `body_<anything>.mp4`. |

*\*Note: At least one video must be provided. You can either pass them under the exact field names `nose_video` and `body_video`, or upload them under a multiple file input field `videos` where one has `"nose"` in the filename and the other has `"body"` in the filename (e.g. `nose_sheru.mp4` and `body_sheru.mp4`). Our service automatically detects the tag!*

### What Our Service Does:
1. **Nose Video**: Detects dog snout with YOLO $\rightarrow$ crops $\rightarrow$ applies CLAHE $\rightarrow$ extracts 40–50 sharp biometric vector frames.
2. **Body Video**: Bypasses the nose detector completely $\rightarrow$ directly splits video into 40–50 sharp body frames $\rightarrow$ extracts body vectors.
3. **Zero Waste**: **No videos or image files are saved to disk.** Only compact vector matrices (`nose_vectors.npz`, `body_vectors.npz`) and your metadata are stored.

### Sample Node.js / Express Code
```javascript
const FormData = require('form-data');
const fs = require('fs');
const fetch = require('node-fetch');

async function registerDogBiometrics(caseId, dogName, noseVideoPath, bodyVideoPath, extraFormData) {
  const form = new FormData();
  form.append('case_id', caseId);
  form.append('dog_name', dogName || 'Unknown');
  form.append('metadata', JSON.stringify(extraFormData || {}));

  if (noseVideoPath) {
    form.append('nose_video', fs.createReadStream(noseVideoPath));
  }
  if (bodyVideoPath) {
    form.append('body_video', fs.createReadStream(bodyVideoPath));
  }

  const response = await fetch('http://localhost:8000/api/v1/register', {
    method: 'POST',
    body: form,
    headers: form.getHeaders(),
  });

  return await response.json();
}
```

### Sample Response (`200 OK`)
```json
{
  "success": true,
  "message": "Successfully enrolled dog 'Sheru' for case 'case-00124'.",
  "case_id": "case-00124",
  "dog_name": "Sheru",
  "nose_frames_enrolled": 45,
  "body_frames_enrolled": 45,
  "metadata": {
    "case_id": "case-00124",
    "dog_name": "Sheru",
    "registered_at": "2026-11-20 13:45:00",
    "nose_frames_enrolled": 45,
    "body_frames_enrolled": 45,
    "has_nose_stream": true,
    "has_body_stream": true
  }
}
```

---

## 3. CALL 2: Identification (`POST /api/v1/identify`)

Trigger this call when someone uploads a query photo to check if a stray dog already exists in the shelter database.

### Request Specifications
* **URL**: `http://localhost:8000/api/v1/identify`
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`

### Form Fields (Key-Value)
| Field Key | Type | Required? | Description |
| :--- | :--- | :--- | :--- |
| `image` | `File` (JPG/PNG) | **Yes** | Single query image (also accepts field name `photo` or `file`). |
| `gender` | `string` | No | Optional gender filter (`"Male"` / `"Female"`). |

### Decision Rules Applied Automatically:
* **If both Nose & Body vectors are in the gallery**: `Score = 80% Nose + 20% Body` (Acceptance threshold: $\ge 0.78$).
* **If only Nose vectors exist**: `Score = 100% Nose` (Acceptance threshold: $\ge 0.78$).
* **If only Body vectors exist**: `Score = 100% Body` (Strict acceptance threshold: $\ge 0.82$).

### Sample Node.js / Express Code
```javascript
const FormData = require('form-data');
const fs = require('fs');
const fetch = require('node-fetch');

async function identifyDogPhoto(imageFilePath) {
  const form = new FormData();
  form.append('image', fs.createReadStream(imageFilePath));

  const response = await fetch('http://localhost:8000/api/v1/identify', {
    method: 'POST',
    body: form,
    headers: form.getHeaders(),
  });

  return await response.json();
}
```

### Sample Response (`200 OK`)

#### Case A: Match Found (`status: "MATCH"`)
```json
{
  "success": true,
  "status": "MATCH",
  "rule_applied": "NOSE_80_BODY_20",
  "top_match": {
    "case_id": "case-00124",
    "similarity_score": 0.8924,
    "nose_score": 0.9120,
    "body_score": 0.8140,
    "best_matching_frame": "frame_0014"
  },
  "candidates": [
    {
      "case_id": "case-00124",
      "similarity_score": 0.8924,
      "status": "MATCH"
    },
    {
      "case_id": "case-00089",
      "similarity_score": 0.6210,
      "status": "UNKNOWN"
    }
  ]
}
```

#### Case B: Unknown Dog (`status: "UNKNOWN"`)
```json
{
  "success": true,
  "status": "UNKNOWN",
  "message": "No match found above acceptance threshold.",
  "top_match": null,
  "candidates": []
}
```

---

## 4. Summary for the Web Developer

1. Run the Python service via `uvicorn api:app --port 8000`.
2. When creating a case, `POST` the form data + video files to `http://localhost:8000/api/v1/register`.
3. When searching/verifying a dog, `POST` the photo to `http://localhost:8000/api/v1/identify`.
4. If `status === "MATCH"`, display the matched `case_id` on the portal. That's it!
