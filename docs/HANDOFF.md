# Vidog Biometric Engine: Developer Handoff Guide

This guide contains everything you need to run the biometric engine on the server and connect it to the website backend.

You do **not** need to handle OpenCV, YOLO detection, frame sampling, cropping, or vector math. You only need to:
1. Run the service on port `8000`.
2. Make **two simple HTTP POST calls** from your backend (one for Registration, one for Identification).

---

## 1. Server Setup & Run Command

On the server where your app runs:

### A. Navigate into the folder & install dependencies
```bash
cd vidog_pure_cosine_similarity
pip install fastapi uvicorn torch torchvision opencv-python pillow numpy timm onnxruntime
```

### B. Start the service
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
*(To run persistently in the background in production, you can use `pm2 start "uvicorn api:app --host 0.0.0.0 --port 8000" --name vidog-service` or a systemd service).*

### C. Verify it is running
Open another terminal or run in browser:
```bash
curl http://localhost:8000/health
# Response: {"status":"ok"}
```

---

## 2. CALL 1: Registration (`POST /api/v1/register`)

Trigger this call when:
- A new case is submitted with video(s).
- An existing case is edited with new video(s).

### What to Send
- **URL**: `http://localhost:8000/api/v1/register`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `case_id`: (Required string, e.g. `"case-00124"`)
  - `nose_video`: (Video file for snout, or file named `nose_*.mp4`)
  - `body_video`: (Video file for full body, or file named `body_*.mp4`)

> **Note**: You can send both videos, or just one if only one was recorded. If both are sent, our model uses 80% Nose + 20% Body weighting. If only one is sent, it uses 100% of whatever was sent. No videos are stored on disk; only lightweight vector files are saved.

### cURL Command Test
```bash
curl -X POST http://localhost:8000/api/v1/register \
  -F "case_id=case-00124" \
  -F "nose_video=@/path/to/nose_video.mp4" \
  -F "body_video=@/path/to/body_video.mp4"
```

### Expected Response
```json
{
  "status": "success",
  "message": "Vectors generated successfully. Dog for case ID 'case-00124' registered/updated successfully.",
  "case_id": "case-00124",
  "metadata": {
    "case_id": "case-00124",
    "created_at": "2026-11-24 10:45:00",
    "updated_at": "2026-11-24 10:45:00",
    "nose_frames_enrolled": 45,
    "body_frames_enrolled": 45,
    "has_nose_stream": true,
    "has_body_stream": true,
    "nose_reupload_recommended": false,
    "body_reupload_recommended": false
  }
}
```

> **Blurry Video Fallback (10 Frames = Re-upload Prompt)**:
> If a video is very blurry, our engine falls back to storing the **10 sharpest frames** so the case still gets enrolled (`nose_frames_enrolled === 10` or `body_frames_enrolled === 10`, and `*_reupload_recommended === true`).
> On the frontend, if you see `10` frames returned for either stream, prompt the user to re-upload a clearer video. When they re-upload and call `POST /api/v1/register` with the same `case_id`, our API **automatically overrides** the old vectors for that stream with fresh frames.
>
> **Database Storage Tip**: You can directly take the returned `"metadata"` JSON object and save it into your PostgreSQL / Prisma `biometric_metadata` (or JSONB) column for this case.

---

## 3. CALL 2: Identification (`POST /api/v1/identify`)

Trigger this call when someone uploads a single photo of a dog to check if it already exists in the shelter records.

### What to Send
- **URL**: `http://localhost:8000/api/v1/identify`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `image`: (Single image file, JPG / PNG)

### cURL Command Test
```bash
curl -X POST http://localhost:8000/api/v1/identify \
  -F "image=@/path/to/query_dog.jpg"
```

### Expected Response

#### Case A: Match Found (`status: "MATCH"`)
```json
{
  "status": "MATCH",
  "top_match": {
    "case_id": "case-00124",
    "similarity_score": 0.8924,
    "rule_applied": "NOSE_80_BODY_20",
    "metadata": {
      "case_id": "case-00124",
      "created_at": "2026-11-24 10:45:00",
      "updated_at": "2026-11-24 10:45:00",
      "nose_frames_enrolled": 45,
      "body_frames_enrolled": 45,
      "has_nose_stream": true,
      "has_body_stream": true
    }
  },
  "candidates": [
    {
      "case_id": "case-00124",
      "similarity_score": 0.8924,
      "status": "MATCH",
      "metadata": { ... }
    }
  ]
}
```

#### Case B: Unknown Dog (`status: "UNKNOWN"`)
```json
{
  "status": "UNKNOWN",
  "message": "No matching dog found.",
  "top_match": null,
  "candidates": []
}
```

---

## 4. Ready-to-Use Express / Node.js Backend Code

Copy and paste this helper into your Node.js backend (e.g., `server/services/biometrics.ts`):

```typescript
import FormData from 'form-data';
import fs from 'fs';
import fetch from 'node-fetch'; // or global fetch in Node.js 18+

const BIOMETRIC_URL = 'http://localhost:8000/api/v1';

/**
 * Call 1: Register Dog Biometrics (New Case or Edit Case)
 */
export async function registerDogBiometrics(
  caseId: string, 
  noseVideoPath?: string, 
  bodyVideoPath?: string
) {
  const form = new FormData();
  form.append('case_id', caseId);

  if (noseVideoPath && fs.existsSync(noseVideoPath)) {
    form.append('nose_video', fs.createReadStream(noseVideoPath));
  }
  if (bodyVideoPath && fs.existsSync(bodyVideoPath)) {
    form.append('body_video', fs.createReadStream(bodyVideoPath));
  }

  const response = await fetch(`${BIOMETRIC_URL}/register`, {
    method: 'POST',
    body: form as any,
    headers: form.getHeaders(),
  });

  return await response.json();
}

/**
 * Call 2: Identify Dog from Photo
 */
export async function identifyDogPhoto(imagePath: string) {
  const form = new FormData();
  form.append('image', fs.createReadStream(imagePath));

  const response = await fetch(`${BIOMETRIC_URL}/identify`, {
    method: 'POST',
    body: form as any,
    headers: form.getHeaders(),
  });

  return await response.json();
}
```

### How to use it in Express Routes

```typescript
// Example: In your Case Creation / Edit Route
app.post('/api/cases', upload.fields([{ name: 'nose_video' }, { name: 'body_video' }]), async (req, res) => {
  // 1. Save case in database as usual
  const newCase = await prisma.case.create({ ... });

  // 2. Trigger biometrics in background
  const noseVid = req.files['nose_video']?.[0]?.path;
  const bodyVid = req.files['body_video']?.[0]?.path;

  if (noseVid || bodyVid) {
    registerDogBiometrics(newCase.id, noseVid, bodyVid)
      .then(res => console.log(`Biometrics enrolled for ${newCase.id}:`, res))
      .catch(err => console.error("Biometrics failed:", err));
  }

  return res.json(newCase);
});

// Example: In your Identification Route
app.post('/api/cases/identify', upload.single('image'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "Image required" });

  const result = await identifyDogPhoto(req.file.path);

  if (result.status === 'MATCH' && result.top_match?.case_id) {
    const matchedCase = await prisma.case.findUnique({
      where: { id: result.top_match.case_id }
    });
    return res.json({ matched: true, case: matchedCase, score: result.top_match.similarity_score });
  }

  return res.json({ matched: false, message: "No matching dog found" });
});
```
