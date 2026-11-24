# Complete System Workflow: PFA Web Portal × Vidog Biometrics

This document maps out the entire architectural and procedural lifecycle of the system. It covers what the website calls, how our service responds, which files are created, and the precise control flow for both **Registration** and **Identification**.

---

## 1. High-Level Architecture & Component Roles

```mermaid
flowchart TD
    subgraph Client["Client Tier"]
        UI["React 19 Frontend<br>(NewCasePage / CasesPage)"]
    end

    subgraph PortalBackend["Node.js / Express Backend"]
        Express["Express API Server<br>(Port 5000 / 3000)"]
        Prisma["Prisma ORM"]
        Postgres[("PostgreSQL Database<br>(Cases, Animals, Metadata)")]
    end

    subgraph BiometricService["Python Biometric Microservice (FastAPI)"]
        FastAPI["FastAPI Engine<br>(Port 8000)"]
        YOLO["YOLOv8 Snout Detector<br>(best.onnx)"]
        TinyViT["DNNetV3 Backbone (1024-d)<br>(teacher_best.pth)"]
        Matcher["Pure Cosine Matcher<br>(Vector Dot Products)"]
        Gallery[("Gallery Storage<br>(data/gallery_v2/{case_id}/)")]
    end

    UI -->|HTTP Multipart Form| Express
    Express -->|Prisma Queries| Prisma
    Prisma --> Postgres
    Express -->|Internal REST Calls| FastAPI
    FastAPI --> YOLO
    FastAPI --> TinyViT
    FastAPI --> Matcher
    FastAPI --> Gallery
```

| Component | Responsibility |
| :--- | :--- |
| **React Frontend** | Form capture (case details, videos, query photo). Displays biometric feedback. |
| **Express Backend** | Business logic, authentication, saving initial case record in PostgreSQL, and forwarding media streams to Python. |
| **Python FastAPI (`api.py`)** | Video frame extraction, snout detection, dual-stream feature embedding (1024-d), vector file creation, and cosine similarity ranking. |
| **PostgreSQL** | Stores shelter records, case IDs, and the returned biometric `metadata` JSON object. |
| **Disk Gallery (`gallery_v2/`)** | Stores only `.npz` vector matrices and `meta.json`. **No raw videos or images are permanently stored on disk.** |

---

## 2. WORKFLOW 1: Registration (Initial Intake or Edit Case)

Triggered when a volunteer/doctor submits a new case report or edits an existing case and uploads video(s).

### Sequence Diagram: Registration Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as Volunteer / Doctor
    participant React as React Frontend
    participant Express as Express Backend
    participant Postgres as PostgreSQL (Prisma)
    participant FastAPI as Python FastAPI (Port 8000)
    participant VideoEngine as Video Engine & YOLO
    participant Disk as Gallery Storage

    User->>React: Selects Case Info + Videos (nose_video, body_video)
    User->>React: Clicks "Register / Submit Case"
    React->>Express: POST /api/cases (Multipart Form)
    
    Express->>Postgres: Creates/Updates Case Record (Generates case_id)
    Postgres-->>Express: Confirms Case Record Created
    
    Express->>FastAPI: POST /api/v1/register<br>(case_id, nose_video stream, body_video stream)
    
    Note over FastAPI: Streams video into temporary RAM/tmpfs buffer
    
    rect rgb(240, 248, 255)
        Note over FastAPI,VideoEngine: Stream 1: Nose Processing (if provided)
        FastAPI->>VideoEngine: Run OpenCV & YOLO snout detection
        VideoEngine-->>FastAPI: 40-50 sharp snout crops
        FastAPI->>FastAPI: Extract 1024-d embeddings (TinyViT)
        FastAPI->>Disk: Atomic write to nose_vectors.npz (via .tmp swap)
    end
    
    rect rgb(255, 250, 240)
        Note over FastAPI,VideoEngine: Stream 2: Body Processing (if provided)
        FastAPI->>VideoEngine: Direct temporal frame sampling (No YOLO)
        VideoEngine-->>FastAPI: 40-50 sharp full-body frames
        FastAPI->>FastAPI: Extract 1024-d embeddings (TinyViT)
        FastAPI->>Disk: Atomic write to body_vectors.npz (via .tmp swap)
    end
    
    Note over FastAPI: Deletes temporary video files immediately (os.remove)
    
    FastAPI->>Disk: Reads existing counts (if edit) & saves updated meta.json
    FastAPI-->>Express: HTTP 200 OK with full metadata JSON payload
    
    Express->>Postgres: Updates Case/Animal with returned biometric metadata JSON
    Express-->>React: Success response
    React-->>User: Displays "Biometrics Registered Successfully (45 frames)"
```

### 1. What the Website Calls:
* **URL**: `POST http://localhost:8000/api/v1/register`
* **Content-Type**: `multipart/form-data`
* **Payload**:
  * `case_id`: `"c4b3a1d2-0000-4000-8000-0123456789ab"`
  * `nose_video`: `Binary Stream (MP4/MOV)` *(Optional if body provided)*
  * `body_video`: `Binary Stream (MP4/MOV)` *(Optional if nose provided)*

### 2. What Happens Inside Python:
1. Videos are read into a temporary buffer (`tempfile.NamedTemporaryFile`).
2. **Nose Stream**:
   * YOLO detects the snout on candidate frames.
   * Filters out blurry frames using Laplacian variance ($> 65.0$).
   * Selects 40–50 diverse frames spaced across the video.
   * Converts snout crops into 1024-d vectors (`det5.py`).
   * Saves atomically to `nose_vectors.npz`.
3. **Body Stream**:
   * Bypasses YOLO completely.
   * Extracts 40–50 diverse full-body frames.
   * Converts full-body frames into 1024-d vectors.
   * Saves atomically to `body_vectors.npz`.
4. **Immediate Cleanup**:
   * `os.remove(tmp_video)` runs inside a `finally:` block. **No video remains on disk.**
5. **Metadata Update**:
   * `meta.json` is updated under thread-lock (`FILE_WRITE_LOCK`) and swapped atomically using `os.replace`.

### 3. What Files Are Created/Updated on Disk:
```
data/gallery_v2/{case_id}/
├── meta.json             <-- Human-readable registration details & frame counts
├── nose_vectors.npz      <-- (45, 1024) float32 matrix of nose embeddings
└── body_vectors.npz      <-- (45, 1024) float32 matrix of body embeddings
```

### 4. What Python Returns to Express:
```json
{
  "status": "success",
  "message": "Vectors generated successfully. Dog for case ID 'c4b3a1d2-...' registered/updated successfully.",
  "case_id": "c4b3a1d2-0000-4000-8000-0123456789ab",
  "metadata": {
    "case_id": "c4b3a1d2-0000-4000-8000-0123456789ab",
    "created_at": "2026-11-24 10:45:00",
    "updated_at": "2026-11-24 10:45:00",
    "nose_frames_enrolled": 45,
    "body_frames_enrolled": 45,
    "has_nose_stream": true,
    "has_body_stream": true
  }
}
```

---

## 3. WORKFLOW 2: Identification (Single-Photo Query)

Triggered when a stray dog is admitted or photographed in the field to check if it already has a case history or is a duplicate report.

### Sequence Diagram: Identification Flow

```mermaid
sequenceDiagram
    autonumber
    actor Volunteer as Rescuer / Volunteer
    participant React as React Frontend
    participant Express as Express Backend
    participant Postgres as PostgreSQL (Prisma)
    participant FastAPI as Python FastAPI (Port 8000)
    participant Matcher as Cosine Matcher
    participant Disk as Gallery Storage

    Volunteer->>React: Uploads single query photo of dog
    Volunteer->>React: Clicks "Scan / Identify Dog"
    React->>Express: POST /api/cases/identify (Image file)
    Express->>FastAPI: POST /api/v1/identify (Image file)
    
    FastAPI->>FastAPI: Runs YOLO snout detector on photo
    alt Snout Detected (Confidence >= 0.40)
        FastAPI->>FastAPI: Crops snout -> Computes 1024-d query_nose_vector
    else Snout Not Found (Far/Angle)
        FastAPI->>FastAPI: query_nose_vector = None
    end
    
    FastAPI->>FastAPI: Computes 1024-d query_body_vector from full photo
    
    FastAPI->>Matcher: match(query_nose, query_body)
    
    loop For Every Registered Dog in Gallery
        Matcher->>Disk: Loads nose_vectors.npz & body_vectors.npz
        Matcher->>Matcher: Compute cosine similarity matrix dot products (np.dot)
        Matcher->>Matcher: Apply Multi-Modal Decision Rules
    end
    
    Matcher-->>FastAPI: Top candidate ranking + similarity scores
    FastAPI->>Disk: Loads meta.json for top matching case
    FastAPI-->>Express: JSON with match status, scores, and metadata
    
    alt Status == "MATCH" (Score >= 0.78)
        Express->>Postgres: Fetch full case details, clinical entries & doctor notes
        Postgres-->>Express: Case & medical history
        Express-->>React: { matched: true, caseData, score: 0.8924 }
        React-->>Volunteer: Displays "Existing Patient Found: Sheru (Case #1042)"
    else Status == "UNKNOWN"
        Express-->>React: { matched: false, message: "No match found" }
        React-->>Volunteer: Displays "New Dog (No prior medical records)"
    end
```

### 1. What the Website Calls:
* **URL**: `POST http://localhost:8000/api/v1/identify`
* **Content-Type**: `multipart/form-data`
* **Payload**:
  * `image`: `Binary Stream (JPG/PNG)`

### 2. What Happens Inside Python:
1. Python reads the photo into RAM.
2. **Dual-Extraction from Single Photo**:
   * Runs YOLO on the photo. If snout found $\rightarrow$ extracts `query_nose_vector` (1024-d).
   * Extracts `query_body_vector` (1024-d) from the full image view.
3. **Vectorized Gallery Comparison**:
   * For every enrolled case, loads the normalized matrix and runs:
     $$\text{sims} = \text{gallery\_matrix} \cdot \mathbf{q}$$
   * Finds the single best matching frame score for both nose and body.
4. **Weighting Rules Applied**:
   * **Both present**: $\text{Score} = 0.80 \times \text{Nose} + 0.20 \times \text{Body}$ (Threshold: $\ge 0.78$).
   * **Only Nose present**: $\text{Score} = 1.0 \times \text{Nose}$ (Threshold: $\ge 0.78$).
   * **Only Body present**: $\text{Score} = 1.0 \times \text{Body}$ (Strict Threshold: $\ge 0.82$).

### 3. What Python Returns to Express:

#### If a Match is Found:
```json
{
  "status": "MATCH",
  "top_match": {
    "case_id": "c4b3a1d2-0000-4000-8000-0123456789ab",
    "similarity_score": 0.8924,
    "rule_applied": "NOSE_80_BODY_20",
    "metadata": {
      "case_id": "c4b3a1d2-0000-4000-8000-0123456789ab",
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
      "case_id": "c4b3a1d2-0000-4000-8000-0123456789ab",
      "similarity_score": 0.8924,
      "status": "MATCH",
      "metadata": { ... }
    },
    {
      "case_id": "8f1e2d3c-9999-4000-8000-9876543210fe",
      "similarity_score": 0.5210,
      "status": "UNKNOWN",
      "metadata": { ... }
    }
  ]
}
```

#### If No Match is Found:
```json
{
  "status": "UNKNOWN",
  "message": "No matching dog found.",
  "top_match": null,
  "candidates": []
}
```

---

## 4. Summary of Data & Control Guarantees

1. **Zero Storage Waste**: Videos uploaded during registration live in temporary buffers only while OpenCV samples frames; they are erased via `os.remove()` in `finally:` blocks.
2. **Dimension Hard-Lock**: Both registration and identification are strictly hard-locked to **1024 dimensions** using `teacher_best.pth`.
3. **Atomic Writes & Thread Safety**: All disk file updates are protected with `threading.Lock()` and atomic `os.replace` to prevent race conditions or file corruption when multiple requests occur simultaneously.
4. **Clean Handoff**: The web backend only needs to forward files and store the returned `metadata` object in PostgreSQL.
