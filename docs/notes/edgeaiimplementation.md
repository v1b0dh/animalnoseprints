# Integrate Edge AI (StudentDNNet) into Dog Biometrics App

The `StudentDNNet` model (MobileNetV3-Small, ~2.5M params, ~8ms CPU inference) is fully defined in `det5.py` but never used anywhere in the application. This plan wires it into the live Streamlit app so users can toggle between the heavy teacher model and the lightweight student model, and export the student for on-device deployment.

## User Review Required

> [!IMPORTANT]
> **Embedding dimension mismatch:** The teacher produces **576-d** (no head) or **1024-d** (with head) embeddings. The student produces **512-d** embeddings. Existing database entries were enrolled with the teacher's dimension. The student cannot match against teacher-enrolled embeddings directly — they live in different vector spaces.
>
> **Proposed solution:** When running in student mode, re-extract embeddings at query time using the student model and compare only against a student-specific gallery. The DB `embeddings` table gets a new `model_type` column (`"teacher"` or `"student"`) so both can coexist. Existing rows default to `"teacher"`.

> [!WARNING]
> **No trained student weights exist yet.** The distillation training loop is defined but was never run. The student model will load with ImageNet-pretrained MobileNetV3 weights only — it has **not** been distilled from the teacher. Embeddings will be functional but accuracy will be lower than the teacher until distillation training is performed. The app will display a clear warning about this.

## Open Questions

1. **Should the student model be the default?** Currently I plan to keep the teacher as default and let users opt into student mode via a sidebar toggle. Should it be the other way around?
2. **ONNX export** — Should I include an "Export to ONNX" button in the UI so users can download the student model for mobile deployment? (The `updatemulti.md` mentions this as a Level 4 scaling goal.)

## Proposed Changes

### Database Layer — Embedding model type tracking

#### [MODIFY] [app.py](file:///c:/Users/papu_/Downloads/dogshi/app.py)

- Add `model_type TEXT DEFAULT 'teacher'` column to the `embeddings` table (with safe migration for existing rows).
- Update `add_embedding()` to accept and store an optional `model_type` parameter.
- Update `get_dog_gallery()` to filter by `model_type` so teacher and student embeddings are queried separately.

---

### Model Loading — Dual-mode support

#### [MODIFY] [app.py](file:///c:/Users/papu_/Downloads/dogshi/app.py)

- Update the import line to also import `StudentDNNet` from `det5.py`.
- Add a new `load_student_model()` function (cached with `@st.cache_resource`) that loads `StudentDNNet` with optional checkpoint support (`checkpoints/student_best.pth`).
- Add a **sidebar toggle**: "⚡ Edge AI Mode" that switches between teacher and student.
- Update `EMBED_DIM` to be dynamic based on which model is active (576/1024 for teacher, 512 for student).
- When student mode is active and no student embeddings exist, show an info banner prompting the user to enroll dogs with the student model.

---

### Inference & Enrollment — Student-aware pipeline

#### [MODIFY] [app.py](file:///c:/Users/papu_/Downloads/dogshi/app.py)

- Update `get_embedding_from_image()` — no changes needed (it's model-agnostic already).
- Update the **Identify** page to use whichever model is active and query the correct gallery subset.
- Update the **Register** page to tag new embeddings with the active `model_type`.
- Add a **student-mode warning banner** at the top when Edge AI mode is on (explaining reduced accuracy without distillation).

---

### ONNX Export Utility

#### [NEW] [export_student_onnx.py](file:///c:/Users/papu_/Downloads/dogshi/export_student_onnx.py)

- Script that loads `StudentDNNet`, traces it with a dummy input, and exports to `student_dnnet.onnx`.
- Includes input/output naming and dynamic batch axis for mobile runtimes.
- Can be triggered from the command line: `python export_student_onnx.py`.

---

### Edge AI Dashboard — New sidebar page

#### [MODIFY] [app.py](file:///c:/Users/papu_/Downloads/dogshi/app.py)

- Add a 5th navigation entry: **"⚡ Edge AI"**.
- Page content:
  - **Model Comparison Card**: Side-by-side metrics table (Teacher vs. Student — params, embed dim, inference time, backbone).
  - **Current Mode Indicator**: Shows which model is active with a visual badge.
  - **Export Section**: Button to export student model to ONNX with download link.
  - **Student Gallery Stats**: How many embeddings exist for the student model, coverage vs. teacher.
  - **Batch Re-enrollment**: Button to re-process all existing dog photos through the student model (so both galleries are populated).

---

### Smoke Test Update

#### [MODIFY] [test_smoke.py](file:///c:/Users/papu_/Downloads/dogshi/test_smoke.py)

- Add test for the new `model_type` column in the `embeddings` table.

## Verification Plan

### Automated Tests
- `python test_smoke.py` — confirms the `model_type` column exists and defaults to `"teacher"`.

### Manual Verification
- Run `streamlit run app.py` and:
  1. Confirm existing gallery still loads correctly (teacher mode, backward compatible).
  2. Toggle "⚡ Edge AI Mode" in the sidebar — verify student model loads.
  3. Register a dog in student mode — verify embedding is tagged `model_type='student'`.
  4. Identify a dog in student mode — verify it only searches student embeddings.
  5. Visit the "⚡ Edge AI" page — verify model comparison card and ONNX export button.
  6. Run `python export_student_onnx.py` — verify `.onnx` file is created.
