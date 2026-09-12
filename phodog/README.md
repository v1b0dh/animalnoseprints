# phodog - DogID V1 (Photo-based)

Single-stage cosine-similarity dog nose identification.
Upload one or more nose photos -> crop -> embed (DNNetV3 / TinyViT-21M) ->
match against data/gallery/ using ensemble max-cosine (85% nose + 15% face).

## Run

    cd phodog
    pip install -r ../requirements.txt
    streamlit run app.py

## Layout

    phodog/
    |- app.py                  Streamlit UI (Identify / Register / Manage / Database)
    |- det5.py                 DNNetV3 teacher + MagFace + CLAHEPipeline + StudentDNNet
    |- nose_detector.py        YOLOv8 ONNX detector (cv2.dnn) + center-crop fallback
    |- checkpoints/nose_detector.onnx
    |- data/gallery/           folder gallery (one subdir per dog)
    |- dataset/
    |  |- dog_samples_original/   69 reference photos
    |  |- nose_samples/           41 cropped reference noses
    |  \- nose_samples_augmented/ 123 synthetic probes
    |- docs/                   training guides
    \- scripts/eval_gallery.py LOO Rank-1/5, mAP, EER, genuine/impostor stats

## Optional: benchmark

    cd phodog
    python scripts/eval_gallery.py
