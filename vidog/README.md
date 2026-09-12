# vidog - DogID V2 (Video + Hybrid Classifier)

Two-stage hybrid biometric matcher.
Upload a video clip -> VideoRegistrationEngine samples sharp, diverse
nose frames -> embeds -> trains a per-gallery logistic-regression head ->
identification combines an Open-Set Cosine Gate with a Fast Linear Boundary
Classifier.

## Run

    cd vidog
    pip install -r ../requirements.txt
    streamlit run app.py

## Layout

    vidog/
    |- app.py                  Streamlit UI (Video Registration / Identify / Gallery)
    |- det5.py                 DNNetV3 teacher (shared with phodog)
    |- nose_detector.py        YOLOv8 ONNX detector (shared with phodog)
    |- video_engine.py         Frame sampling + Laplacian sharpness + greedy diversity
    |- matcher.py              2-Stage Hybrid (cosine gate + linear head fusion)
    |- classifier_head.py      Few-shot sklearn LogisticRegression head
    |- checkpoints/nose_detector.onnx
    |- data/gallery_v2/        folder gallery + classifier.pkl + class_map.json
    |- dataset/
    |  |- dog_samples_original/   69 reference photos
    |  |- nose_samples_augmented/ 123 synthetic probes
    |  \- video_samples/          7 sample registration videos
    |- docs/                   training guides
    \- scripts/
       |- batch_reenroll_student.py     generates 512-d student embeddings
       \- eval_augmented_benchmark.py   Rank-1/3 + open-set FAR

## Optional: benchmark

    cd vidog
    python scripts/eval_augmented_benchmark.py --holdout 5 --threshold 0.64

## Optional: distill student embeddings

    cd vidog
    python scripts/batch_reenroll_student.py --dry-run
