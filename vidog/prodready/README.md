# DogID v2 — Production Package

Production-ready package for integrating DogID biometric dog identification into a website backend.

## Quick Start

```python
from vidog.prodready.api import DogIDApi

# Initialize
api = DogIDApi(gallery_dir="data/gallery_v2")

# Register a dog from video
result = api.register_dog(
    video_path="video.mp4",
    dog_name="Rex",
    gender="Male",
    breed="Labrador",
    age=3.0,
    color="Golden"
)

# Identify a dog from a photo
result = api.identify_dog("query.jpg", gender_filter="Male")
```

## API Reference

### DogIDApi

| Method | Description |
|--------|-------------|
| `register_dog(video_path, dog_name, gender, breed, age, color)` | Register a dog from a video |
| `identify_dog(image_path, gender_filter)` | Identify a dog from a photo |
| `get_gallery_info()` | List all registered dogs |

## Gender Filtering

The API supports gender-filtered matching: male dogs are only compared against males, females against females. Pass `gender_filter` to `identify_dog`.

## Requirements

See `requirements.txt`.