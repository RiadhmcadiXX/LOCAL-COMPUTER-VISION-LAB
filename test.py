
import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# Choose GPU when available, otherwise CPU.
device = (
    "cpu"
)

print("Using device:", device)

# Download model automatically on first run
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL_ID
).to(device)

model.eval()

# Load local image
image = Image.open("outfit.jpg").convert("RGB")

# Clothing categories to detect
labels = [[
    "jacket",
    "shirt",
    "t-shirt",
    "hoodie",
    "pants",
    "jeans",
    "shoes",
    "sneakers",
    "dress",
    "skirt",
]]

inputs = processor(
    images=image,
    text=labels,
    return_tensors="pt",
).to(device)

with torch.inference_mode():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.3,
    text_threshold=0.25,
    target_sizes=[(image.height, image.width)],
)[0]

draw = ImageDraw.Draw(image)

for box, score, label in zip(
    results["boxes"],
    results["scores"],
    results["text_labels"],
):
    coordinates = [int(x) for x in box.tolist()]

    draw.rectangle(
        coordinates,
        outline="red",
        width=4,
    )

    draw.text(
        (coordinates[0], coordinates[1]),
        f"{label}: {score:.2f}",
        fill="red",
    )

    print(f"Detected: {label} ({score:.2f})")

image.save("result.jpg")
print("Saved result.jpg")
