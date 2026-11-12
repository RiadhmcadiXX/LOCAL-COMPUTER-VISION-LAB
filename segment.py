
import os
import re
import json
import numpy as np
import torch

import io
import uuid

from dotenv import load_dotenv
from supabase import create_client

from PIL import Image, ImageFilter
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)
from segment_anything import (
    sam_model_registry,
    SamPredictor,
)

from channel3_search import search_all_from_metadata

# =====================================================
# CONFIGURATION
# =====================================================

IMAGE_PATH = "outfit.jpg"
OUTPUT_DIR = "results"

DINO_MODEL = "IDEA-Research/grounding-dino-tiny"

SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

CANVAS_SIZE = 512
CANVAS_PADDING = 25

DETECTION_THRESHOLD = 0.20
TEXT_THRESHOLD = 0.20

# Set to True for smoother clothing edges.
SMOOTH_EDGES = True

# Use CPU for maximum Intel Mac compatibility.
DEVICE = "cpu"

# Grounding DINO text prompt
PROMPT = ("jacket. coat. blazer. hoodie. sweatshirt. sweater. " "t-shirt. shirt. top. trousers. pants. jeans. " "shorts. skirt. dress. " "shoe. sneaker. boot. sandal. " "handbag. backpack. belt. hat.")

os.makedirs(OUTPUT_DIR, exist_ok = True)

# =====================================================
# UTILITY: NORMALIZE IMAGE FOR APP CANVAS
# =====================================================

def normalize_for_canvas(
    image,
    canvas_size=512,
    padding=25,
):
    """
    Resize a transparent clothing image while
    preserving its aspect ratio.

    Centers the garment inside a transparent
    square canvas.

    Returns an RGBA Pillow image.
    """

    image = image.convert("RGBA")

    available_size = canvas_size - (padding * 2)

    # Resize while maintaining aspect ratio
    image.thumbnail(
                     (available_size, available_size),
                     Image.Resampling.LANCZOS,
                   )

    # Create transparent canvas
    canvas = Image.new(
                        "RGBA",
                        (canvas_size, canvas_size),
                        (0, 0, 0, 0),
                      )

    # Center the clothing item
    x = (canvas_size - image.width) // 2
    y = (canvas_size - image.height) // 2

    canvas.paste(image, (x, y))

    return canvas


# =====================================================
# UTILITY: CREATE TRANSPARENT CLOTHING IMAGE
# =====================================================

def extract_transparent_item(
    original_image,
    mask,
    smooth_edges=True,
):
    """
    Convert SAM segmentation mask into
    a transparent RGBA clothing image.

    Returns a tightly cropped Pillow image,
    or None if the mask is empty.
    """

    original_array = np.array(
                               original_image.convert("RGB")
                             )

    # Convert mask to alpha channel
    alpha = (mask.astype(np.uint8) * 255)

    if smooth_edges:
        alpha_image = Image.fromarray(alpha)
        alpha_image = alpha_image.filter(
                                          ImageFilter.GaussianBlur(radius = 0.7)
                                        )
        alpha = np.array(alpha_image)

    # Combine RGB with alpha
    rgba = np.dstack(
                      (original_array, alpha)
                    )

    transparent_image = Image.fromarray(
                                         rgba,
                                         mode = "RGBA",
                                       )

    # Find visible pixels
    ys, xs = np.where(alpha > 5)

    if len(xs) == 0 or len(ys) == 0:
        return None

    # Crop around the item
    padding = 10

    left = max(0, int(xs.min()) - padding)
    top = max(0, int(ys.min()) - padding)

    right = min(
                 original_image.width,
                 int(xs.max()) + padding + 1,
               )

    bottom = min(
                  original_image.height,
                  int(ys.max()) + padding + 1,
                )

    return transparent_image.crop(
                                   (left, top, right, bottom)
                                 )


# =====================================================
# STEP 1: LOAD ORIGINAL IMAGE
# =====================================================

print("Loading image...")

original_image = (
                   Image.open(
                               IMAGE_PATH
                             )
                        .convert("RGB")
                 )

image_array = np.array(original_image)

print(
       f"Image size: " f"{original_image.width}x{original_image.height}"
     )



# =====================================================
# SUPABASE STORAGE
# =====================================================

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
SUPABASE_BUCKET = os.getenv(
                             "SUPABASE_BUCKET",
                             "clothing-items",
                           )

supabase = create_client(
                          SUPABASE_URL,
                          SUPABASE_KEY,
                        )

def authenticate_supabase():
    """
    Authenticate as a Supabase user so that
    Storage RLS policies apply correctly.
    """
    email = os.environ["SUPABASE_EMAIL"]
    password = os.environ["SUPABASE_PASSWORD"]

    response = supabase.auth.sign_in_with_password(
                                                    {
                                                      "email": email,
                                                      "password": password,
                                                    }
                                                  )

    if not response.user:
        raise RuntimeError(
                            "Supabase authentication failed"
                          )

    print("Supabase authenticated")
    return response.user.id


def upload_clothing_image(
    image,
    filename,
    user_id,
):
    """
    Upload a Pillow RGBA image to a private
    Supabase Storage bucket.

    Returns the storage path and signed URL.
    """

    # Encode PNG in memory.
    buffer = io.BytesIO()
    image.save(
                buffer,
                format = "PNG",
                optimize = True,
              )

    image_bytes = buffer.getvalue()

    # Avoid filename collisions.
    unique_id = uuid.uuid4().hex[: 12]

    storage_path = (f"{user_id}/{unique_id}_{filename}")

    print(f"Uploading: {storage_path}")

    (
      supabase.storage.from_(
                              SUPABASE_BUCKET
                            )
              .upload(
                       path = storage_path,
                       file = image_bytes,
                       file_options = {
                                        "content-type": "image/png",
                                        "upsert": "false",
                                      },
                     )
    )

    # Temporary URL valid for 1 hour.
    signed_response = (
                        supabase.storage.from_(SUPABASE_BUCKET)
                                .create_signed_url(
                                                    storage_path,
                                                    3600,
                                                  )
                      )

    signed_url = signed_response.get("signedURL")
    if not signed_url:
        raise RuntimeError(
                            "Could not generate signed URL"
                          )

    return {
             "storage_path": storage_path,
             "signed_url": signed_url,
           }


user_id = authenticate_supabase()


# =====================================================
# STEP 2: LOAD GROUNDING DINO
# =====================================================

print("Loading Grounding DINO...")

processor = AutoProcessor.from_pretrained(
                                           DINO_MODEL
                                         )

dino = (
         AutoModelForZeroShotObjectDetection.from_pretrained(
                                                              DINO_MODEL
                                                            )
                                            .to(DEVICE)
       )

dino.eval()

# =====================================================
# STEP 3: DETECT CLOTHING
# =====================================================

print("Detecting clothing items...")

inputs = processor(
                    images = original_image,
                    text = PROMPT,
                    return_tensors = "pt",
                  ).to(DEVICE)

with torch.inference_mode():
    outputs = dino(**inputs)

results = processor.post_process_grounded_object_detection(
                                                            outputs,
                                                            inputs.input_ids,
                                                            threshold = DETECTION_THRESHOLD,
                                                            text_threshold = TEXT_THRESHOLD,
                                                            target_sizes = [
                                                                             (original_image.height, original_image.width)
                                                                           ],
                                                          )[0]

boxes = (
          results["boxes"].cpu()
                          .numpy()
        )
scores = (
           results["scores"].cpu()
                            .numpy()
         )

# Transformers 4.49.0
labels = results["labels"]



from PIL import ImageDraw

preview = original_image.copy()
draw = ImageDraw.Draw(preview)

for box, label, score in zip(
    boxes, labels, scores
):
    x1, y1, x2, y2 = map(int, box)

    draw.rectangle(
                    (x1, y1, x2, y2),
                    outline = "red",
                    width = 3,
                  )

    draw.text(
               (x1, max(0, y1 - 15)),
               f"{label}: {float(score):.2f}",
               fill = "red",
             )

preview.save("results/detection_preview.jpg")
print("Detection preview saved")


print(f"Found {len(boxes)} clothing items.")

# Free memory before loading SAM
del dino, inputs, outputs

# =====================================================
# STEP 4: LOAD SAM 1
# =====================================================

if len(boxes) == 0:
    print("No clothing detected.")
    raise SystemExit(0)

print("Loading SAM 1...")

sam = sam_model_registry[SAM_MODEL_TYPE](
                                          checkpoint = SAM_CHECKPOINT
                                        )

sam.to(device = DEVICE)
sam.eval()

predictor = SamPredictor(sam)

# Compute image embedding once
predictor.set_image(image_array)


# =====================================================
# STEP 5: SEGMENT AND EXPORT EVERY ITEM
# =====================================================

metadata = []

print(f"\nTotal detected boxes: {len(boxes)}")

for index, (box, label, score) in enumerate(
    zip(boxes, labels, scores),
    start=1,
):
    label = str(label)

    print(f"\n========== ITEM {index} ==========")
    print("Label:", label)
    print("Confidence:", round(float(score), 3))
    print("Box:", box.tolist())

    try:
        # 1. Segment the CURRENT bounding box
        with torch.inference_mode():
            masks, mask_scores, _ = predictor.predict(
                                                       box = box.astype(np.float32),
                                                       multimask_output = True,
                                                     )

        # 2. Select the best SAM mask
        best_index = int(np.argmax(mask_scores))
        mask = masks[best_index]

        print("SAM score:", float(mask_scores[best_index]))
        print("Mask pixels:", int(mask.sum()))

        if not mask.any():
            print("WARNING: Empty mask, skipping.")
            continue

        # 3. Save mask for debugging
        mask_image = Image.fromarray(
                                      (mask.astype(np.uint8) * 255)
                                    )

        mask_image.save(
                         os.path.join(
                                       OUTPUT_DIR,
                                       f"debug_mask_{index}.png",
                                     )
                       )

        # 4. Extract transparent garment
        extracted = extract_transparent_item(
                                              original_image,
                                              mask,
                                              smooth_edges = SMOOTH_EDGES,
                                            )

        if extracted is None:
            print("WARNING: Extraction returned None.")
            continue

        # 5. Normalize for React Native canvas
        canvas_image = normalize_for_canvas(
                                             extracted,
                                             canvas_size = CANVAS_SIZE,
                                             padding = CANVAS_PADDING,
                                           )

        # 6. Create a unique filename
        safe_label = (
                       (
                         re.sub(
                                 r"[^a-z0-9_-]+",
                                 "_",
                                 label.lower(),
                               )
                           .strip("_")
                       )
                       or ("item")
                     )

        filename = f"{safe_label}_{index}.png"

        output_path = os.path.join(
                                    OUTPUT_DIR,
                                    filename,
                                  )

        # 7. Save locally FIRST
        canvas_image.save(
                           output_path,
                           format = "PNG",
                           optimize = True,
                         )

        print("LOCAL SAVE SUCCESS:", output_path)

        # 8. Upload to Supabase
        upload_result = None

        try:
            upload_result = upload_clothing_image(
                                                   image = canvas_image,
                                                   filename = filename,
                                                   user_id = user_id,
                                                 )

            print(
                   "SUPABASE SUCCESS:",
                   upload_result["storage_path"],
                 )

        except Exception as upload_error:
            print(
                   "SUPABASE FAILED:",
                   str(upload_error),
                 )

        # 9. Store metadata
        metadata.append(
                         {
                           "id": index,
                           "label": label,
                           "confidence": round(float(score), 3),
                           "sam_score": round(
                                               float(mask_scores[best_index]),
                                               3
                                             ),
                           "filename": filename,
                           "storage_path": (upload_result["storage_path"] if upload_result else None),
                           "signed_url": (upload_result["signed_url"] if upload_result else None),
                         }
                       )

    except Exception as error:
        print(
               f"FAILED TO PROCESS ITEM {index}: {error}"
             )
        import traceback
        traceback.print_exc()

print("\n========== SUMMARY ==========")
print("Detected boxes:", len(boxes))
print("Successfully exported:", len(metadata))



# =====================================================
# STEP 6: EXPORT METADATA
# =====================================================

metadata_path = os.path.join(
                              OUTPUT_DIR,
                              "metadata.json",
                            )

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
               {
                 "source_image": IMAGE_PATH,
                 "image_width": original_image.width,
                 "image_height": original_image.height,
                 "items": metadata,
               },
               file,
               indent = 2,
             )


    print("\nStarting Channel3 product search...")

search_all_from_metadata(
                          metadata_path = metadata_path
                        )

print("\nExtraction completed!")
print(f"Exported {len(metadata)} items.")
print(f"Results directory: {OUTPUT_DIR}")
