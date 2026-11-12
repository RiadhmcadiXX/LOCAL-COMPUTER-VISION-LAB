
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

# Load .env from project folder
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
BUCKET = os.environ["SUPABASE_BUCKET"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
)

# Authenticate
print("Authenticating...")

auth = supabase.auth.sign_in_with_password({
    "email": os.environ["SUPABASE_EMAIL"],
    "password": os.environ["SUPABASE_PASSWORD"],
})

if not auth.user:
    raise RuntimeError("Authentication failed")

print("Authentication successful")
print("User ID:", auth.user.id)

# Upload a generated PNG
image_path = BASE_DIR / "results" / "jacket_1.png"

if not image_path.exists():
    raise FileNotFoundError(
        f"Image not found: {image_path}"
    )

storage_path = (
    f"{auth.user.id}/test_{uuid.uuid4().hex[:8]}.png"
)

print("Uploading to:", BUCKET)

with image_path.open("rb") as file:
    result = supabase.storage.from_(BUCKET).upload(
        path=storage_path,
        file=file,
        file_options={
            "content-type": "image/png",
            "upsert": "false",
        },
    )

print("Upload successful:", result)

# Generate URL
signed = supabase.storage.from_(
    BUCKET
).create_signed_url(
    storage_path,
    3600,
)

print("Signed URL created:", bool(signed.get("signedURL")))
print("Storage path:", storage_path)
