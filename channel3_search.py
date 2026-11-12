
import os
import json
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

CHANNEL3_API_KEY = "1GejdSMAoI5emFaGzwupM6SIraqxUcGq9R2XrWpm"
CHANNEL3_SEARCH_URL = (
    "https://api.trychannel3.com/v1/search"
)


def search_clothing(image_url, label, limit=5):
    """
    Search Channel3 using an uploaded garment image.

    image_url: Supabase public or signed URL
    label: Grounding DINO clothing label
    """

    payload = {
        "image_url": image_url,
        "query": label,
        "limit": limit,
    }

    headers = {
        "x-api-key": CHANNEL3_API_KEY,
        "Content-Type": "application/json",
    }

    response = requests.post(
        CHANNEL3_SEARCH_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"Channel3 HTTP {response.status_code}: "
            f"{response.text[:1500]}"
        )

    return response.json()


def search_all_from_metadata(
    metadata_path="results/metadata.json",
):
    """
    Search all successfully uploaded garments
    and save the complete API responses.
    """

    path = Path(metadata_path)
    metadata = json.loads(path.read_text())

    results = []

    for item in metadata["items"]:
        image_url = item.get("signed_url")

        if not image_url:
            print("Skipping:", item["filename"])
            continue

        try:
            print("Searching:", item["label"])

            matches = search_clothing(
                image_url=image_url,
                label=item["label"],
            )

            results.append({
                "filename": item["filename"],
                "label": item["label"],
                "storage_path": item["storage_path"],
                "matches": matches,
            })

            print("Search successful")

        except Exception as error:
            print("Search failed:", error)

    output_path = path.parent / "channel3_matches.json"

    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("Saved:", output_path)


if __name__ == "__main__":
    search_all_from_metadata()
