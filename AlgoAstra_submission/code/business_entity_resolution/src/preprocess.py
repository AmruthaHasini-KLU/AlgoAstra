from pathlib import Path
import pandas as pd
import re
import unicodedata


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

# Current file:
# D:\ML Challenge\AlgoAstra_submission\code\
# business_entity_resolution\src\preprocess.py

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATASET = PROJECT_ROOT / "student_resource" / "dataset"
OUTPUT_FOLDER = PROJECT_ROOT / "student_resource" / "cleaned"


# --------------------------------------------------
# TEXT CLEANING FUNCTION
# --------------------------------------------------

def clean_text(value):
    text = unicodedata.normalize("NFKD", str(value))

    # Remove accents
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    # Lowercase and replace &
    text = text.lower().replace("&", " and ")

    # Remove punctuation and keep letters/numbers
    text = re.sub(r"[^a-z0-9]+", " ", text)

    # Remove extra spaces
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------
# FIND SOURCE FILES
# --------------------------------------------------

files = sorted(
    DATASET.glob("*/*_source[123].tsv")
)

if not files:
    raise FileNotFoundError(
        f"No source files found in {DATASET}"
    )


# --------------------------------------------------
# PROCESS FILES
# --------------------------------------------------

for file in files:

    output_file = (
        OUTPUT_FOLDER
        / file.parent.name
        / file.name
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    first_chunk = True
    total = 0

    # Read 50,000 rows at a time
    for chunk in pd.read_csv(
        file,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=50_000
    ):

        # Clean business name
        chunk["clean_name"] = (
            chunk["business_name"]
            .map(clean_text)
        )

        # Clean business address
        chunk["clean_address"] = (
            chunk["business_address"]
            .map(clean_text)
        )

        # Clean country
        chunk["clean_country"] = (
            chunk["country"]
            .map(clean_text)
        )

        # Save cleaned chunk
        chunk.to_csv(
            output_file,
            sep="\t",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk
        )

        total += len(chunk)
        first_chunk = False

    print(
        f"{file.parent.name}/{file.name}: "
        f"{total:,} rows cleaned"
    )


print("\nDone!")
print(
    f"Cleaned files are in: {OUTPUT_FOLDER}"
)