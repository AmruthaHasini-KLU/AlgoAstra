from pathlib import Path
import csv
import re
import sys


# ============================================================
# PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = next(
    p for p in CURRENT_FILE.parents
    if (p / "student_resource").exists()
)

CLEANED = PROJECT_ROOT / "student_resource" / "cleaned"
SUBMISSION_ROOT = CURRENT_FILE.parents[3]
OUTPUT = SUBMISSION_ROOT / "output"


# ============================================================
# SETTINGS
# ============================================================

MAX_BLOCK_SIZE = 50


# ============================================================
# BLOCKING KEYS
# ============================================================

def make_keys(row):
    """
    Create V1 blocking keys for a business record.

    V1 uses three blocking strategies:

    1. Exact normalized name + country
    2. First 6 characters of normalized name + country
    3. Address number + first 4 characters of name + country
    """

    country = row["clean_country"].strip()
    name = row["clean_name"].strip()
    address = row["clean_address"].strip()

    keys = set()

    # --------------------------------------------------------
    # 1. Exact normalized name + country
    # --------------------------------------------------------
    if len(name) >= 4:
        keys.add(
            f"NAME_EXACT|{country}|{name}"
        )

    # --------------------------------------------------------
    # 2. Name prefix + country
    # --------------------------------------------------------
    if len(name) >= 6:
        keys.add(
            f"NAME_PREFIX|{country}|{name[:6]}"
        )

    # --------------------------------------------------------
    # 3. Address number + name prefix + country
    # --------------------------------------------------------
    number_match = re.search(r"\b\d+\b", address)

    if number_match and len(name) >= 4:
        number = number_match.group()

        keys.add(
            f"NUMBER_NAME|{country}|{number}|{name[:4]}"
        )

    return keys


# ============================================================
# FILE READER
# ============================================================

def read_rows(path):
    """
    Read a TSV file and yield rows as dictionaries.
    """

    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        yield from csv.DictReader(
            file,
            delimiter="\t"
        )


# ============================================================
# PASS 1
# COUNT BLOCKING KEYS
# ============================================================

def count_keys(split):
    """
    Pass 1:

    Count how many Source2/Source3 records belong to
    each blocking key.

    Only keys appearing in <= MAX_BLOCK_SIZE records
    will be used in Pass 2.
    """

    counts = {}

    total_records = 0

    for source in (2, 3):

        path = (
            CLEANED
            / split
            / f"{split}_source{source}.tsv"
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        print(
            f"Scanning Source {source}...",
            flush=True
        )

        for row in read_rows(path):

            total_records += 1

            for key in make_keys(row):

                counts[key] = counts.get(key, 0) + 1

            if total_records % 1_000_000 == 0:

                print(
                    f"Scanned {total_records:,} records",
                    flush=True
                )

    print(
        f"Finished counting {total_records:,} records.",
        flush=True
    )

    useful_keys = {
        key
        for key, count in counts.items()
        if count <= MAX_BLOCK_SIZE
    }

    print(
        f"Total unique keys: {len(counts):,}",
        flush=True
    )

    print(
        f"Useful keys (<= {MAX_BLOCK_SIZE} records): "
        f"{len(useful_keys):,}",
        flush=True
    )

    return useful_keys


# ============================================================
# PASS 2
# BUILD COMPACT CANDIDATE INDEX
# ============================================================

def build_index(split, useful_keys):
    """
    Pass 2:

    Build a compact in-memory dictionary containing only
    useful blocking keys.

    key -> list of target entity IDs
    """

    index = {}

    total_records = 0

    for source in (2, 3):

        path = (
            CLEANED
            / split
            / f"{split}_source{source}.tsv"
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        print(
            f"Indexing Source {source}...",
            flush=True
        )

        for row in read_rows(path):

            total_records += 1

            entity_id = row["entity_id"]

            for key in make_keys(row):

                if key not in useful_keys:
                    continue

                bucket = index.get(key)

                if bucket is None:
                    index[key] = [entity_id]
                else:
                    bucket.append(entity_id)

            if total_records % 1_000_000 == 0:

                print(
                    f"Indexed {total_records:,} records",
                    flush=True
                )

    print(
        f"Finished indexing {total_records:,} records.",
        flush=True
    )

    return index


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(split, index, useful_keys):
    """
    Generate candidate entity IDs for every Source1 record.
    """

    source1_path = (
        CLEANED
        / split
        / f"{split}_source1.tsv"
    )

    if not source1_path.is_file():
        raise FileNotFoundError(source1_path)

    if split == "train":

        output_file = (
            OUTPUT
            / "train_candidate_pairs.tsv"
        )

    else:

        output_file = (
            OUTPUT
            / "candidate_pairs.tsv"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    total_records = 0
    total_candidates = 0
    maximum_candidates = 0

    print(
        "Generating Source1 candidates...",
        flush=True
    )

    with source1_path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as input_file:

        reader = csv.DictReader(
            input_file,
            delimiter="\t"
        )

        with output_file.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as output_handle:

            writer = csv.writer(
                output_handle,
                delimiter="\t"
            )

            writer.writerow(
                [
                    "source1_entity_id",
                    "candidate_entity_ids"
                ]
            )

            for row in reader:

                total_records += 1

                candidates = set()

                for key in make_keys(row):

                    if key not in useful_keys:
                        continue

                    matches = index.get(key)

                    if matches:
                        candidates.update(matches)

                candidate_count = len(candidates)

                total_candidates += candidate_count

                if candidate_count > maximum_candidates:
                    maximum_candidates = candidate_count

                writer.writerow(
                    [
                        row["entity_id"],
                        ",".join(sorted(candidates))
                    ]
                )

                if total_records % 100_000 == 0:

                    average = (
                        total_candidates
                        / total_records
                    )

                    print(
                        f"Processed {total_records:,} "
                        f"Source1 records | "
                        f"average candidates: {average:.2f} | "
                        f"max: {maximum_candidates}",
                        flush=True
                    )

    average_candidates = (
        total_candidates / total_records
        if total_records
        else 0
    )

    print()
    print(
        "CANDIDATE GENERATION COMPLETE"
    )
    print(
        f"Source1 records : {total_records:,}"
    )
    print(
        f"Total candidates: {total_candidates:,}"
    )
    print(
        f"Average         : {average_candidates:.2f}"
    )
    print(
        f"Maximum         : {maximum_candidates}"
    )
    print(
        f"Output          : {output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if (
        len(sys.argv) != 2
        or sys.argv[1] not in ("train", "test")
    ):
        raise SystemExit(
            "Usage: python3 candidates.py train "
            "OR python3 candidates.py test"
        )

    split = sys.argv[1]

    print("=" * 60)
    print("FAST V1 BLOCKING / CANDIDATE GENERATION")
    print("=" * 60)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Cleaned data : {CLEANED}"
    )

    print(
        f"Submission   : {SUBMISSION_ROOT}"
    )

    print(
        f"Output       : {OUTPUT}"
    )

    print(
        f"Split        : {split}"
    )

    print()

    # --------------------------------------------------------
    # PASS 1
    # --------------------------------------------------------

    print(
        "PASS 1: counting blocking keys"
    )

    useful_keys = count_keys(split)

    print()

    # --------------------------------------------------------
    # PASS 2
    # --------------------------------------------------------

    print(
        "PASS 2: building compact candidate index"
    )

    index = build_index(
        split,
        useful_keys
    )

    print()

    # --------------------------------------------------------
    # CANDIDATE GENERATION
    # --------------------------------------------------------

    generate_candidates(
        split,
        index,
        useful_keys
    )


if __name__ == "__main__":
    main()