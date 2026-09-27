from pathlib import Path
import csv
import hashlib
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
DATASET = PROJECT_ROOT / "student_resource" / "dataset"
SUBMISSION_ROOT = CURRENT_FILE.parents[3]
OUTPUT = SUBMISSION_ROOT / "output"


CANDIDATES = OUTPUT / "train_candidate_pairs_v2fast.tsv"

GROUND_TRUTH = (
    DATASET
    / "train"
    / "train_ground_truth.tsv"
)

FEATURE_OUTPUT = (
    OUTPUT
    / "training_features_sample.tsv"
)


# ============================================================
# SETTINGS
# ============================================================

# Number of Source1 records used for the first experiment.
MAX_SOURCE1 = 50_000

# Deterministic sampling fraction.
# This prevents always taking the first 50k records.
SAMPLE_MOD = 100_000

# About 50% of Source1 records will be selected until
# MAX_SOURCE1 is reached.
SAMPLE_THRESHOLD = 50_000


# ============================================================
# FAST TEXT FEATURES
# ============================================================

LEGAL_WORDS = {
    "inc",
    "llc",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "pvt",
    "private",
    "company",
}


def tokens(text):
    return {
        x
        for x in text.split()
        if len(x) >= 3
        and x not in LEGAL_WORDS
    }


def first_number(text):
    match = re.search(r"\b\d+\b", text)
    return match.group() if match else ""


def jaccard(a, b):
    if not a or not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


def prefix_match(a, b, length=4):
    if not a or not b:
        return 0

    return int(a[:length] == b[:length])


def exact_match(a, b):
    return int(bool(a) and a == b)


def length_difference(a, b):
    return abs(len(a) - len(b))


def character_overlap(a, b):
    """
    Very cheap character-set similarity.
    Not edit distance.
    """

    if not a or not b:
        return 0.0

    sa = set(a.replace(" ", ""))
    sb = set(b.replace(" ", ""))

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / len(sa | sb)


def make_features(source1, target):
    name1 = source1["clean_name"]
    name2 = target["clean_name"]

    address1 = source1["clean_address"]
    address2 = target["clean_address"]

    country1 = source1["clean_country"]
    country2 = target["clean_country"]

    name_tokens1 = tokens(name1)
    name_tokens2 = tokens(name2)

    address_tokens1 = set(address1.split())
    address_tokens2 = set(address2.split())

    number1 = first_number(address1)
    number2 = first_number(address2)

    return [
        exact_match(name1, name2),
        prefix_match(name1, name2, 4),
        prefix_match(name1, name2, 6),
        jaccard(name_tokens1, name_tokens2),

        exact_match(address1, address2),
        jaccard(address_tokens1, address_tokens2),

        int(country1 == country2),
        int(number1 != "" and number1 == number2),

        length_difference(name1, name2),
        length_difference(address1, address2),

        character_overlap(name1, name2),
        character_overlap(address1, address2),
    ]


# ============================================================
# READ GROUND TRUTH
# ============================================================

def load_ground_truth():
    """
    Load training ground truth.

    Ground-truth format:

        source1_entity_id    matched_entity_ids

    matched_entity_ids contains the matching Source2/Source3
    entity IDs, separated by commas.
    """

    print("Loading ground truth...", flush=True)

    truth = {}

    with GROUND_TRUTH.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t"
        )

        fieldnames = reader.fieldnames or []

        print(
            "Ground truth columns:",
            fieldnames,
            flush=True
        )

        required_columns = {
            "source1_entity_id",
            "matched_entity_ids",
        }

        missing = required_columns - set(fieldnames)

        if missing:
            raise RuntimeError(
                "Missing ground-truth columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            source1_id = row["source1_entity_id"]

            matched_value = (
                row.get("matched_entity_ids", "")
                or ""
            )

            targets = set()

            for target_id in matched_value.split(","):

                target_id = target_id.strip()

                if target_id:
                    targets.add(target_id)

            truth[source1_id] = targets

    total_matches = sum(
        len(targets)
        for targets in truth.values()
    )

    print(
        f"Ground-truth Source1 records: "
        f"{len(truth):,}",
        flush=True
    )

    print(
        f"Ground-truth matching pairs: "
        f"{total_matches:,}",
        flush=True
    )

    return truth
# ============================================================
# SELECT SOURCE1 SAMPLE
# ============================================================

def select_source1(truth):

    selected = set()

    for source1_id in truth:

        digest = hashlib.md5(
            source1_id.encode("utf-8")
        ).digest()

        value = int.from_bytes(
            digest[:4],
            byteorder="big"
        ) % SAMPLE_MOD

        if value < SAMPLE_THRESHOLD:

            selected.add(source1_id)

            if len(selected) >= MAX_SOURCE1:
                break

    print(
        f"Selected Source1 records: {len(selected):,}",
        flush=True
    )

    return selected


# ============================================================
# LOAD SOURCE1 RECORDS
# ============================================================

def load_source1(selected):

    path = (
        CLEANED
        / "train"
        / "train_source1.tsv"
    )

    records = {}

    print(
        "Loading selected Source1 records...",
        flush=True
    )

    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t"
        )

        for row in reader:

            entity_id = row["entity_id"]

            if entity_id in selected:
                records[entity_id] = row

    print(
        f"Loaded Source1 records: {len(records):,}",
        flush=True
    )

    return records


# ============================================================
# READ CANDIDATES
# ============================================================

def read_selected_candidates(selected):

    print(
        "Reading selected candidate pairs...",
        flush=True
    )

    pairs = []

    target_ids = set()

    with CANDIDATES.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t"
        )

        for row in reader:

            source1_id = row["source1_entity_id"]

            if source1_id not in selected:
                continue

            candidate_string = (
                row["candidate_entity_ids"]
            )

            if not candidate_string:
                continue

            for target_id in candidate_string.split(","):

                target_id = target_id.strip()

                if not target_id:
                    continue

                pairs.append(
                    (source1_id, target_id)
                )

                target_ids.add(target_id)

    print(
        f"Candidate pairs selected: {len(pairs):,}",
        flush=True
    )

    print(
        f"Unique target IDs needed: {len(target_ids):,}",
        flush=True
    )

    return pairs, target_ids


# ============================================================
# LOAD ONLY REQUIRED TARGET RECORDS
# ============================================================

def load_targets(target_ids):

    records = {}

    remaining = set(target_ids)

    print(
        "Loading required Source2/Source3 records...",
        flush=True
    )

    for source in (2, 3):

        if not remaining:
            break

        path = (
            CLEANED
            / "train"
            / f"train_source{source}.tsv"
        )

        print(
            f"Scanning Source {source}...",
            flush=True
        )

        with path.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as file:

            reader = csv.DictReader(
                file,
                delimiter="\t"
            )

            for row in reader:

                entity_id = row["entity_id"]

                if entity_id in remaining:

                    records[entity_id] = row
                    remaining.remove(entity_id)

        print(
            f"Found so far: {len(records):,}",
            flush=True
        )

    if remaining:

        print(
            f"WARNING: {len(remaining):,} "
            "target IDs were not found.",
            flush=True
        )

    return records


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(
    pairs,
    source1_records,
    target_records,
    truth
):

    print(
        "Building feature matrix...",
        flush=True
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_names = [
        "name_exact",
        "name_prefix4",
        "name_prefix6",
        "name_token_jaccard",

        "address_exact",
        "address_token_jaccard",

        "country_exact",
        "address_number_exact",

        "name_length_diff",
        "address_length_diff",

        "name_character_overlap",
        "address_character_overlap",

        "label",
    ]

    positive = 0
    negative = 0
    written = 0

    with FEATURE_OUTPUT.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.writer(
            file,
            delimiter="\t"
        )

        writer.writerow(
            ["source1_entity_id", "candidate_entity_id"]
            + feature_names
        )

        for source1_id, target_id in pairs:

            source1 = source1_records.get(
                source1_id
            )

            target = target_records.get(
                target_id
            )

            if source1 is None or target is None:
                continue

            features = make_features(
                source1,
                target
            )

            label = int(
                target_id in truth.get(
                    source1_id,
                    set()
                )
            )

            if label:
                positive += 1
            else:
                negative += 1

            writer.writerow(
                [
                    source1_id,
                    target_id,
                    *features,
                    label,
                ]
            )

            written += 1

    print()
    print("=" * 60)
    print("TRAINING FEATURE SAMPLE COMPLETE")
    print("=" * 60)
    print(
        f"Rows written : {written:,}"
    )
    print(
        f"Positive     : {positive:,}"
    )
    print(
        f"Negative     : {negative:,}"
    )

    if written:
        print(
            f"Positive %   : "
            f"{100 * positive / written:.3f}%"
        )

    print(
        f"Output       : {FEATURE_OUTPUT}"
    )
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FAST TRAINING FEATURE BUILDER")
    print("=" * 60)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Candidates   : {CANDIDATES}"
    )

    print(
        f"Ground truth : {GROUND_TRUTH}"
    )

    print(
        f"Output       : {FEATURE_OUTPUT}"
    )

    print()

    # 1. Ground truth
    truth = load_ground_truth()

    # 2. Select a deterministic Source1 sample
    selected = select_source1(truth)

    # 3. Load Source1 rows
    source1_records = load_source1(
        selected
    )

    # 4. Read candidate IDs for selected Source1
    pairs, target_ids = read_selected_candidates(
        selected
    )

    # 5. Load only the required target records
    target_records = load_targets(
        target_ids
    )

    # 6. Build features
    build_features(
        pairs,
        source1_records,
        target_records,
        truth
    )


if __name__ == "__main__":
    main()