from pathlib import Path
import csv


# ============================================================
# FIND PROJECT ROOT
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = None

for parent in CURRENT_FILE.parents:

    if (parent / "student_resource").exists():

        PROJECT_ROOT = parent
        break


if PROJECT_ROOT is None:

    raise FileNotFoundError(
        "Could not find project root containing "
        "'student_resource'."
    )


# ============================================================
# FILE PATHS
# ============================================================

GROUND_TRUTH = (
    PROJECT_ROOT
    / "student_resource"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)


SUBMISSION_ROOT = CURRENT_FILE.parents[3]


CANDIDATES = (
    SUBMISSION_ROOT
    / "output"
    / "train_candidate_pairs_v2fast.tsv"
)


# ============================================================
# CHECK FILES
# ============================================================

if not GROUND_TRUTH.is_file():

    raise FileNotFoundError(
        f"Ground truth file not found:\n"
        f"{GROUND_TRUTH}"
    )


if not CANDIDATES.is_file():

    raise FileNotFoundError(
        f"Candidate file not found:\n"
        f"{CANDIDATES}"
    )


# ============================================================
# HELPER
# ============================================================

def id_set(value):
    """
    Convert comma-separated IDs into a set.

    Empty values become an empty set.

    Example:

        "A123,A456,A789"

    becomes:

        {"A123", "A456", "A789"}
    """

    if not value:

        return set()

    return set(
        item
        for item in value.split(",")
        if item
    )


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("=" * 60)
print("FAST CANDIDATE EVALUATION")
print("=" * 60)

print(
    f"Project root : {PROJECT_ROOT}"
)

print(
    f"Ground truth : {GROUND_TRUTH}"
)

print(
    f"Candidates   : {CANDIDATES}"
)

print("=" * 60)


print(
    "Loading training ground truth into memory...",
    flush=True
)


ground_truth = {}

ground_truth_match_count = 0


with GROUND_TRUTH.open(
    "r",
    encoding="utf-8",
    newline=""
) as file:

    reader = csv.DictReader(
        file,
        delimiter="\t"
    )

    expected_columns = {
        "source1_entity_id",
        "matched_entity_ids"
    }

    actual_columns = set(
        reader.fieldnames or []
    )

    if not expected_columns.issubset(
        actual_columns
    ):

        raise ValueError(
            "train_ground_truth.tsv does not contain "
            "the expected columns.\n"
            f"Expected: {expected_columns}\n"
            f"Found: {reader.fieldnames}"
        )


    for row in reader:

        source1_id = row[
            "source1_entity_id"
        ]

        matches = id_set(
            row["matched_entity_ids"]
        )

        ground_truth[
            source1_id
        ] = matches

        ground_truth_match_count += len(
            matches
        )


        if len(ground_truth) % 500_000 == 0:

            print(
                f"Loaded "
                f"{len(ground_truth):,} "
                f"ground-truth records",
                flush=True
            )


print(
    f"Ground truth loaded: "
    f"{len(ground_truth):,} Source1 records",
    flush=True
)


# ============================================================
# EVALUATE CANDIDATES
# ============================================================

print()
print(
    "Evaluating candidate pairs...",
    flush=True
)

source1_count = 0

candidate_count = 0

empty_candidate_count = 0

true_matches_found = 0

missing_ground_truth_count = 0

# Useful diagnostic statistics.
source1_with_true_match = 0

source1_with_candidate_match = 0

max_candidates = 0


with CANDIDATES.open(
    "r",
    encoding="utf-8",
    newline=""
) as file:

    reader = csv.DictReader(
        file,
        delimiter="\t"
    )

    expected_columns = {
        "source1_entity_id",
        "candidate_entity_ids"
    }

    actual_columns = set(
        reader.fieldnames or []
    )

    if not expected_columns.issubset(
        actual_columns
    ):

        raise ValueError(
            "train_candidate_pairs.tsv does not contain "
            "the expected columns.\n"
            f"Expected: {expected_columns}\n"
            f"Found: {reader.fieldnames}"
        )


    for row in reader:

        source1_id = row[
            "source1_entity_id"
        ]


        candidates = id_set(
            row["candidate_entity_ids"]
        )


        # ----------------------------------------------------
        # Find actual ground-truth matches directly from
        # the Python dictionary.
        # ----------------------------------------------------

        actual_matches = ground_truth.get(
            source1_id
        )


        if actual_matches is None:

            missing_ground_truth_count += 1

            raise ValueError(
                f"{source1_id} is missing from "
                "train_ground_truth.tsv"
            )


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        source1_count += 1

        candidate_count += len(
            candidates
        )


        if not candidates:

            empty_candidate_count += 1


        if len(candidates) > max_candidates:

            max_candidates = len(
                candidates
            )


        if actual_matches:

            source1_with_true_match += 1


        # ----------------------------------------------------
        # Candidate recall
        # ----------------------------------------------------

        found = (
            actual_matches
            & candidates
        )


        if found:

            source1_with_candidate_match += 1


        true_matches_found += len(
            found
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if source1_count % 200_000 == 0:

            average_candidates = (
                candidate_count
                / source1_count
            )

            print(
                f"Checked "
                f"{source1_count:,} "
                f"Source1 records | "
                f"average candidates: "
                f"{average_candidates:.2f}",
                flush=True
            )


# ============================================================
# CALCULATE METRICS
# ============================================================

if ground_truth_match_count:

    pair_recall = (
        true_matches_found
        / ground_truth_match_count
    )

else:

    pair_recall = 0.0


if source1_with_true_match:

    source1_recall = (
        source1_with_candidate_match
        / source1_with_true_match
    )

else:

    source1_recall = 0.0


if source1_count:

    average_candidates = (
        candidate_count
        / source1_count
    )

else:

    average_candidates = 0.0


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("CANDIDATE EVALUATION RESULTS")
print("=" * 60)


print(
    f"Source 1 records: "
    f"{source1_count:,}"
)


print(
    f"Total candidate pairs: "
    f"{candidate_count:,}"
)


print(
    f"Average candidates per Source 1: "
    f"{average_candidates:.2f}"
)


print(
    f"Maximum candidates for one Source 1: "
    f"{max_candidates}"
)


print(
    f"Source 1 records with no candidates: "
    f"{empty_candidate_count:,}"
)


print(
    f"Source 1 records with true matches: "
    f"{source1_with_true_match:,}"
)


print(
    f"Actual matching pairs: "
    f"{ground_truth_match_count:,}"
)


print(
    f"Actual matching pairs found: "
    f"{true_matches_found:,}"
)


print(
    f"Pair-level candidate recall: "
    f"{pair_recall:.2%}"
)


print(
    f"Source1-level candidate recall: "
    f"{source1_recall:.2%}"
)


print("=" * 60)


# ============================================================
# SANITY CHECK
# ============================================================

if missing_ground_truth_count:

    print(
        f"WARNING: "
        f"{missing_ground_truth_count:,} "
        "Source1 IDs were missing from ground truth."
    )

else:

    print(
        "All candidate Source1 IDs were found "
        "in the training ground truth."
    )


print()
print(
    "Candidate evaluation completed."
)