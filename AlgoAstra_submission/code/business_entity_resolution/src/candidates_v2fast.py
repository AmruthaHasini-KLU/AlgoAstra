from pathlib import Path
import csv
import sqlite3
import sys


# ============================================================
# PATHS
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

CLEANED = PROJECT_ROOT / "student_resource" / "cleaned"

SUBMISSION_ROOT = CURRENT_FILE.parents[3]

OUTPUT = SUBMISSION_ROOT / "output"

V1_CANDIDATES = (
    OUTPUT / "train_candidate_pairs.tsv"
)

V2_CANDIDATES = (
    OUTPUT / "train_candidate_pairs_v2fast.tsv"
)

WORK = (
    SUBMISSION_ROOT / "work"
)

WORK.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

MAX_ADDRESS_BLOCK_SIZE = 50


# ============================================================
# READ TSV
# ============================================================

def read_rows(path):

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
# SOURCE PATH
# ============================================================

def source_path(
    split,
    source
):

    path = (
        CLEANED
        / split
        / f"{split}_source{source}.tsv"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


# ============================================================
# BUILD EXACT-ADDRESS SQLITE INDEX
# ============================================================

def build_address_index():

    database = (
        WORK
        / "address_index_v2fast.sqlite"
    )

    if database.exists():
        database.unlink()

    connection = sqlite3.connect(
        database
    )

    try:

        connection.execute(
            "PRAGMA journal_mode=OFF"
        )

        connection.execute(
            "PRAGMA synchronous=OFF"
        )

        connection.execute(
            "PRAGMA temp_store=FILE"
        )

        connection.execute(
            """
            CREATE TABLE address_keys (
                address_key TEXT,
                entity_id TEXT
            )
            """
        )

        total = 0

        for source in (2, 3):

            path = source_path(
                "train",
                source
            )

            print(
                f"Indexing Source {source} "
                f"addresses...",
                flush=True
            )

            batch = []

            for row in read_rows(path):

                total += 1

                country = (
                    row["clean_country"]
                    .strip()
                )

                address = (
                    row["clean_address"]
                    .strip()
                )

                if (
                    len(address) >= 8
                    and country
                ):

                    key = (
                        country
                        + "|"
                        + address
                    )

                    batch.append(
                        (
                            key,
                            row["entity_id"]
                        )
                    )

                if len(batch) >= 100_000:

                    connection.executemany(
                        """
                        INSERT INTO address_keys
                        VALUES (?, ?)
                        """,
                        batch
                    )

                    connection.commit()

                    batch.clear()

                if total % 1_000_000 == 0:

                    print(
                        f"Indexed "
                        f"{total:,} records",
                        flush=True
                    )

            if batch:

                connection.executemany(
                    """
                    INSERT INTO address_keys
                    VALUES (?, ?)
                    """,
                    batch
                )

                connection.commit()

        print()
        print(
            "Creating address index...",
            flush=True
        )

        connection.execute(
            """
            CREATE INDEX address_key_index
            ON address_keys(address_key)
            """
        )

        connection.commit()

        print(
            f"Address index ready: "
            f"{total:,} target records",
            flush=True
        )

        return database

    finally:

        connection.close()


# ============================================================
# MERGE V1 + EXACT ADDRESS CANDIDATES
# ============================================================

def generate_v2fast():

    database = (
        WORK
        / "address_index_v2fast.sqlite"
    )

    connection = sqlite3.connect(
        database
    )

    try:

        source1 = source_path(
            "train",
            1
        )

        total = 0
        total_candidates = 0
        max_candidates = 0

        with (
            V1_CANDIDATES.open(
                "r",
                encoding="utf-8",
                newline=""
            ) as v1_file,
            V2_CANDIDATES.open(
                "w",
                encoding="utf-8",
                newline=""
            ) as output_file
        ):

            v1_reader = csv.DictReader(
                v1_file,
                delimiter="\t"
            )

            writer = csv.writer(
                output_file,
                delimiter="\t"
            )

            writer.writerow(
                [
                    "source1_entity_id",
                    "candidate_entity_ids"
                ]
            )

            for row in read_rows(
                source1
            ):

                total += 1

                source1_id = row[
                    "entity_id"
                ]

                # ------------------------------------------------
                # Get V1 candidates.
                # ------------------------------------------------

                v1_row = next(
                    v1_reader,
                    None
                )

                if v1_row is None:

                    raise ValueError(
                        "V1 candidate file ended "
                        "before Source1."
                    )

                if (
                    v1_row[
                        "source1_entity_id"
                    ]
                    != source1_id
                ):

                    raise ValueError(
                        "Source1 ordering mismatch "
                        "between V1 candidates "
                        "and cleaned Source1."
                    )

                candidates = set()

                existing = v1_row[
                    "candidate_entity_ids"
                ]

                if existing:

                    candidates.update(
                        existing.split(",")
                    )

                # ------------------------------------------------
                # Add exact-address candidates.
                # ------------------------------------------------

                country = (
                    row["clean_country"]
                    .strip()
                )

                address = (
                    row["clean_address"]
                    .strip()
                )

                if (
                    len(address) >= 8
                    and country
                ):

                    address_key = (
                        country
                        + "|"
                        + address
                    )

                    matches = connection.execute(
                        """
                        SELECT entity_id
                        FROM address_keys
                        WHERE address_key = ?
                        LIMIT ?
                        """,
                        (
                            address_key,
                            MAX_ADDRESS_BLOCK_SIZE + 1
                        )
                    ).fetchall()

                    if (
                        len(matches)
                        <= MAX_ADDRESS_BLOCK_SIZE
                    ):

                        for match in matches:

                            candidates.add(
                                match[0]
                            )

                candidate_count = len(
                    candidates
                )

                total_candidates += (
                    candidate_count
                )

                if candidate_count > max_candidates:

                    max_candidates = (
                        candidate_count
                    )

                writer.writerow(
                    [
                        source1_id,
                        ",".join(
                            sorted(candidates)
                        )
                    ]
                )

                if total % 200_000 == 0:

                    average = (
                        total_candidates
                        / total
                    )

                    print(
                        f"Processed "
                        f"{total:,} Source1 records | "
                        f"average candidates: "
                        f"{average:.2f} | "
                        f"max: "
                        f"{max_candidates}",
                        flush=True
                    )

        print()
        print("=" * 60)
        print("V2-FAST COMPLETE")
        print("=" * 60)
        print(
            f"Source1 records : "
            f"{total:,}"
        )
        print(
            f"Total candidates: "
            f"{total_candidates:,}"
        )
        print(
            f"Average         : "
            f"{total_candidates / total:.2f}"
        )
        print(
            f"Maximum         : "
            f"{max_candidates}"
        )
        print(
            f"Output          : "
            f"{V2_CANDIDATES}"
        )
        print("=" * 60)

    finally:

        connection.close()


# ============================================================
# MAIN
# ============================================================

def main():

    if not V1_CANDIDATES.is_file():

        raise FileNotFoundError(
            f"V1 candidate file not found:\n"
            f"{V1_CANDIDATES}"
        )

    print("=" * 60)
    print("V2-FAST CANDIDATE GENERATION")
    print("=" * 60)
    print(
        f"Project root : "
        f"{PROJECT_ROOT}"
    )
    print(
        f"V1 candidates: "
        f"{V1_CANDIDATES}"
    )
    print(
        f"Output       : "
        f"{V2_CANDIDATES}"
    )
    print("=" * 60)

    build_address_index()

    print()
    print(
        "Merging exact-address candidates...",
        flush=True
    )

    generate_v2fast()

    print()
    print(
        "V2-fast candidate generation completed."
    )


if __name__ == "__main__":
    main()