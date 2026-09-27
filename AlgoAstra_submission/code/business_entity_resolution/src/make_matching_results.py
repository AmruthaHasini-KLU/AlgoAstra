from pathlib import Path
import pandas as pd
import csv

ROOT = next(
    p for p in Path(__file__).resolve().parents
    if (p / "student_resource").exists()
)

TEST = ROOT / "student_resource" / "cleaned" / "test"
OUTPUT = ROOT / "AlgoAstra_submission" / "output"

S1 = TEST / "test_source1.tsv"
S2 = TEST / "test_source2.tsv"
S3 = TEST / "test_source3.tsv"

CANDIDATES = OUTPUT / "candidate_pairs.tsv"
RESULT = OUTPUT / "matching_results.tsv"

print("Building exact-name target index...", flush=True)

name_index = {}

for path in [S2, S3]:

    print("Reading", path.name, flush=True)

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["entity_id", "clean_name", "clean_country"],
        dtype=str
    ).fillna("")

    print("Rows:", len(df), flush=True)

    for r in df.itertuples(index=False):

        if not r.clean_country or not r.clean_name:
            continue

        key = r.clean_country + "|" + r.clean_name

        bucket = name_index.setdefault(key, [])

        # Ignore extremely common names
        if len(bucket) < 50:
            bucket.append(r.entity_id)

    del df

print("Name index ready:", len(name_index), "keys", flush=True)

print("Generating final matching_results.tsv...", flush=True)

df1 = pd.read_csv(
    S1,
    sep="\t",
    usecols=["entity_id", "clean_name", "clean_country"],
    dtype=str
).fillna("")

count = 0
matches_count = 0

with open(RESULT, "w", encoding="utf-8", newline="") as fout:

    writer = csv.writer(fout, delimiter="\t")

    writer.writerow([
        "source1_entity_id",
        "matched_entity_ids"
    ])

    for r in df1.itertuples(index=False):

        key = r.clean_country + "|" + r.clean_name

        matches = name_index.get(key, [])

        writer.writerow([
            r.entity_id,
            ",".join(matches)
        ])

        count += 1
        matches_count += len(matches)

        if count % 100000 == 0:
            print(
                f"{count:,} Source1 | "
                f"matches={matches_count:,}",
                flush=True
            )

print()
print("DONE")
print("Source1 records:", f"{count:,}")
print("Matched IDs:", f"{matches_count:,}")
print("Output:", RESULT)