from pathlib import Path
import csv

CURRENT = Path(__file__).resolve()

PROJECT_ROOT = next(
    p for p in CURRENT.parents
    if (p / "student_resource").exists()
)

CLEANED = PROJECT_ROOT / "student_resource" / "cleaned"
OUTPUT = PROJECT_ROOT / "AlgoAstra_submission" / "output"

TEST = CLEANED / "test"

S1 = TEST / "test_source1.tsv"
S2 = TEST / "test_source2.tsv"
S3 = TEST / "test_source3.tsv"

OUT = OUTPUT / "candidate_pairs.tsv"

MAX_BLOCK = 50


def val(x):
    return str(x or "").strip()


def address_number(address):
    for token in address.split():
        if any(c.isdigit() for c in token):
            return token
    return ""


def add(index, key, entity_id):
    if not key:
        return

    bucket = index.setdefault(key, [])

    if len(bucket) <= MAX_BLOCK:
        bucket.append(entity_id)


print("Building fast test indexes...", flush=True)

name_exact = {}
name_prefix = {}
number_name = {}

total = 0

for filename in [S2, S3]:

    print("Reading:", filename.name, flush=True)

    with open(filename, encoding="utf-8", newline="") as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            entity_id = val(row.get("entity_id"))
            country = val(row.get("clean_country"))
            name = val(row.get("clean_name"))
            address = val(row.get("clean_address"))

            if not entity_id:
                continue

            total += 1

            if country and name:
                add(
                    name_exact,
                    country + "|" + name,
                    entity_id
                )

            if country and len(name) >= 6:
                add(
                    name_prefix,
                    country + "|" + name[:6],
                    entity_id
                )

            number = address_number(address)

            if country and number and len(name) >= 4:
                add(
                    number_name,
                    country + "|" + number + "|" + name[:4],
                    entity_id
                )

print("Target records:", f"{total:,}", flush=True)

# Remove oversized blocks
name_exact = {
    k: v for k, v in name_exact.items()
    if len(v) <= MAX_BLOCK
}

name_prefix = {
    k: v for k, v in name_prefix.items()
    if len(v) <= MAX_BLOCK
}

number_name = {
    k: v for k, v in number_name.items()
    if len(v) <= MAX_BLOCK
}

print("Indexes ready.", flush=True)

OUTPUT.mkdir(parents=True, exist_ok=True)

total_s1 = 0
total_candidates = 0
max_candidates = 0

print("Generating candidate_pairs.tsv...", flush=True)

with open(S1, encoding="utf-8", newline="") as f1, \
     open(OUT, "w", encoding="utf-8", newline="") as fout:

    reader = csv.DictReader(f1, delimiter="\t")

    writer = csv.writer(fout, delimiter="\t")

    writer.writerow([
        "source1_entity_id",
        "candidate_entity_ids"
    ])

    for row in reader:

        source1_id = val(row.get("entity_id"))
        country = val(row.get("clean_country"))
        name = val(row.get("clean_name"))
        address = val(row.get("clean_address"))

        candidates = set()

        # Exact name
        if country and name:
            candidates.update(
                name_exact.get(
                    country + "|" + name,
                    []
                )
            )

        # Name prefix
        if country and len(name) >= 6:
            candidates.update(
                name_prefix.get(
                    country + "|" + name[:6],
                    []
                )
            )

        # Number + name
        number = address_number(address)

        if country and number and len(name) >= 4:
            candidates.update(
                number_name.get(
                    country + "|" + number + "|" + name[:4],
                    []
                )
            )

        candidate_list = sorted(candidates)

        writer.writerow([
            source1_id,
            ",".join(candidate_list)
        ])

        total_s1 += 1
        total_candidates += len(candidate_list)

        max_candidates = max(
            max_candidates,
            len(candidate_list)
        )

        if total_s1 % 100000 == 0:
            print(
                f"{total_s1:,} Source1 | "
                f"avg={total_candidates / total_s1:.2f} | "
                f"max={max_candidates}",
                flush=True
            )

print()
print("DONE")
print("Source1:", f"{total_s1:,}")
print("Candidates:", f"{total_candidates:,}")
print("Average:", f"{total_candidates / total_s1:.2f}")
print("Maximum:", max_candidates)
print("Output:", OUT)