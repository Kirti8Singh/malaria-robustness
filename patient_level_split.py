from pathlib import Path
import csv
import ast
import random
from collections import defaultdict, Counter

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PARASITIZED_DIR = PROJECT_ROOT / "cell_images" / "Parasitized"
UNINFECTED_DIR = PROJECT_ROOT / "cell_images" / "Uninfected"

PARASITIZED_CSV = PROJECT_ROOT / "patientid_cellmapping_parasitized.csv"
UNINFECTED_CSV = PROJECT_ROOT / "patientid_cellmapping_uninfected.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "dataset"

RANDOM_SEED = 42


# ============================================================
# LOAD CSV MAPPING
# ============================================================

def load_mapping(csv_path):

    mapping = {}

    with open(csv_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            patient_id, filenames_text = line.split(",", 1)

            patient_id = patient_id.strip()

            start = filenames_text.find("[")
            end = filenames_text.find("]")

            if start == -1 or end == -1:
                continue

            filenames_text = filenames_text[start:end + 1]

            try:
                filenames = ast.literal_eval(filenames_text)
            except (ValueError, SyntaxError):
                continue

            if isinstance(filenames, list):
                mapping[patient_id] = filenames

    return mapping


# ============================================================
# EXTRACT PATIENT ID
# ============================================================

def patient_id_from_filename(filename):
    """
    Example:
    C33P1thinF_IMG_20150619_114756a_cell_179.png

    -> C33P1thinF
    """

    return filename.split("_IMG_")[0]


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("PATIENT-LEVEL DATASET SPLIT")
print("=" * 70)

random.seed(RANDOM_SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load mappings
# ------------------------------------------------------------

print("\nLoading patient mappings...")

parasitized_mapping = load_mapping(PARASITIZED_CSV)
uninfected_mapping = load_mapping(UNINFECTED_CSV)

print(f"Parasitized patients : {len(parasitized_mapping)}")
print(f"Uninfected patients  : {len(uninfected_mapping)}")


# ------------------------------------------------------------
# Create patient -> class information
# ------------------------------------------------------------

all_patients = set(parasitized_mapping.keys()) | set(uninfected_mapping.keys())

both_classes = (
    set(parasitized_mapping.keys())
    & set(uninfected_mapping.keys())
)

uninfected_only = (
    set(uninfected_mapping.keys())
    - set(parasitized_mapping.keys())
)

print(f"Unique patients      : {len(all_patients)}")
print(f"Patients in both     : {len(both_classes)}")
print(f"Uninfected only      : {len(uninfected_only)}")


# ------------------------------------------------------------
# Patient-level split
# ------------------------------------------------------------

print("\nCreating patient-level split...")

# We deliberately split the two patient groups separately:
#
# Group 1:
# patients having BOTH parasitized and uninfected cells
#
# Group 2:
# patients having ONLY uninfected cells
#
# This keeps the class structure reasonably balanced.

both_list = sorted(both_classes)
uninfected_only_list = sorted(uninfected_only)

random.shuffle(both_list)
random.shuffle(uninfected_only_list)


# 151 patients -> 106 train, 23 validation, 22 test
both_train = both_list[:106]
both_val = both_list[106:129]
both_test = both_list[129:]


# 50 patients -> 35 train, 7 validation, 8 test
uninfected_train = uninfected_only_list[:35]
uninfected_val = uninfected_only_list[35:42]
uninfected_test = uninfected_only_list[42:]


patient_split = {}

for patient in both_train:
    patient_split[patient] = "train"

for patient in both_val:
    patient_split[patient] = "validation"

for patient in both_test:
    patient_split[patient] = "test"

for patient in uninfected_train:
    patient_split[patient] = "train"

for patient in uninfected_val:
    patient_split[patient] = "validation"

for patient in uninfected_test:
    patient_split[patient] = "test"


# ------------------------------------------------------------
# Verify every patient assigned exactly once
# ------------------------------------------------------------

assert len(patient_split) == len(all_patients)

assert set(patient_split.keys()) == all_patients

print("\nPatient split:")
print(f"Train      : {sum(v == 'train' for v in patient_split.values())}")
print(f"Validation : {sum(v == 'validation' for v in patient_split.values())}")
print(f"Test       : {sum(v == 'test' for v in patient_split.values())}")


# ------------------------------------------------------------
# Create master metadata
# ------------------------------------------------------------

print("\nCreating master metadata...")

records = []

# Parasitized images
for patient_id, filenames in parasitized_mapping.items():

    for filename in filenames:

        records.append({
            "filename": filename,
            "image_path": str(
                Path("cell_images") / "Parasitized" / filename
            ),
            "class": "parasitized",
            "label": 1,
            "patient_id": patient_id,
            "split": patient_split[patient_id]
        })


# Uninfected images
for patient_id, filenames in uninfected_mapping.items():

    for filename in filenames:

        records.append({
            "filename": filename,
            "image_path": str(
                Path("cell_images") / "Uninfected" / filename
            ),
            "class": "uninfected",
            "label": 0,
            "patient_id": patient_id,
            "split": patient_split[patient_id]
        })


# Sort for reproducibility
records.sort(key=lambda x: (x["split"], x["class"], x["filename"]))


# ------------------------------------------------------------
# Save master metadata
# ------------------------------------------------------------

metadata_path = OUTPUT_DIR / "master_metadata_patient_split.csv"

with open(metadata_path, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "image_path",
            "class",
            "label",
            "patient_id",
            "split"
        ]
    )

    writer.writeheader()
    writer.writerows(records)


# ------------------------------------------------------------
# Save patient-level split
# ------------------------------------------------------------

patient_split_path = OUTPUT_DIR / "patient_split.csv"

with open(patient_split_path, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "patient_id",
        "split"
    ])

    for patient_id in sorted(patient_split):

        writer.writerow([
            patient_id,
            patient_split[patient_id]
        ])


# ------------------------------------------------------------
# Calculate image-level statistics
# ------------------------------------------------------------

split_class_counts = defaultdict(Counter)

for record in records:

    split_class_counts[record["split"]][record["class"]] += 1


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SPLIT SUMMARY")
print("=" * 70)

for split in ["train", "validation", "test"]:

    parasitized_count = split_class_counts[split]["parasitized"]
    uninfected_count = split_class_counts[split]["uninfected"]

    total = parasitized_count + uninfected_count

    print(f"\n{split.upper()}")
    print("-" * 40)
    print(f"Patients   : {sum(v == split for v in patient_split.values())}")
    print(f"Parasitized: {parasitized_count:,}")
    print(f"Uninfected : {uninfected_count:,}")
    print(f"Total      : {total:,}")


# ------------------------------------------------------------
# Verify no patient appears in multiple splits
# ------------------------------------------------------------

patient_to_splits = defaultdict(set)

for record in records:

    patient_to_splits[record["patient_id"]].add(
        record["split"]
    )

leakage_patients = {
    patient: splits
    for patient, splits in patient_to_splits.items()
    if len(splits) > 1
}


# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

if len(leakage_patients) == 0:

    print("PASS: No patient appears in multiple splits.")

else:

    print("WARNING: Patient leakage detected!")
    print(f"Patients appearing in multiple splits: {len(leakage_patients)}")


print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(f"\n{metadata_path}")
print(f"{patient_split_path}")

print("\nPatient-level splitting completed successfully.")