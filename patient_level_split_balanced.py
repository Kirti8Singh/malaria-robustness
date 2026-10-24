from pathlib import Path
import csv
import ast
import random
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PARASITIZED_CSV = PROJECT_ROOT / "patientid_cellmapping_parasitized.csv"
UNINFECTED_CSV = PROJECT_ROOT / "patientid_cellmapping_uninfected.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "dataset"

RANDOM_SEED = 42

# Number of random candidate splits to try.
# Higher = more chances of finding a good split.
N_TRIALS = 5000

# Desired patient proportions
TRAIN_PATIENT_RATIO = 0.70
VAL_PATIENT_RATIO = 0.15
TEST_PATIENT_RATIO = 0.15

# Desired image proportions
TRAIN_IMAGE_RATIO = 0.70
VAL_IMAGE_RATIO = 0.15
TEST_IMAGE_RATIO = 0.15


# ============================================================
# LOAD PATIENT MAPPING
# ============================================================

def load_mapping(csv_path):

    mapping = {}

    with open(csv_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # First comma separates patient ID from the list
            patient_id, remaining = line.split(",", 1)

            patient_id = patient_id.strip()

            # Find the Python list inside the line.
            # The raw CSV has many trailing empty columns.
            start = remaining.find("[")
            end = remaining.find("]")

            if start == -1 or end == -1:
                continue

            filenames_text = remaining[start:end + 1]

            try:
                filenames = ast.literal_eval(filenames_text)

            except (ValueError, SyntaxError):

                continue

            if isinstance(filenames, list):

                mapping[patient_id] = filenames

    return mapping


# ============================================================
# BUILD PATIENT INFORMATION
# ============================================================

def build_patient_info(parasitized_mapping, uninfected_mapping):

    all_patients = set(parasitized_mapping) | set(uninfected_mapping)

    patient_info = {}

    for patient_id in all_patients:

        parasitized = len(
            parasitized_mapping.get(patient_id, [])
        )

        uninfected = len(
            uninfected_mapping.get(patient_id, [])
        )

        patient_info[patient_id] = {
            "parasitized": parasitized,
            "uninfected": uninfected,
            "total": parasitized + uninfected
        }

    return patient_info


# ============================================================
# CREATE RANDOM PATIENT SPLIT
# ============================================================

def create_random_split(patient_ids):

    shuffled = patient_ids.copy()

    random.shuffle(shuffled)

    n_patients = len(shuffled)

    n_train = round(
        n_patients * TRAIN_PATIENT_RATIO
    )

    n_val = round(
        n_patients * VAL_PATIENT_RATIO
    )

    # Everything remaining goes to test.
    # This guarantees every patient is assigned.
    n_test = n_patients - n_train - n_val

    train_patients = shuffled[:n_train]

    val_patients = shuffled[
        n_train:n_train + n_val
    ]

    test_patients = shuffled[
        n_train + n_val:
    ]

    return {
        "train": train_patients,
        "validation": val_patients,
        "test": test_patients
    }


# ============================================================
# CALCULATE SPLIT STATISTICS
# ============================================================

def calculate_statistics(split, patient_info):

    statistics = {}

    total_parasitized = sum(
        info["parasitized"]
        for info in patient_info.values()
    )

    total_uninfected = sum(
        info["uninfected"]
        for info in patient_info.values()
    )

    total_images = (
        total_parasitized +
        total_uninfected
    )

    total_patients = len(patient_info)

    for split_name, patients in split.items():

        parasitized = sum(
            patient_info[p]["parasitized"]
            for p in patients
        )

        uninfected = sum(
            patient_info[p]["uninfected"]
            for p in patients
        )

        total = parasitized + uninfected

        statistics[split_name] = {
            "patients": len(patients),
            "parasitized": parasitized,
            "uninfected": uninfected,
            "total": total,

            "patient_ratio":
                len(patients) / total_patients,

            "parasitized_ratio":
                parasitized / total_parasitized,

            "uninfected_ratio":
                uninfected / total_uninfected,

            "image_ratio":
                total / total_images
        }

    return statistics


# ============================================================
# SPLIT SCORE
# ============================================================

def calculate_score(statistics):

    score = 0.0

    target_patient_ratios = {
        "train": TRAIN_PATIENT_RATIO,
        "validation": VAL_PATIENT_RATIO,
        "test": TEST_PATIENT_RATIO
    }

    target_image_ratios = {
        "train": TRAIN_IMAGE_RATIO,
        "validation": VAL_IMAGE_RATIO,
        "test": TEST_IMAGE_RATIO
    }

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        stats = statistics[split_name]

        # ----------------------------------------------------
        # Patient-count difference
        # ----------------------------------------------------

        patient_error = abs(
            stats["patient_ratio"]
            - target_patient_ratios[split_name]
        )

        # ----------------------------------------------------
        # Parasitized image difference
        # ----------------------------------------------------

        parasitized_error = abs(
            stats["parasitized_ratio"]
            - target_image_ratios[split_name]
        )

        # ----------------------------------------------------
        # Uninfected image difference
        # ----------------------------------------------------

        uninfected_error = abs(
            stats["uninfected_ratio"]
            - target_image_ratios[split_name]
        )

        # ----------------------------------------------------
        # Total image difference
        # ----------------------------------------------------

        image_error = abs(
            stats["image_ratio"]
            - target_image_ratios[split_name]
        )

        # ----------------------------------------------------
        # Weighted score
        #
        # Image/class distribution matters more than
        # exact patient count.
        # ----------------------------------------------------

        score += (
            1.0 * patient_error
            + 4.0 * parasitized_error
            + 4.0 * uninfected_error
            + 2.0 * image_error
        )

    return score


# ============================================================
# CREATE MASTER METADATA
# ============================================================

def create_master_metadata(
    split,
    parasitized_mapping,
    uninfected_mapping
):

    records = []

    # --------------------------------------------------------
    # Parasitized images
    # --------------------------------------------------------

    for split_name, patients in split.items():

        for patient_id in patients:

            filenames = parasitized_mapping.get(
                patient_id,
                []
            )

            for filename in filenames:

                records.append({
                    "filename": filename,
                    "patient_id": patient_id,
                    "class": "parasitized",
                    "split": split_name,
                    "relative_path":
                        f"cell_images/Parasitized/{filename}"
                })

    # --------------------------------------------------------
    # Uninfected images
    # --------------------------------------------------------

    for split_name, patients in split.items():

        for patient_id in patients:

            filenames = uninfected_mapping.get(
                patient_id,
                []
            )

            for filename in filenames:

                records.append({
                    "filename": filename,
                    "patient_id": patient_id,
                    "class": "uninfected",
                    "split": split_name,
                    "relative_path":
                        f"cell_images/Uninfected/{filename}"
                })

    return records


# ============================================================
# SAVE MASTER METADATA
# ============================================================

def save_master_metadata(records, output_file):

    fieldnames = [
        "filename",
        "patient_id",
        "class",
        "split",
        "relative_path"
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(records)


# ============================================================
# SAVE PATIENT SPLIT
# ============================================================

def save_patient_split(split, output_file):

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "patient_id",
            "split"
        ])

        for split_name in [
            "train",
            "validation",
            "test"
        ]:

            for patient_id in split[split_name]:

                writer.writerow([
                    patient_id,
                    split_name
                ])


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_statistics(statistics):

    print("\n" + "=" * 70)
    print("SELECTED BALANCED SPLIT")
    print("=" * 70)

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        stats = statistics[split_name]

        print(
            f"\n{split_name.upper()}"
        )

        print(
            f"Patients        : "
            f"{stats['patients']}"
        )

        print(
            f"Parasitized     : "
            f"{stats['parasitized']:,}"
        )

        print(
            f"Uninfected      : "
            f"{stats['uninfected']:,}"
        )

        print(
            f"Total images    : "
            f"{stats['total']:,}"
        )

        print(
            f"Patient ratio   : "
            f"{stats['patient_ratio'] * 100:.2f}%"
        )

        print(
            f"Parasitized % of all parasitized: "
            f"{stats['parasitized_ratio'] * 100:.2f}%"
        )

        print(
            f"Uninfected % of all uninfected: "
            f"{stats['uninfected_ratio'] * 100:.2f}%"
        )

        print(
            f"Image ratio     : "
            f"{stats['image_ratio'] * 100:.2f}%"
        )


# ============================================================
# LEAKAGE CHECK
# ============================================================

def check_leakage(split):

    patient_to_split = {}

    for split_name, patients in split.items():

        for patient_id in patients:

            if patient_id in patient_to_split:

                print(
                    f"LEAKAGE ERROR: "
                    f"{patient_id} appears in "
                    f"{patient_to_split[patient_id]} "
                    f"and {split_name}"
                )

                return False

            patient_to_split[patient_id] = split_name

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BALANCED PATIENT-LEVEL DATASET SPLIT")
    print("=" * 70)

    print("\nLoading patient mappings...")

    parasitized_mapping = load_mapping(
        PARASITIZED_CSV
    )

    uninfected_mapping = load_mapping(
        UNINFECTED_CSV
    )

    print(
        f"Parasitized patients : "
        f"{len(parasitized_mapping)}"
    )

    print(
        f"Uninfected patients  : "
        f"{len(uninfected_mapping)}"
    )

    # --------------------------------------------------------
    # Patient information
    # --------------------------------------------------------

    patient_info = build_patient_info(
        parasitized_mapping,
        uninfected_mapping
    )

    print(
        f"Unique patients      : "
        f"{len(patient_info)}"
    )

    both_classes = sum(
        1
        for info in patient_info.values()
        if info["parasitized"] > 0
        and info["uninfected"] > 0
    )

    uninfected_only = sum(
        1
        for info in patient_info.values()
        if info["parasitized"] == 0
        and info["uninfected"] > 0
    )

    print(
        f"Patients in both     : "
        f"{both_classes}"
    )

    print(
        f"Uninfected only      : "
        f"{uninfected_only}"
    )

    # --------------------------------------------------------
    # Search for best split
    # --------------------------------------------------------

    print("\nSearching for a balanced patient-level split...")

    patient_ids = list(patient_info.keys())

    best_split = None
    best_statistics = None
    best_score = float("inf")

    random.seed(RANDOM_SEED)

    for trial in range(N_TRIALS):

        candidate = create_random_split(
            patient_ids
        )

        statistics = calculate_statistics(
            candidate,
            patient_info
        )

        score = calculate_score(
            statistics
        )

        if score < best_score:

            best_score = score

            best_split = candidate

            best_statistics = statistics

    print(
        f"\nSearch completed: "
        f"{N_TRIALS:,} candidate splits evaluated."
    )

    print(
        f"Best score: "
        f"{best_score:.6f}"
    )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    print("\nChecking patient leakage...")

    if check_leakage(best_split):

        print(
            "PASS: No patient appears "
            "in multiple splits."
        )

    else:

        print(
            "FAIL: Patient leakage detected."
        )

        return

    # --------------------------------------------------------
    # Print selected split
    # --------------------------------------------------------

    print_statistics(
        best_statistics
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    patient_split_file = (
        OUTPUT_DIR /
        "patient_split_balanced.csv"
    )

    master_metadata_file = (
        OUTPUT_DIR /
        "master_metadata_patient_split_balanced.csv"
    )

    save_patient_split(
        best_split,
        patient_split_file
    )

    records = create_master_metadata(
        best_split,
        parasitized_mapping,
        uninfected_mapping
    )

    save_master_metadata(
        records,
        master_metadata_file
    )

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(
        f"\n{patient_split_file}"
    )

    print(
        f"{master_metadata_file}"
    )

    print("\n" + "=" * 70)
    print("BALANCED SPLIT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()