from pathlib import Path
from collections import Counter, defaultdict
import csv
import ast
import hashlib
import json

from PIL import Image


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_ROOT = PROJECT_ROOT / "cell_images"

PARASITIZED_DIR = IMAGE_ROOT / "Parasitized"
UNINFECTED_DIR = IMAGE_ROOT / "Uninfected"

PARASITIZED_CSV = PROJECT_ROOT / "patientid_cellmapping_parasitized.csv"
UNINFECTED_CSV = PROJECT_ROOT / "patientid_cellmapping_uninfected.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "dataset"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

VALID_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_image_files(folder):
    """
    Return actual image files only.
    Files such as thumbs.db are ignored.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    return sorted(
        [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
    )


def sha256_file(path, chunk_size=1024 * 1024):
    """
    Calculate SHA-256 hash of a file.
    Used to detect exact duplicate files.
    """
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def extract_patient_id(filename):
    """
    Extract patient ID from filenames such as:

    C33P1thinF_IMG_20150619_114756_cell_179.png

    Patient ID = C33P1thinF
    """

    if "_IMG_" in filename:
        return filename.split("_IMG_")[0]

    return None


def read_mapping_csv(csv_path):
    """
    Read the NLM patient-to-cell mapping CSV.

    Each row has the form:

        PatientID,['image1.png', 'image2.png', ...],,,,,,

    The image list occupies the beginning of the row and is followed
    by many empty CSV fields. We therefore extract only the portion
    containing the Python-style list.
    """

    mapping = {}
    problems = []

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:

        for line_no, raw_line in enumerate(f, start=1):

            line = raw_line.strip()

            if not line:
                continue

            # Split only at the first comma.
            parts = line.split(",", 1)

            if len(parts) != 2:
                problems.append({
                    "line": line_no,
                    "problem": "Could not split patient ID from image list"
                })
                continue

            patient_id = parts[0].strip()

            remainder = parts[1].strip()

            # Find the closing bracket of the image list.
            closing_bracket = remainder.rfind("]")

            if closing_bracket == -1:
                problems.append({
                    "line": line_no,
                    "patient_id": patient_id,
                    "problem": "No closing ] found"
                })
                continue

            # Keep only:
            # ['image1.png', 'image2.png', ...]
            image_list_text = remainder[:closing_bracket + 1]

            try:
                image_list = ast.literal_eval(image_list_text)

            except Exception as e:
                problems.append({
                    "line": line_no,
                    "patient_id": patient_id,
                    "problem": f"Could not parse image list: {e}"
                })
                continue

            if not isinstance(image_list, list):
                problems.append({
                    "line": line_no,
                    "patient_id": patient_id,
                    "problem": "Parsed object is not a list"
                })
                continue

            mapping[patient_id] = image_list

    return mapping, problems


def write_csv(path, rows, fieldnames):
    """
    Write a list of dictionaries to CSV.
    """

    with open(path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 4. LOAD PATIENT MAPPINGS
# ============================================================

print("\n" + "=" * 70)
print("MALARIA DATASET AUDIT")
print("=" * 70)

print("\nLoading patient mapping CSV files...")

parasitized_mapping, parasitized_csv_problems = read_mapping_csv(
    PARASITIZED_CSV
)

uninfected_mapping, uninfected_csv_problems = read_mapping_csv(
    UNINFECTED_CSV
)

print(f"Parasitized patient IDs : {len(parasitized_mapping)}")
print(f"Uninfected patient IDs  : {len(uninfected_mapping)}")

# ============================================================
# 5. COLLECT IMAGE FILES
# ============================================================

print("\nCollecting image files...")

parasitized_files = get_image_files(PARASITIZED_DIR)
uninfected_files = get_image_files(UNINFECTED_DIR)

all_files = (
    [(p, "parasitized") for p in parasitized_files]
    +
    [(p, "uninfected") for p in uninfected_files]
)

print(
    f"Parasitized image files : {len(parasitized_files)}"
)

print(
    f"Uninfected image files  : {len(uninfected_files)}"
)

print(
    f"Total image files       : {len(all_files)}"
)


# ============================================================
# 6. CHECK NON-IMAGE FILES
# ============================================================

print("\nChecking non-image files...")

non_image_files = []

for folder in [PARASITIZED_DIR, UNINFECTED_DIR]:

    for p in folder.iterdir():

        if p.is_file() and p.suffix.lower() not in VALID_EXTENSIONS:

            non_image_files.append(
                str(p.relative_to(PROJECT_ROOT))
            )

print(
    f"Non-image files found : {len(non_image_files)}"
)

if non_image_files:

    print("Examples:")

    for item in non_image_files[:10]:
        print("  ", item)


with open(
    RESULTS_DIR / "non_image_files.txt",
    "w",
    encoding="utf-8"
) as f:

    for item in non_image_files:
        f.write(item + "\n")


# ============================================================
# 7. IMAGE PROPERTIES + CORRUPTION CHECK
# ============================================================

print("\nInspecting images...")
print("This may take a few minutes for 27,558 images.\n")

image_rows = []
corrupted_images = []

dimension_counter = Counter()
mode_counter = Counter()
extension_counter = Counter()

hash_to_files = defaultdict(list)

for index, (path, label) in enumerate(all_files, start=1):

    if index % 1000 == 0:
        print(f"Processed {index}/{len(all_files)} images...")

    relative_path = str(
        path.relative_to(PROJECT_ROOT)
    )

    extension = path.suffix.lower()

    extension_counter[extension] += 1

    patient_id = extract_patient_id(path.name)

    try:

        # Open image
        with Image.open(path) as img:

            # Force Pillow to actually read the image
            img.load()

            width, height = img.size
            mode = img.mode

        dimensions = f"{width}x{height}"

        dimension_counter[dimensions] += 1
        mode_counter[mode] += 1

        # Exact file hash
        file_hash = sha256_file(path)

        hash_to_files[file_hash].append(
            relative_path
        )

        image_rows.append({
            "filename": path.name,
            "relative_path": relative_path,
            "class": label,
            "patient_id_from_filename": patient_id,
            "extension": extension,
            "width": width,
            "height": height,
            "mode": mode,
            "sha256": file_hash,
            "status": "OK"
        })

    except Exception as e:

        corrupted_images.append({
            "filename": path.name,
            "relative_path": relative_path,
            "class": label,
            "error": str(e)
        })

        image_rows.append({
            "filename": path.name,
            "relative_path": relative_path,
            "class": label,
            "patient_id_from_filename": patient_id,
            "extension": extension,
            "width": "",
            "height": "",
            "mode": "",
            "sha256": "",
            "status": "CORRUPTED"
        })


# ============================================================
# 8. SAVE IMAGE PROPERTIES
# ============================================================

write_csv(
    RESULTS_DIR / "image_properties.csv",
    image_rows,
    [
        "filename",
        "relative_path",
        "class",
        "patient_id_from_filename",
        "extension",
        "width",
        "height",
        "mode",
        "sha256",
        "status"
    ]
)


# ============================================================
# 9. EXACT DUPLICATE CHECK
# ============================================================

duplicate_groups = {
    file_hash: paths
    for file_hash, paths in hash_to_files.items()
    if len(paths) > 1
}

duplicate_rows = []

for file_hash, paths in duplicate_groups.items():

    for path in paths:

        duplicate_rows.append({
            "sha256": file_hash,
            "duplicate_group_size": len(paths),
            "relative_path": path
        })


write_csv(
    RESULTS_DIR / "exact_duplicate_report.csv",
    duplicate_rows,
    [
        "sha256",
        "duplicate_group_size",
        "relative_path"
    ]
)


# ============================================================
# 10. CHECK FILENAME DUPLICATES
# ============================================================

filename_to_paths = defaultdict(list)

for row in image_rows:

    filename_to_paths[row["filename"]].append(
        row["relative_path"]
    )

duplicate_filenames = {
    filename: paths
    for filename, paths in filename_to_paths.items()
    if len(paths) > 1
}


duplicate_filename_rows = []

for filename, paths in duplicate_filenames.items():

    for path in paths:

        duplicate_filename_rows.append({
            "filename": filename,
            "relative_path": path,
            "duplicate_count": len(paths)
        })


write_csv(
    RESULTS_DIR / "duplicate_filename_report.csv",
    duplicate_filename_rows,
    [
        "filename",
        "relative_path",
        "duplicate_count"
    ]
)


# ============================================================
# 11. CSV ↔ IMAGE MATCHING
# ============================================================

print("\nChecking CSV ↔ image consistency...")

actual_parasitized_names = {
    p.name for p in parasitized_files
}

actual_uninfected_names = {
    p.name for p in uninfected_files
}


def mapping_to_filename_set(mapping):

    result = set()

    for filenames in mapping.values():

        result.update(filenames)

    return result


csv_parasitized_names = mapping_to_filename_set(
    parasitized_mapping
)

csv_uninfected_names = mapping_to_filename_set(
    uninfected_mapping
)


parasitized_missing_from_folder = (
    csv_parasitized_names
    -
    actual_parasitized_names
)

parasitized_not_in_csv = (
    actual_parasitized_names
    -
    csv_parasitized_names
)

uninfected_missing_from_folder = (
    csv_uninfected_names
    -
    actual_uninfected_names
)

uninfected_not_in_csv = (
    actual_uninfected_names
    -
    csv_uninfected_names
)


# ============================================================
# 12. PATIENT ID CONSISTENCY
# ============================================================

patient_mismatch_rows = []

for row in image_rows:

    filename = row["filename"]

    patient_from_filename = row[
        "patient_id_from_filename"
    ]

    if patient_from_filename is None:
        patient_mismatch_rows.append({
            "filename": filename,
            "class": row["class"],
            "problem": "Could not extract patient ID"
        })


write_csv(
    RESULTS_DIR / "patient_id_problems.csv",
    patient_mismatch_rows,
    [
        "filename",
        "class",
        "problem"
    ]
)


# ============================================================
# 13. OVERALL PATIENT INFORMATION
# ============================================================

all_patient_ids = (
    set(parasitized_mapping.keys())
    |
    set(uninfected_mapping.keys())
)

overlap_patient_ids = (
    set(parasitized_mapping.keys())
    &
    set(uninfected_mapping.keys())
)

parasitized_only_patients = (
    set(parasitized_mapping.keys())
    -
    set(uninfected_mapping.keys())
)

uninfected_only_patients = (
    set(uninfected_mapping.keys())
    -
    set(parasitized_mapping.keys())
)


# ============================================================
# 14. PATIENT-LEVEL IMAGE COUNTS
# ============================================================

patient_rows = []

for patient_id in sorted(all_patient_ids):

    parasitized_count = len(
        parasitized_mapping.get(patient_id, [])
    )

    uninfected_count = len(
        uninfected_mapping.get(patient_id, [])
    )

    patient_rows.append({
        "patient_id": patient_id,
        "parasitized_images": parasitized_count,
        "uninfected_images": uninfected_count,
        "total_images": (
            parasitized_count +
            uninfected_count
        ),
        "has_parasitized": parasitized_count > 0,
        "has_uninfected": uninfected_count > 0
    })


write_csv(
    RESULTS_DIR / "patient_summary.csv",
    patient_rows,
    [
        "patient_id",
        "parasitized_images",
        "uninfected_images",
        "total_images",
        "has_parasitized",
        "has_uninfected"
    ]
)


# ============================================================
# 15. MASTER METADATA TABLE
# ============================================================

master_rows = []

for row in image_rows:

    master_rows.append({
        "filename": row["filename"],
        "relative_path": row["relative_path"],
        "class": row["class"],
        "patient_id": row["patient_id_from_filename"],
        "extension": row["extension"],
        "width": row["width"],
        "height": row["height"],
        "mode": row["mode"],
        "sha256": row["sha256"],
        "status": row["status"],
        "split": ""
    })


write_csv(
    RESULTS_DIR / "master_metadata.csv",
    master_rows,
    [
        "filename",
        "relative_path",
        "class",
        "patient_id",
        "extension",
        "width",
        "height",
        "mode",
        "sha256",
        "status",
        "split"
    ]
)


# ============================================================
# 16. SUMMARY REPORT
# ============================================================

valid_images = [
    row for row in image_rows
    if row["status"] == "OK"
]

summary = {

    "dataset": {
        "total_images": len(all_files),
        "parasitized_images": len(parasitized_files),
        "uninfected_images": len(uninfected_files),
        "corrupted_images": len(corrupted_images),
    },

    "non_image_files": {
        "count": len(non_image_files),
        "files": non_image_files
    },

    "image_extensions": dict(extension_counter),

    "image_dimensions": dict(dimension_counter),

    "image_modes": dict(mode_counter),

    "duplicates": {
        "exact_duplicate_groups": len(duplicate_groups),
        "files_in_duplicate_groups": sum(
            len(paths)
            for paths in duplicate_groups.values()
        ),
        "duplicate_filenames": len(
            duplicate_filenames
        )
    },

    "patients": {
        "unique_patients_overall": len(
            all_patient_ids
        ),
        "parasitized_patient_ids": len(
            parasitized_mapping
        ),
        "uninfected_patient_ids": len(
            uninfected_mapping
        ),
        "patients_present_in_both_classes": len(
            overlap_patient_ids
        ),
        "parasitized_only_patients": len(
            parasitized_only_patients
        ),
        "uninfected_only_patients": len(
            uninfected_only_patients
        )
    },

    "csv_matching": {

        "parasitized_csv_images": len(
            csv_parasitized_names
        ),

        "uninfected_csv_images": len(
            csv_uninfected_names
        ),

        "parasitized_missing_from_folder": len(
            parasitized_missing_from_folder
        ),

        "parasitized_not_in_csv": len(
            parasitized_not_in_csv
        ),

        "uninfected_missing_from_folder": len(
            uninfected_missing_from_folder
        ),

        "uninfected_not_in_csv": len(
            uninfected_not_in_csv
        )
    }
}


with open(
    RESULTS_DIR / "dataset_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# 17. PRINT FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("DATASET AUDIT COMPLETE")
print("=" * 70)

print("\nIMAGE COUNTS")
print("-" * 70)

print(
    f"Parasitized images : {len(parasitized_files):,}"
)

print(
    f"Uninfected images  : {len(uninfected_files):,}"
)

print(
    f"Total images       : {len(all_files):,}"
)

print("\nIMAGE QUALITY")
print("-" * 70)

print(
    f"Corrupted images   : {len(corrupted_images):,}"
)

print(
    f"Image dimensions   : {dict(dimension_counter)}"
)

print(
    f"Image modes        : {dict(mode_counter)}"
)

print(
    f"Extensions         : {dict(extension_counter)}"
)

print("\nDUPLICATES")
print("-" * 70)

print(
    f"Exact duplicate groups : {len(duplicate_groups):,}"
)

print(
    f"Files in duplicate groups : "
    f"{sum(len(paths) for paths in duplicate_groups.values()):,}"
)

print(
    f"Duplicate filenames : {len(duplicate_filenames):,}"
)

print("\nPATIENT INFORMATION")
print("-" * 70)

print(
    f"Unique patients overall : "
    f"{len(all_patient_ids):,}"
)

print(
    f"Parasitized patients : "
    f"{len(parasitized_mapping):,}"
)

print(
    f"Uninfected patients : "
    f"{len(uninfected_mapping):,}"
)

print(
    f"Patients in BOTH classes : "
    f"{len(overlap_patient_ids):,}"
)

print("\nCSV ↔ IMAGE MATCHING")
print("-" * 70)

print(
    f"Parasitized CSV filenames : "
    f"{len(csv_parasitized_names):,}"
)

print(
    f"Parasitized missing from folder : "
    f"{len(parasitized_missing_from_folder):,}"
)

print(
    f"Parasitized not in CSV : "
    f"{len(parasitized_not_in_csv):,}"
)

print(
    f"Uninfected CSV filenames : "
    f"{len(csv_uninfected_names):,}"
)

print(
    f"Uninfected missing from folder : "
    f"{len(uninfected_missing_from_folder):,}"
)

print(
    f"Uninfected not in CSV : "
    f"{len(uninfected_not_in_csv):,}"
)

print("\nRESULT FILES")
print("-" * 70)

print(
    f"Saved in: {RESULTS_DIR}"
)

print("\nAudit finished successfully.")