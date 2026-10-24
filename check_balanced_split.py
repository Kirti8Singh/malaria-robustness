from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results" / "dataset"

MASTER_FILE = (
    RESULTS_DIR /
    "master_metadata_patient_split_balanced.csv"
)

PATIENT_SPLIT_FILE = (
    RESULTS_DIR /
    "patient_split_balanced.csv"
)


print("=" * 70)
print("BALANCED PATIENT-LEVEL SPLIT CHECK")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

print("\nLoading metadata...")

df = pd.read_csv(MASTER_FILE)
patient_split = pd.read_csv(PATIENT_SPLIT_FILE)

print(f"Master metadata rows : {len(df):,}")
print(f"Patient split rows   : {len(patient_split):,}")


# ============================================================
# PATIENT COUNTS
# ============================================================

print("\n" + "=" * 70)
print("CHECK 1 — PATIENT COUNTS")
print("=" * 70)

print(
    patient_split["split"]
    .value_counts()
    .sort_index()
)


# ============================================================
# IMAGE COUNTS
# ============================================================

print("\n" + "=" * 70)
print("CHECK 2 — IMAGE COUNTS")
print("=" * 70)

image_counts = pd.crosstab(
    df["split"],
    df["class"]
)

print("\nImages by split and class:")
print(image_counts)


# ============================================================
# CLASS BALANCE
# ============================================================

print("\n" + "=" * 70)
print("CHECK 3 — CLASS BALANCE")
print("=" * 70)

for split in [
    "train",
    "validation",
    "test"
]:

    subset = df[df["split"] == split]

    parasitized = (
        subset["class"] == "parasitized"
    ).sum()

    uninfected = (
        subset["class"] == "uninfected"
    ).sum()

    total = len(subset)

    print(f"\n{split.upper()}")

    print(
        f"Parasitized : {parasitized:,}"
    )

    print(
        f"Uninfected  : {uninfected:,}"
    )

    print(
        f"Parasitized %: "
        f"{parasitized / total * 100:.2f}%"
    )

    print(
        f"Uninfected % : "
        f"{uninfected / total * 100:.2f}%"
    )


# ============================================================
# PATIENT LEAKAGE
# ============================================================

print("\n" + "=" * 70)
print("CHECK 4 — PATIENT LEAKAGE")
print("=" * 70)

patient_split_counts = (
    df.groupby("patient_id")["split"]
    .nunique()
)

leaked = patient_split_counts[
    patient_split_counts > 1
]

if len(leaked) == 0:

    print(
        "PASS: No patient appears "
        "in multiple splits."
    )

else:

    print(
        "FAIL: Patient leakage detected!"
    )

    print(leaked)


# ============================================================
# DUPLICATE FILENAMES
# ============================================================

print("\n" + "=" * 70)
print("CHECK 5 — DUPLICATE FILENAMES")
print("=" * 70)

duplicates = df[
    df.duplicated(
        subset=["filename", "class"],
        keep=False
    )
]

if len(duplicates) == 0:

    print(
        "PASS: No duplicate filename/class "
        "entries found."
    )

else:

    print(
        f"WARNING: {len(duplicates)} "
        f"duplicate entries found."
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("BALANCED SPLIT CHECK COMPLETED")
print("=" * 70)