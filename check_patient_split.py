from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results" / "dataset"

MASTER_FILE = RESULTS_DIR / "master_metadata_patient_split.csv"
PATIENT_SPLIT_FILE = RESULTS_DIR / "patient_split.csv"


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 70)
print("PATIENT-LEVEL SPLIT CHECK")
print("=" * 70)

print("\nLoading metadata...")

df = pd.read_csv(MASTER_FILE)
patient_split = pd.read_csv(PATIENT_SPLIT_FILE)

print(f"Master metadata rows : {len(df):,}")
print(f"Patient split rows   : {len(patient_split):,}")


# ============================================================
# CHECK 1 — PATIENT COUNTS
# ============================================================

print("\n" + "=" * 70)
print("CHECK 1 — PATIENT COUNTS")
print("=" * 70)

print("\nPatients per split:")

print(
    patient_split["split"]
    .value_counts()
    .sort_index()
)


# ============================================================
# CHECK 2 — IMAGE COUNTS BY SPLIT AND CLASS
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

print("\nPercent of total images:")

total_images = len(df)

for split in ["train", "validation", "test"]:

    count = len(df[df["split"] == split])

    percentage = (count / total_images) * 100

    print(
        f"{split:12s}: "
        f"{count:6,} images "
        f"({percentage:.2f}%)"
    )


# ============================================================
# CHECK 3 — CLASS BALANCE
# ============================================================

print("\n" + "=" * 70)
print("CHECK 3 — CLASS BALANCE")
print("=" * 70)

for split in ["train", "validation", "test"]:

    subset = df[df["split"] == split]

    parasitized = (subset["class"] == "parasitized").sum()
    uninfected = (subset["class"] == "uninfected").sum()

    total = len(subset)

    print(f"\n{split.upper()}")

    print(f"Parasitized : {parasitized:,}")
    print(f"Uninfected  : {uninfected:,}")

    if total > 0:

        p_percent = parasitized / total * 100
        u_percent = uninfected / total * 100

        print(f"Parasitized %: {p_percent:.2f}%")
        print(f"Uninfected % : {u_percent:.2f}%")


# ============================================================
# CHECK 4 — CELLS PER PATIENT
# ============================================================

print("\n" + "=" * 70)
print("CHECK 4 — CELLS PER PATIENT")
print("=" * 70)

patient_counts = (
    df.groupby(["patient_id", "split", "class"])
      .size()
      .reset_index(name="image_count")
)

print("\nNumber of patients in each split:")

for split in ["train", "validation", "test"]:

    subset = patient_counts[
        patient_counts["split"] == split
    ]

    print(f"\n{split.upper()}")

    print(
        f"Patients: "
        f"{subset['patient_id'].nunique()}"
    )

    print(
        f"Minimum cells per patient: "
        f"{subset['image_count'].min()}"
    )

    print(
        f"Maximum cells per patient: "
        f"{subset['image_count'].max()}"
    )

    print(
        f"Average cells per patient: "
        f"{subset['image_count'].mean():.2f}"
    )


# ============================================================
# CHECK 5 — PATIENT LEAKAGE
# ============================================================

print("\n" + "=" * 70)
print("CHECK 5 — PATIENT LEAKAGE")
print("=" * 70)

patient_split_counts = (
    df.groupby("patient_id")["split"]
      .nunique()
)

leaked_patients = patient_split_counts[
    patient_split_counts > 1
]

if len(leaked_patients) == 0:

    print("PASS: No patient appears in multiple splits.")

else:

    print("FAIL: Patient leakage detected!")

    print(leaked_patients)


# ============================================================
# PATIENT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PATIENT SUMMARY")
print("=" * 70)

summary = (
    df.groupby(["split", "patient_id"])
      .agg(
          total_images=("filename", "count"),
          parasitized=("class", lambda x:
                       (x == "parasitized").sum()),
          uninfected=("class", lambda x:
                      (x == "uninfected").sum())
      )
      .reset_index()
)

summary_file = RESULTS_DIR / "patient_split_detailed_summary.csv"

summary.to_csv(summary_file, index=False)

print(f"\nDetailed patient summary saved to:")
print(summary_file)

print("\n" + "=" * 70)
print("CHECK COMPLETED")
print("=" * 70)