from pathlib import Path
import ast

PROJECT_ROOT = Path(__file__).resolve().parent

PARASITIZED_CSV = PROJECT_ROOT / "patientid_cellmapping_parasitized.csv"
UNINFECTED_CSV = PROJECT_ROOT / "patientid_cellmapping_uninfected.csv"


def load_mapping(csv_path):

    mapping = {}

    print(f"\nReading: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                patient_id, filenames_text = line.split(",", 1)
            except ValueError:
                print("Could not split line:", line_number)
                continue

            patient_id = patient_id.strip()

            try:
                filenames = ast.literal_eval(filenames_text)
            except (ValueError, SyntaxError) as e:
                print("Could not parse line:", line_number, e)
                continue

            if isinstance(filenames, list):
                mapping[patient_id] = filenames

    return mapping


p = load_mapping(PARASITIZED_CSV)
u = load_mapping(UNINFECTED_CSV)

print("\nRESULT")
print("=" * 50)

print("Parasitized patients:", len(p))
print("Uninfected patients :", len(u))

print("\nFirst 5 parasitized patient IDs:")
print(list(p.keys())[:5])

print("\nFirst 5 uninfected patient IDs:")
print(list(u.keys())[:5])

both = set(p) & set(u)

print("\nPatients in both:", len(both))
print("Total unique patients:", len(set(p) | set(u)))
