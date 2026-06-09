import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

INPUT_FILE = "API Functions.csv"
OUTPUT_FILE = "API_Functions_flat.csv"

print("Reading CSV...")
df = pd.read_csv(INPUT_FILE, header=None, index_col=0, low_memory=False)
df.index.name = "sha256"

# Column 1 is the malware type label; columns 2+ are API function names
label_col = df.iloc[:, 0]
api_cols = df.iloc[:, 1:]

print(f"Loaded {len(df)} rows. Parsing API sets per row...")

EXCLUDED = {"NA", ""}

def row_to_api_set(row):
    return {v for v in row.dropna().astype(str) if v not in EXCLUDED}

api_sets = api_cols.apply(row_to_api_set, axis=1)

print("Building binary API presence matrix...")
mlb = MultiLabelBinarizer(sparse_output=False)
api_matrix = pd.DataFrame(
    mlb.fit_transform(api_sets),
    index=df.index,
    columns=mlb.classes_,
    dtype="int8",
)
print(f"API matrix shape: {api_matrix.shape}")

print("Building binary label matrix...")
label_matrix = pd.get_dummies(label_col.rename("label"), dtype="int8")
label_matrix.index = df.index

print(f"Label columns: {list(label_matrix.columns)}")

print("Concatenating and saving...")
result = pd.concat([api_matrix, label_matrix], axis=1)
print(f"Final dataset shape: {result.shape}")

result.to_csv(OUTPUT_FILE)
print(f"Saved to {OUTPUT_FILE}")
