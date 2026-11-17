import pandas as pd
from sklearn.model_selection import train_test_split
import json

# 1. Load the dataset
df = pd.read_csv("YABATECH_Official_Policies_QA_600.csv")

# 2. Perform stratified split to maintain the ratio of simple/multi/unanswerable questions
# First split: 60% Train, 40% Temp (Val + Test)
train_df, temp_df = train_test_split(
    df, test_size=0.40, stratify=df['question_type'], random_state=42
)

# Second split: The 40% Temp is split into 15% Val and 25% Test (which is 37.5% and 62.5% of the temp dataframe)
val_df, test_df = train_test_split(
    temp_df, test_size=(0.25/0.40), stratify=temp_df['question_type'], random_state=42
)

# 3. Convert DataFrames to dictionary records
train_data = train_df.to_dict(orient="records")
val_data = val_df.to_dict(orient="records")
test_data = test_df.to_dict(orient="records")

# 4. Save to JSON files
with open("train.json", "w") as f:
    json.dump(train_data, f, indent=4)

with open("val.json", "w") as f:
    json.dump(val_data, f, indent=4)

with open("test.json", "w") as f:
    json.dump(test_data, f, indent=4)

print(f"Split complete! Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")