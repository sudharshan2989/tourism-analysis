import pandas as pd
import os
from huggingface_hub import login
from huggingface_hub import hf_hub_download

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import make_pipeline

# Define repo and filenames
repo_id = "sudharshanc/tourism-analysis"

csv_path = hf_hub_download(repo_id=repo_id,repo_type="dataset", filename="tourism.csv")
# Read the CSV into a DataFrame
df_hf_main = pd.read_csv(csv_path)

HF_TOKEN = os.getenv("HF_TOKEN")
# Initialize API client
api = HfApi(token=HF_TOKEN)

## Lets drop the ids and separate features and target
X = df_hf_main.drop(columns=["ProdTaken", "CustomerID", "Unnamed: 0"])
y = df_hf_main["ProdTaken"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
## Startified sampling ensures that the class distribution in the train and test sets is similar to the original dataset.

print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Class distribution in train:", y_train.value_counts())
print("Class distribution in test:", y_test.value_counts())

## Uploading the train and test files to Hugging Face Hub
X_train.to_csv("Xtrain.csv",index=False)
X_test.to_csv("Xtest.csv",index=False)
y_train.to_csv("ytrain.csv",index=False)
y_test.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="sudharshanc/tourism-analysis",
        repo_type="dataset",
    )
