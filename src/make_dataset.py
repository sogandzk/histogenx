import os 
import pandas as pd 
import numpy as np
import torch 
from torchvision import models 
from torchvision.models import ResNet50_Weights 
from torchvision import transforms 
from datasets import load_dataset, Dataset, DatasetDict
from utils.config import PROJECT_DIR
import subprocess


# ------------------------------------------------------------------------------------------------------------------
print("Load the main dataset initially containing images")

ds = load_dataset("dakomura/tcga-ut", "external")
ds_train = ds["train"].filter(lambda x: x["json"]["label"] == "Breast_invasive_carcinoma" and x["__key__"][13:15] == '01' )
ds_valid = ds["valid"].filter(lambda x: x["json"]["label"] == "Breast_invasive_carcinoma" and x["__key__"][13:15] == '01')
ds_test = ds["test"].filter(lambda x: x["json"]["label"] == "Breast_invasive_carcinoma" and x["__key__"][13:15] == '01')
# ------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------
print("Map ResNet50 features to the dataset at the level of individual images")

# Check for CUDA (NVIDIA), then MPS (Apple), then fallback to CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load the Backbone
backbone_pth = os.path.join(PROJECT_DIR, "data", "resnet50_pretrained.pth")
if os.path.exists(backbone_pth):
    state_dict = torch.load(backbone_pth, map_location="cpu")
    backbone = models.resnet50(weights=None)
    backbone.load_state_dict(state_dict)
else:
    backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    torch.save(backbone.state_dict(), backbone_pth)

backbone.fc = torch.nn.Identity()
backbone = backbone.to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),       
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def add_resnet50_features(batch):
    imgs = [preprocess(img) for img in batch["jpg"]]
    imgs_tensor = torch.stack(imgs).to(device)
    with torch.no_grad():
        features = backbone(imgs_tensor).cpu().tolist()
    return {"resnet50_features": features}

ds_train = ds_train.map(add_resnet50_features, batched=True, batch_size=64)
ds_valid = ds_valid.map(add_resnet50_features, batched=True, batch_size=64)
ds_test = ds_test.map(add_resnet50_features, batched=True, batch_size=64)
# ------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------
print("Reshape the dataset")
# One data point per patient with one nested list (list of lists) for resnet50_features

def reshape_ds(ds):
    cols = ["__key__", "resnet50_features"]
    df = ds.select_columns(cols).to_pandas()
    df['patient_id'] = df['__key__'].str[:15]
    df = df.groupby("patient_id").agg({ "resnet50_features": lambda x: x.tolist() }).reset_index()
    return Dataset.from_pandas(df)

ds_train = reshape_ds(ds_train)
ds_valid = reshape_ds(ds_valid)
ds_test = reshape_ds(ds_test)

# remove the very few patients that have 10 image patches (or less)
ds_train = ds_train.filter(lambda x: len(x["resnet50_features"]) > 10)
ds_valid = ds_valid.filter(lambda x: len(x["resnet50_features"]) > 10)
ds_test = ds_test.filter(lambda x: len(x["resnet50_features"]) > 10)

# now assert that every patient must have at least 30 image patches
assert 30 == min([len(d["resnet50_features"]) for d in ds_train])
assert 30 == min([len(d["resnet50_features"]) for d in ds_valid])
assert 30 == min([len(d["resnet50_features"]) for d in ds_test])
# ------------------------------------------------------------------------------------------------------------------







# ------------------------------------------------------------------------------------------------------------------
print("Filter for top 90% most variable features based on training data")

# 1. Extract and flatten training features to compute global variance per feature
# Each row in ds_train is a patient with a list of patches (each patch is a 2048-dim vector)
train_features_nested = ds_train["resnet50_features"]
train_features_flat = np.concatenate(train_features_nested, axis=0) # Shape: (Total_Patches, 2048)

# 2. Calculate standard deviation for each of the 2048 features
feature_stds = np.std(train_features_flat, axis=0)

# 3. Determine the threshold for the top 90% (keep 1843 features out of 2048)
threshold = np.percentile(feature_stds, 10)
keep_indices = np.where(feature_stds >= threshold)[0]

print(f"Keeping {len(keep_indices)} features with std >= {threshold:.4f}")

# 4. Apply the subsetting to train, valid, and test
def subset_features(batch):
    # batch["resnet50_features"] is a list of lists of lists
    # Structure: [Patient][Patch][Feature]
    new_features = []
    for patient_patches in batch["resnet50_features"]:
        # Convert patient's patches to numpy for efficient indexing, then back to list
        patient_array = np.array(patient_patches)
        subset_array = patient_array[:, keep_indices]
        new_features.append(subset_array.tolist())
    return {"resnet50_features": new_features}

ds_train = ds_train.map(subset_features, batched=True)
ds_valid = ds_valid.map(subset_features, batched=True)
ds_test = ds_test.map(subset_features, batched=True)

# Verify the new feature dimension
assert len(ds_train[0]["resnet50_features"][0]) == len(keep_indices)
# ------------------------------------------------------------------------------------------------------------------






# ------------------------------------------------------------------------------------------------------------------
print("Preprocess gene expression data")

expr_df = pd.read_csv(os.path.join(PROJECT_DIR, "data", "data_mrna_seq_v2_rsem.txt"), sep="\t")
assert expr_df.shape[1] == len(set(expr_df.columns))
expr_df.insert(0, 'row_index', range(len(expr_df)))

# Select top 2000 most variable genes 
# Note that the variance calculation is done within the train data
columns_in_train = list(set(expr_df.columns) & set(ds_train["patient_id"]))
variances = expr_df[columns_in_train].var(axis=1)
index_top2000_genes = variances.sort_values(ascending=False).head(2000).index
expr_df_2000 = expr_df.loc[index_top2000_genes].filter(regex="^TCGA-")

# log1p-transform all expression data
norm_expr_df_2000 = np.log1p(expr_df_2000)
# ------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------
print("Filter the dataset to exclude the patients with no matching gene expression data")

ds_train = ds_train.filter(lambda x: x["patient_id"] in norm_expr_df_2000.columns)
ds_valid = ds_valid.filter(lambda x: x["patient_id"] in norm_expr_df_2000.columns)
ds_test = ds_test.filter(lambda x: x["patient_id"] in norm_expr_df_2000.columns)
# ------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------
print("Map gene expression vectors to the image dataset at the level of patient IDs")

def mapper(x): return {"gene_expression_features": norm_expr_df_2000[x["patient_id"]].astype("float32").tolist()}
ds_train = ds_train.map(mapper)
ds_valid = ds_valid.map(mapper)
ds_test = ds_test.map(mapper)
# ------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------
print("Store the 'dataset' folder and make the tarball")

ds_on_disk = DatasetDict({
    "train": ds_train,
    "valid": ds_valid,
    "test":  ds_test,
})

pth = os.path.join(PROJECT_DIR, "data", "dataset")
ds_on_disk.save_to_disk(pth)

parent_dir = os.path.dirname(pth)
target_folder = os.path.basename(pth)
cmd = ["tar", "--disable-copyfile", "--no-xattrs", "-czvf", target_folder+".tar.gz", target_folder]
subprocess.run(cmd, cwd=parent_dir, check=True)
# ------------------------------------------------------------------------------------------------------------------
