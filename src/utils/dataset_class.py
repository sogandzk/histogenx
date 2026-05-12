import torch
from torch.utils.data import Dataset

class Multi_Modal_Dataset(Dataset):
    def __init__(self, dataset, k_patches, n_views=1):
        assert k_patches == "all" or (isinstance(k_patches, int) and k_patches > 0)
        self.dataset = dataset
        self.k_patches = k_patches
        self.n_views = n_views

    def __len__(self):
        return len(self.dataset) * self.n_views

    def __getitem__(self, idx):
        patient_idx = idx % len(self.dataset)
        item = self.dataset[patient_idx]

        # -----------------------------------------------------------------------------------------
        # image features
        # make the tensor of all patch-level features
        all_feats = torch.as_tensor(item["resnet50_features"], dtype=torch.float32)
        
        # Mean-pool patch features: 
        # --- use all patches for stable validation, 
        # --- or a random subset of K for training augmentation.
        if self.k_patches == "all":
            img_features = torch.mean(all_feats, dim=0)
        else:
            indices = torch.randperm(len(all_feats))[:self.k_patches]
            img_features = torch.mean(all_feats[indices], dim=0)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        # gene expression features
        expr_features = torch.as_tensor(item["gene_expression_features"], dtype=torch.float32)
        # -----------------------------------------------------------------------------------------

        pid = item['patient_id']        

        return img_features, expr_features, pid
    