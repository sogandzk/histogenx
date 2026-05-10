import torch
from torch.utils.data import Dataset

class Multi_Modal_Dataset(Dataset):
    def __init__(self, dataset, k_patches):
        assert k_patches == "all" or (isinstance(k_patches, int) and k_patches > 0)
        self.dataset = dataset
        self.k_patches = k_patches

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # patient_idx = idx % len(self.dataset)
        item = self.dataset[idx]

        # -----------------------------------------------------------------------------------------
        # image features
        # make the tensor of all patch image features
        img_features = torch.as_tensor(item["resnet50_features"], dtype=torch.float32)
        # if k_patches is a number, take a random subset
        if isinstance(self.k_patches, int):
            indices = torch.randperm(len(img_features))[:self.k_patches]
            img_features = img_features[indices]
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        # gene expression features
        expr_features = torch.as_tensor(item["gene_expression_features"], dtype=torch.float32)
        # -----------------------------------------------------------------------------------------

        pid = item['patient_id']     

        return img_features, expr_features, pid
    
    def collate_fn(batch):
        # 1. Extract individual components from the batch
        # img_features_list will be a list of tensors: [ [K1, 1843], [K2, 1843], ... ]
        img_features_list = [item[0] for item in batch]
        expr_features_list = [item[1] for item in batch]
        pids = [item[2] for item in batch]

        # 2. Create the 'ks' list to track how many patches belong to each patient
        ks = [img.shape[0] for img in img_features_list]

        # 3. Stack image features into [sum(K_n), :]
        # We use torch.cat instead of torch.stack because dimensions vary
        img_features = torch.cat(img_features_list, dim=0)

        # 4. Stack gene expression into [N, :]
        # Since these are all of the size, we can use torch.stack
        expr_features = torch.stack(expr_features_list, dim=0)

        return img_features, ks, expr_features, pids

