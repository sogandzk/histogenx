import torch
import torch.nn.functional as F

def contrastive_loss(z_v, ks, z_g, temperature):
    """
    z_v: Tensor of shape (sum(ks), d) -- visual embeddings (patches)
    ks: list of [k1, k2, ..., kn] -- number of patches per sample
    z_g: Tensor of shape (N, d) -- genomic embeddings
    temperature: temperature scalar
    """
    # 1. Assert that the number of rows in z_v matches the sum of ks
    assert z_v.size(0) == sum(ks), f"Expected {sum(ks)} rows in z_v, but got {z_v.size(0)}"

    # 2. Mean pooling over patches per sample
    # Split z_v into a list of tensors with sizes k1, k2, ..., kn
    z_v_splits = torch.split(z_v, ks, dim=0)
    
    # Calculate mean for each split and stack into (N, d)
    z_v_pooled = torch.stack([split.mean(dim=0) for split in z_v_splits])

    # 3. Assert that the shapes of 'pooled z_v' and 'z_g' are now identical (N, d)
    assert z_v_pooled.shape == z_g.shape, \
        f"Shape mismatch: pooled z_v is {list(z_v_pooled.shape)}, but z_g is {list(z_g.shape)}"

    # 4. Normalize embeddings
    z_v_normalized = F.normalize(z_v_pooled, dim=1)  # shape (N, d)
    z_g_normalized = F.normalize(z_g, dim=1)         # shape (N, d)
    
    # 5. Compute cosine similarity matrix: (N, N)
    sim_matrix = torch.matmul(z_v_normalized, z_g_normalized.T)
    
    # 6. Scale by temperature
    sim_matrix = sim_matrix / temperature
    
    # 7. For each sample i, the positive example is at the diagonal (i,i)
    labels = torch.arange(z_v_normalized.size(0), device=z_v_normalized.device)

    # 8. Compute symmetric cross-entropy loss
    loss1 = F.cross_entropy(sim_matrix, labels)
    loss2 = F.cross_entropy(sim_matrix.T, labels)
    
    # 9. Average over the two
    loss = (loss1 + loss2) / 2

    return loss





# import torch
# import torch.nn.functional as F

# def contrastive_loss(z_v, z_g, temperature):
#     """
#     z_v: Tensor of shape (N, d) -- visual embeddings
#     z_g: Tensor of shape (N, d) -- genomic embeddings
#     temperature: temperature scalar
#     """
#     # Normalize embeddings
#     z_v = F.normalize(z_v, dim=1)  # shape (N, d)
#     z_g = F.normalize(z_g, dim=1)  # shape (N, d)
    
#     # Compute cosine similarity matrix: (N, N)
#     sim_matrix = torch.matmul(z_v, z_g.T)  # cosine similarity after normalization
    
#     # Scale by temperature
#     sim_matrix = sim_matrix / temperature
    
#     # For each sample i, the positive example is at the diagonal (i,i)
#     labels = torch.arange(z_v.size(0), device=z_v.device)  # shape (N,)

#     # Compute cross-entropy loss (both g to v & v to g)
#     loss1 = F.cross_entropy(sim_matrix, labels)
#     loss2 = F.cross_entropy(sim_matrix.T, labels)
    
#     # then average over the two
#     loss = (loss1 + loss2) / 2

#     return loss


# # z_v = torch.tensor([[1,2,3,4], [234, 234, 425, 323], [-56, 2, -42, 3]], dtype=torch.float32)
# # z_g = torch.tensor([[5, 6, 12, 24], [343, 234, 122, 667], [-34, 23, -1, 6]], dtype=torch.float32)
# # loss = contrastive_loss(z_v, z_g)
# # print(loss)





