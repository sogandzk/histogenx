import torch
import torch.nn.functional as F

def contrastive_loss(z_v, z_g, temperature):
    """
    z_v: Tensor of shape (N, d) -- visual embeddings
    z_g: Tensor of shape (N, d) -- genomic embeddings
    temperature: temperature scalar
    """
    # Normalize embeddings
    z_v = F.normalize(z_v, dim=1)  # shape (N, d)
    z_g = F.normalize(z_g, dim=1)  # shape (N, d)
    
    # Compute cosine similarity matrix: (N, N)
    sim_matrix = torch.matmul(z_v, z_g.T)  # cosine similarity after normalization
    
    # Scale by temperature
    sim_matrix = sim_matrix / temperature
    
    # For each sample i, the positive example is at the diagonal (i,i)
    labels = torch.arange(z_v.size(0), device=z_v.device)  # shape (N,)

    # Compute cross-entropy loss (both g to v & v to g)
    loss1 = F.cross_entropy(sim_matrix, labels)
    loss2 = F.cross_entropy(sim_matrix.T, labels)
    
    # then average over the two
    loss = (loss1 + loss2) / 2

    return loss


# z_v = torch.tensor([[1,2,3,4], [234, 234, 425, 323], [-56, 2, -42, 3]], dtype=torch.float32)
# z_g = torch.tensor([[5, 6, 12, 24], [343, 234, 122, 667], [-34, 23, -1, 6]], dtype=torch.float32)
# loss = contrastive_loss(z_v, z_g)
# print(loss)
