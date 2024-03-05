# Mini simple version for predictions

import torch
import torch.nn as nn

class NewsMF(nn.Module):
    def __init__(self, num_users, num_items, dim=10):
        super().__init__()
        self.dim = dim
        self.useremb = nn.Embedding(num_embeddings=num_users, embedding_dim=dim)
        self.itememb = nn.Embedding(num_embeddings=num_items, embedding_dim=dim)

    def forward(self, user, item):
        batch_size = user.size(0)
        uservec = self.useremb(user)
        itemvec = self.itememb(item)

        score = (uservec * itemvec).sum(-1).unsqueeze(-1)

        return score
