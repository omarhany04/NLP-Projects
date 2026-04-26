import torch 
import torch.nn as nn
class TransformerEmbedding(nn.Module):
    def __init(self,vocab_size : int,embed_dim:int,max_seq_len:int):
        super.__init__()
        self.token_embedding = nn.Embedding(vocab_size,embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len,embed_dim)

    def forward(self,token_ids : torch.Tensor):
        B , n = token_ids.shape # Batch size and sequence length
        positions = torch.arange(n , device= token_ids.device).unsqueeze(0)    # Shape: (1, n)
        return self.token_embedding(token_ids) + self.position_embedding(positions) # Shape: (B, n, embed_dim)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads # Dimension of each head
        self.W_q = nn.Linear(embed_dim, embed_dim , bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim , bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim , bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim , bias=False)

    def _split_heads(self , x : torch.Tensor):
        B , n , _ = x.shape
        return x.view(B,n,self.H,self.dH).transpose(1,2) # Shape: (B, H, n, dH)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,key_padding_mask : torch.Tensor | None = None,is_causal : bool = False):
        B , tgt_len , _ = query.shape
        src_len = key.shape[1]
        Q = self._split_heads(self.W_q(query)) # Shape: (B, H, tgt_len, dH)
        K = self._split_heads(self.W_k(key))   # Shape: (B, H, src_len, dH)
        V = self._split_heads(self.W_v(value)) # Shape: (B, H, src_len, dH)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5) # Shape: (B, H, tgt_len, src_len)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')) # Mask padding tokens
        if is_causal:
            causal_mask = torch.triu(torch.ones(tgt_len, src_len, device=query.device), diagonal=1).bool() # Shape: (tgt_len, src_len)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf')) # Mask future tokens
        attn_weights = torch.softmax(scores, dim=-1) # Shape: (B, H, tgt_len, src_len)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0) # Replace NaNs with zeros
        context = torch.matmul(attn_weights, V) # Shape: (B, H, tgt_len, dH)
        context = context.transpose(1, 2).contiguous().view(B, tgt_len, self.embed_dim) # Shape: (B, tgt_len, embed_dim)
        output = self.W_o(context) # Shape: (B, tgt_len, embed_dim)
        return output , attn_weights
    




        
