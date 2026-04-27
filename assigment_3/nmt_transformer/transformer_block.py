import torch 
import torch.nn as nn
class TransformerEmbedding(nn.Module):
    def __init__(self,vocab_size : int,embed_dim:int,max_seq_len:int):
        super().__init__()
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
        return x.view(B,n,self.num_heads,self.head_dim).transpose(1,2) # Shape: (B, self.num_heads, n, self.head_dim)
    
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
    
class FeedForward(nn.Module):
    def __init__(self,embed_dim:int,intermediate_dim : int ,dropout : float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim,intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim,embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self,x : torch.Tensor):
        return self.net(x) # Shape: (B, n, embed_dim)

class addNorm(nn.Module):

    def __init__(self,embed_dim : int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,x : torch.Tensor,sub_layer_output : torch.Tensor):
        return self.norm(x + sub_layer_output) # Shape: (B, n, embed_dim)
    
class EncoderLayer(nn.Module):
    def __init__(self,embed_dim : int,num_heads : int,intermediate_dim :int,dropout : float):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim,num_heads)
        self.ffn = FeedForward(embed_dim,intermediate_dim,dropout)
        self.add_norm1 = addNorm(embed_dim)
        self.add_norm2 = addNorm(embed_dim)

    def forward(self,x : torch.Tensor,src_key_padding_mask : torch.Tensor | None = None):
        attn_out , attn_weights = self.self_attn(query=x,key=x,value=x,key_padding_mask=src_key_padding_mask,is_causal=False) # Self-attention
        x = self.add_norm1(x,attn_out) # Add & Norm
        x = self.add_norm2(x,self.ffn(x)) # Add & Norm
        return x , attn_weights

class DecoderLayer(nn.Module):
    def __init__(self,embed_dim : int,num_heads :int,intermediate_dim : int,droput:float):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim,num_heads)
        self.cross_attn = MultiHeadAttention(embed_dim,num_heads)
        self.ffn = FeedForward(embed_dim,intermediate_dim,droput)
        self.add_norm1 = addNorm(embed_dim)
        self.add_norm2 = addNorm(embed_dim)
        self.add_norm3 = addNorm(embed_dim)
    
    def forward(self,x : torch.Tensor # (B, tgt_len, embed_dim)
                ,enc_output : torch.Tensor # (B, src_len, embed_dim)
                ,tgt_key_padding_mask : torch.Tensor | None = None, # (B, tgt_len)
                src_key_padding_mask : torch.Tensor | None = None # (B, src_len)
                ):
        self_attn_out , self_attn_weights = self.self_attn(query=x,key=x,value=x,key_padding_mask=tgt_key_padding_mask,is_causal=True) # Causal Self-attention
        x = self.add_norm1(x,self_attn_out)
        cross_attn_out , cross_attn_weights = self.cross_attn(query=x,key=enc_output,value=enc_output,key_padding_mask=src_key_padding_mask,is_causal=False) # Cross-attention
        x = self.add_norm2(x,cross_attn_out)
        x = self.add_norm3(x,self.ffn(x))
        return x , self_attn_weights , cross_attn_weights

if __name__ == "__main__":
    d      = 32          # hidden size
    dI     = 32 * 4      # intermediate size
    H      = 4           # num attention heads
    N      = 32          # max sequence length
    V_enc  = 8000        # encoder vocab size (example)
    V_dec  = 6000        # decoder vocab size (example)
    B      = 4           # batch size
    src_n  = 10          # source sequence length
    tgt_n  = 8           # target sequence length
    drop   = 0.1

    # ---- Embedding -------------------------------------------------------
    enc_embed = TransformerEmbedding(V_enc, d, N)
    dec_embed = TransformerEmbedding(V_dec, d, N)

    src_ids = torch.randint(0, V_enc, (B, src_n))
    tgt_ids = torch.randint(0, V_dec, (B, tgt_n))

    src_emb = enc_embed(src_ids)     # (B, src_n, d)
    tgt_emb = dec_embed(tgt_ids)     # (B, tgt_n, d)
    print(f"Encoder embedding : {src_emb.shape}")   # (4, 10, 32)
    print(f"Decoder embedding : {tgt_emb.shape}")   # (4,  8, 32)

    # ---- Padding masks (last 2 tokens are PAD) ---------------------------
    src_pad_mask = torch.zeros(B, src_n, dtype=torch.bool)
    src_pad_mask[:, -2:] = True      # last 2 positions are PAD

    tgt_pad_mask = torch.zeros(B, tgt_n, dtype=torch.bool)
    tgt_pad_mask[:, -1:] = True      # last position is PAD

    # ---- Encoder Layer ---------------------------------------------------
    enc_layer = EncoderLayer(d, H, dI, drop)
    enc_out, enc_attn = enc_layer(src_emb, src_key_padding_mask=src_pad_mask)
    print(f"Encoder output    : {enc_out.shape}")   # (4, 10, 32)
    print(f"Enc attn weights  : {enc_attn.shape}")  # (4,  4, 10, 10)

    # ---- Decoder Layer ---------------------------------------------------
    dec_layer = DecoderLayer(d, H, dI, drop)
    dec_out, self_w, cross_w = dec_layer(
        tgt_emb, enc_out,
        tgt_key_padding_mask=tgt_pad_mask,
        src_key_padding_mask=src_pad_mask,
    )
    print(f"Decoder output    : {dec_out.shape}")   # (4, 8, 32)
    print(f"Self  attn weights: {self_w.shape}")    # (4, 4, 8,  8)
    print(f"Cross attn weights: {cross_w.shape}")   # (4, 4, 8, 10)

    print("\nAll shapes correct!")



    




        
