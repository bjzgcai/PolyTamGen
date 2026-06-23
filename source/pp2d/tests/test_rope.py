import torch
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.set_transformer import RoPEAwareEncoderLayer, ProteinSetTransformer
from src.config import Config

def test_rope_rigorous():
    embed_dim = 64
    num_heads = 4
    head_dim = embed_dim // num_heads

    config = Config()
    config.pp2d.set_transformer_heads = num_heads
    config.pp2d.set_transformer_layers = 1
    
    model = ProteinSetTransformer(config, embed_dim)
    model.eval()
    
    device = torch.device("cpu")

    q_base = torch.randn(1, 1, 1, head_dim)
    k_base = torch.randn(1, 1, 1, head_dim)
    
    def get_dot_product(pos_q, pos_k):
        max_pos = max(pos_q, pos_k) + 1
        freqs_cis = model._precompute_freqs_cis(max_pos, device)
        
        freq_q = freqs_cis[:, pos_q:pos_q+1, :, :]
        freq_k = freqs_cis[:, pos_k:pos_k+1, :, :]
        
        q_in = q_base.expand(1, 1, num_heads, head_dim)
        k_in = k_base.expand(1, 1, num_heads, head_dim)
        
        q_rot = model.layers[0].apply_rotary_emb(q_in, freq_q)
        k_rot = model.layers[0].apply_rotary_emb(k_in, freq_k)
        
        vec_q = q_rot[0, 0, 0, :]
        vec_k = k_rot[0, 0, 0, :]
        
        return torch.dot(vec_q, vec_k).item()

    score_A = get_dot_product(pos_q=0, pos_k=5)
    score_B = get_dot_product(pos_q=100, pos_k=105)
    
    diff = abs(score_A - score_B)
    assert diff < 1e-5

    x_test = torch.zeros(1, 1, num_heads, head_dim)
    x_test[..., 0::2] = 1.0 
    x_test[..., 1::2] = 0.0
    
    target_pos = 1
    
    freqs_cis = model._precompute_freqs_cis(target_pos + 1, device)
    freq_at_pos = freqs_cis[:, target_pos:target_pos+1, :, :]
    
    out_test = model.layers[0].apply_rotary_emb(x_test, freq_at_pos)
    
    expected_cos = math.cos(1.0)
    expected_sin = math.sin(1.0)
    
    actual_cos = out_test[0, 0, 0, 0].item()
    actual_sin = out_test[0, 0, 0, 1].item()
    
    assert abs(actual_cos - expected_cos) < 1e-5
    assert abs(actual_sin - expected_sin) < 1e-5

if __name__ == "__main__":
    test_rope_rigorous()
    
