import torch
from src.models.adapter import ProteinAdapter
from src.config import Config

def test_qformer():
    config = Config().from_yaml("configs/polytamgen_sft.yaml")
    config.pp2d.protein_adapter_type = 'qformer'
    config.pp2d.num_queries = 32
    config.pp2d.set_transformer_heads = 4
    config.pp2d.adapter_layers = 2
    
    input_dim = 640
    output_dim = 768
    
    adapter = ProteinAdapter(config, input_dim, output_dim)
    
    protein_feats = torch.randn(2, 100, input_dim)
    
    protein_mask = torch.ones(2, 100)
    protein_mask[1, 50:] = 0
    
    output = adapter(protein_feats, protein_mask)
    
    expected_shape = (2, 32, output_dim)
    assert output.shape == expected_shape

if __name__ == "__main__":
    test_qformer()
