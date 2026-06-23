import torch
from src.data import LigandProfileDataset
from src.config import Config
from transformers import AutoTokenizer

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
    def __call__(self, text, **kwargs):
        return {'input_ids': torch.tensor([1, 2, 3])}
    def encode(self, text, **kwargs):
        return [1]
    def add_tokens(self, tokens):
        pass

def test_subset():
    config = Config().from_yaml("configs/polytamgen_sft.yaml")
    config.data.train_subset = True
    config.data.train_subset_size = 50
    
    dataset = LigandProfileDataset(
        config, 
        tokenizer=MockTokenizer(), 
        split='train',
    )
    
    assert len(dataset) == 50

if __name__ == "__main__":
    test_subset()
