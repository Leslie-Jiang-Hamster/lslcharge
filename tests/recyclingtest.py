import esm_util
import torch
from depthcharge.components import RecyclingPeakEncoder

def seq2embedding(data):
    return esm_util.ESMUtil().load_data(data).get_sequence_representations()

def test_recycling_peak_encoder():
    enc = RecyclingPeakEncoder(8)

    # We want to test that the addition between the peak and
    # m/z encoding is happening:
    enc.int_encoder.weight = torch.nn.Parameter(torch.ones(8, 1))

    dummy_data = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ]
    X = torch.tensor([[[0.0, 0], [0, 1]]])
    Y = seq2embedding(dummy_data)
    Z = enc(X, Y[0].unsqueeze(0))
    
    assert Z.shape == (1, 2, 8)