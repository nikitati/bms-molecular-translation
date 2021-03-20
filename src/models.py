import torch
import pytorch_lightning as pl

from src.modules import Encoder, Attention, DecoderWithAttention

# TODO
class MolecularCaptioningModel(pl.LightningModule):

    def __init__(self):
        super(MolecularCaptioningModel, self).__init__()
        self.encoder = Encoder()
        self.attention = Attention()
        self.decoder = DecoderWithAttention()
