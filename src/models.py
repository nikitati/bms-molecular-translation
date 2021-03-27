import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence
import pytorch_lightning as pl

from src.modules import Encoder, DecoderWithAttention


class MolecularCaptioningModel(pl.LightningModule):

    def __init__(
            self,
            learning_rate,
            encoded_image_size,
            input_grayscale,
            encoder_model,
            encoder_fine_tune,
            encoder_pretrained,
            attention_dim,
            embedding_dim,
            decoder_dim,
            encoder_dim,
            decoder_dropout,
            tokenizer,
            target,
            translate_fn,
            evaluator,
            max_pred_len=120
    ):
        super(MolecularCaptioningModel, self).__init__()
        self.save_hyperparameters(
            "learning_rate",
            "encoded_image_size",
            "input_grayscale",
            "encoder_model",
            "encoder_fine_tune",
            "encoder_pretrained",
            "attention_dim",
            "embedding_dim",
            "decoder_dim",
            "encoder_dim",
            "decoder_dropout",
            "target",
            "max_pred_len"
        )
        self.learning_rate = learning_rate
        self.encoder = Encoder(
            encoded_image_size=encoded_image_size,
            is_grayscale=input_grayscale,
            model=encoder_model,
            fine_tune=encoder_fine_tune,
            pretrained=encoder_pretrained
        )
        self.decoder = DecoderWithAttention(
            attention_dim=attention_dim,
            embed_dim=embedding_dim,
            decoder_dim=decoder_dim,
            vocab_size=len(tokenizer),
            encoder_dim=encoder_dim,
            dropout=decoder_dropout,
            device=self.device
        )
        self.max_pred_len = max_pred_len
        self.tokenizer = tokenizer
        self.evaluator = evaluator
        self.target = target
        self.translate_fn = translate_fn
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi["<PAD>"])

    def forward(self, images):
        pass

    def _compute_loss(self, images, labels, label_lengths):
        features = self.encoder(images)
        predictions_, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(features, labels, label_lengths)
        targets = caps_sorted[:, 1:]
        predictions = pack_padded_sequence(predictions_, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        loss = self.criterion(predictions, targets)
        return loss, features

    def training_step(self, batch, batch_idx):
        images, labels, label_lengths = batch
        loss, _ = self._compute_loss(images, labels, label_lengths)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, label_lengths, idxs = batch
        loss, encoded = self._compute_loss(images, labels, label_lengths)
        predictions = self.decoder.decode(encoded, self.max_pred_len, self.tokenizer)
        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        predicted_sequences = [self.tokenizer.reverse_tokenize(seq) for seq in predicted_sequence]
        translated_sequences = [self.translate_fn(seq) for seq in predicted_sequences]
        cv_ld = self.evaluator.eval_batch(idxs.detach().cpu().numpy().ravel(), translated_sequences)
        self.log('val_loss', loss)
        self.log('cv_ld', cv_ld)

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            amsgrad=False
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=4,
            eps=1e-6,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
