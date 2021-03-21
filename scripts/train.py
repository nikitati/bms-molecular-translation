import os

import click
import yaml
import pytorch_lightning as pl

from src.data import MolecularCaptioningDataModule
from src.models import MolecularCaptioningModel
from src.utils import selfies2inchi
from src.evaluation import Evaluator


@click.command()
@click.option('--config', type=click.Path())
@click.option('--dataset', type=click.Path())
@click.option('--targets', type=click.Path())
@click.option('--test-path', type=click.Path())
def main(config, dataset, targets, test_path):
    pl.seed_everything(42)
    with open(config, 'r') as f:
        conf = yaml.safe_load(f)
    datamodule = MolecularCaptioningDataModule(
        dataset_path=dataset,
        target=conf['target'],
        targets_path=targets,
        test_ids_path=test_path,
        imsize=conf['imsize'],
        batch_size=conf['batch_size']
    )
    datamodule.setup('fit')
    evaluator = Evaluator(datamodule.val)
    model = MolecularCaptioningModel(
        learning_rate=conf['learning_rate'],
        encoded_image_size=conf['encoded_image_size'],
        input_grayscale=conf['input_grayscale'],
        encoder_model=conf['encoder_model'],
        encoder_fine_tune=conf['encoder_fine_tune'],
        encoder_pretrained=conf['encoder_pretrained'],
        attention_dim=conf['attention_dim'],
        embedding_dim=conf['embedding_dim'],
        decoder_dim=conf['decoder_dim'],
        encoder_dim=conf['encoder_dim'],
        decoder_dropout=conf['decoder_dropout'],
        tokenizer=datamodule.tokenizer,
        target=conf['target'],
        translate_fn=selfies2inchi,
        max_pred_len=conf['max_pred_len'],
        evaluator=evaluator
    )
    trainer = pl.Trainer(
        deterministic=True,
        gpus=conf['gpus'],
        max_epochs=conf['epochs'],
        gradient_clip_val=conf['max_grad_norm']
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
