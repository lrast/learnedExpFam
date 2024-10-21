import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def trainModel(model, directory, project='learnedExpFam', **trainer_kwargs):
    """Simple training behavior with checkpointing"""
    wandb.init(reinit=True)

    trainer_params = ['max_epochs', 'patience', 'gradient_clip_val']
    model_defaults = dict(filter(lambda i: (i[0] in trainer_params), model.hparams.items()))
    model_defaults.update(trainer_kwargs)
    trainer_kwargs = model_defaults

    wandb_logger = WandbLogger(project=project)

    earlystopping_callback = EarlyStopping(monitor='Val Loss', mode='min', 
                                           patience=trainer_kwargs.pop('patience')
                                           )

    checkpoint_callback_val = ModelCheckpoint(dirpath=directory,
                                              filename='validation-{epoch}-{step}',
                                              every_n_epochs=1, 
                                              save_top_k=1,
                                              monitor='Val Loss',
                                              save_weights_only=True,
                                              save_last=False
                                              )

    callbacks = [checkpoint_callback_val, earlystopping_callback]

    trainer = Trainer(logger=wandb_logger,
                      callbacks=callbacks,
                      log_every_n_steps=30,
                      **trainer_kwargs
                      )
    trainer.fit(model)

    wandb.finish()

    return trainer.checkpoint_callback.best_model_path
