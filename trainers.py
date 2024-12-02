import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def train_model(model, directory, log_wandb=True, project='learnedExpFam',
                **trainer_kwargs):
    """Simple training with earlystopping and checkpointing"""

    trainer_params = {'max_epochs': 1000,
                      'patience': 100, 
                      'gradient_clip_val': 0}

    model_defaults = dict(filter(
                            lambda i: (i[0] in trainer_params.keys()),
                            model.hparams.items()
                          ))
    trainer_params.update(model_defaults)
    trainer_params.update(trainer_kwargs)

    earlystopping_callback = EarlyStopping(monitor='Val Loss', mode='min', 
                                           patience=trainer_params.pop('patience')
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

    logger = True
    if log_wandb:
        wandb.init(reinit=True)
        logger = WandbLogger(project=project)

    trainer = Trainer(logger=logger,
                      callbacks=callbacks,
                      log_every_n_steps=30,
                      **trainer_params
                      )
    trainer.fit(model)

    if log_wandb:
        wandb.finish()

    return trainer.checkpoint_callback.best_model_path
