# initial exploration sweep: getting ICNN to fit a simple convex function
import torch
import wandb

from icnn_debug import ICNN

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def objective(model):
    # test at grid points in the space.
    x1 = torch.linspace(-2, 2, 100)
    x2 = torch.linspace(-2, 2, 100)
    X1, X2 = torch.meshgrid(x1, x2)

    test_inputs = torch.stack([X1, X2])
    test_inputs = test_inputs.permute((1, 2, 0)).reshape(-1, 2)

    test_targets = model.target(test_inputs)

    model.eval()
    score = model.lossFn(model.fwd_cpu(test_inputs), test_targets)

    return score.item()


def train(model, run_id):
    """ Simple training: early stopping """
    trainer_params = {'max_epochs': 3000,
                      'patience': 100, 
                      'gradient_clip_val': 0}

    earlystopping_callback = EarlyStopping(monitor='Val Loss', mode='min', 
                                           patience=trainer_params.pop('patience')
                                           )

    checkpoint_callback = ModelCheckpoint(dirpath='trainedParameters/sweep01/',
                                          filename=f'{run_id}',
                                          every_n_epochs=1, 
                                          save_top_k=1,
                                          monitor='Val Loss',
                                          save_weights_only=True,
                                          save_last=False
                                          )

    callbacks = [checkpoint_callback, earlystopping_callback]

    logger = WandbLogger()

    trainer = Trainer(logger=logger,
                      callbacks=callbacks,
                      log_every_n_steps=20,
                      **trainer_params
                      )
    trainer.fit(model)

    return trainer.checkpoint_callback.best_model_path


def main():
    run = wandb.init()

    config = wandb.config

    wandb_logger = WandbLogger()
    model = ICNN(**config)

    ckpt = train(model, run.id)

    model = ICNN.load_from_checkpoint(ckpt)

    score = objective(model)
    wandb.log({"score": score})


# 2: Define the search space
if __name__ == '__main__':
    wandb.login()
    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "minimize", "name": "score"},
        "parameters": {
            "hidden_dims": {"values": [(10, 6), (6, 6, 6), (20, 10, 6)]},
            "dropout": {"min": 0., "max": 0.8},
            "CELU_alpha": {"min": 0.5, "max": 5.},
            "lr": {"min": 1E-5, "max": 1E-1},
            "batchsize": {"min": 16, "max": 256},
            "num_samples": {"min": 1000, "max": 10000},
            "variance": {"min": 0.05, "max": 3.}
        },
    }

    # 3: Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='learnedExpFam')

    wandb.agent(sweep_id, function=main, count=30)
