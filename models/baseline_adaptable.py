import pytorch_lightning as pl

from torch.utils.data import TensorDataset, DataLoader


class BaselineAdaptable(pl.LightningModule):
    """BaselineAdaptable: parent class for adaptable distributions"""
    def __init__(self, layer_ind):
        super(BaselineAdaptable, self).__init__()
        self.layer_ind = layer_ind

    def freeze_layers(self, n_layers):
        """ freeze the first n layers of the model """
        i = 0
        for layer in self.model:
            if i >= n_layers:
                break
            i += 1

            for p in layer.parameters():
                p.requires_grad = False

    def unfreeze_all_layers(self):
        """ unfreeze all layers """
        for p in self.model.parameters():
            p.requires_grad_()

    def internal_activity_dataset(self, data):
        """ make a dataset of the activity of a given layer in the model """
        dl = DataLoader(data, batch_size=len(data))
        inputs, targets = next(iter(dl))
        inputs = inputs.to(self.device)
        model_activity = self.model[0:(self.layer_ind+1)](inputs).detach()

        return TensorDataset(model_activity.to('cpu'), targets)

    def make_weighted_loss(self, baseline_loss):
        loss_expanded = baseline_loss(reduction='none')

        def weighted_loss(outputs, targets, weights):
            return weights.reshape(-1, 1) * loss_expanded(outputs, targets).sum()
        return weighted_loss
