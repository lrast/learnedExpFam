import torch
import pytorch_lightning as pl

from torch.utils.data import TensorDataset, DataLoader


class BaselineAdaptable(pl.LightningModule):
    """BaselineAdaptable: parent class for adaptable distributions"""
    def __init__(self, layer_ind,
                 data_transform=lambda mu, std: (lambda x: (x-mu)/std)):
        super(BaselineAdaptable, self).__init__()
        self.layer_ind = layer_ind
        self.make_transform = data_transform

    # activity data extraction and preprocessing
    def internal_activity(self, inputs):
        """ returns activity of the internal layer on inputs """
        return self.model[0:(self.layer_ind+1)](inputs.to(self.device))

    def fit_activity_transform(self, training_inputs):
        """ Fits the activity transform for this training data """
        inputs, targets = input_target_extraction(training_inputs)
        activity_data = self.internal_activity(inputs)

        self.mean = activity_data.mean()
        self.stdev = activity_data.var()**0.5

    def transform_activity_data(self, activity_data):
        return (activity_data - self.mean) / self.stdev

    def internal_activity_dataset(self, data):
        """
            make a dataset of the activity of a given layer in the model 
            after preprocessing
        """
        inputs, targets = input_target_extraction(data)

        model_activity = self.internal_activity(inputs)
        model_activity = self.transform_activity_data(model_activity).detach()

        if targets is None:
            return TensorDataset(model_activity.to('cpu'))
        else: 
            return TensorDataset(model_activity.to('cpu'), targets)

    # layer freezing and unfreezing 
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

    # loss function adjustment
    def make_weighted_loss(self, baseline_loss):
        loss_expanded = baseline_loss(reduction='none')

        def weighted_loss(outputs, targets, weights):
            return weights.reshape(-1, 1) * loss_expanded(outputs, targets).sum()
        return weighted_loss


# utility
def input_target_extraction(dataset):
    """ Extracts inputs and targets (if they exist) from the dataset """

    # pure tensor
    if isinstance(dataset, torch.Tensor):
        return dataset, None

    dl = DataLoader(dataset, batch_size=len(dataset))
    all_data = next(iter(dl))

    if len(all_data) == 1:
        targets = None
        inputs = all_data[0]
    else:
        inputs, targets = all_data

    return inputs, targets
