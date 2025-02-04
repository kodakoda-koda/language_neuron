import json
import os

import numpy as np
from torch.optim import Adam

from src.exp.exp_base import Exp_base
from src.exp.exp_utils import compute_ap, plot_indices


class Exp_main(Exp_base):
    def detect(self):
        train_dataloader = self._get_dataloader(train_flag=True)
        self.model.eval()
        all_neurons = []
        all_labels = []
        self.logger.info("Start detecting neurons")
        for i, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_neurons=True)
            neurons = outputs.neurons
            all_neurons.append(neurons.cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())

        all_neurons = np.concatenate(all_neurons, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        self.logger.info("Start computing scores")
        indices = compute_ap(all_neurons, all_labels)

        self.logger.info("Saving outputs")
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)
        np.save(self.args.output_path + "/neurons.npy", all_neurons)
        json.dump(indices, open(self.args.output_path + "/indices.json", "w"))

        if self.args.plot:
            self.logger.info("Plotting Top-Bottom Indices")
            plot_indices(indices, self.model.config.num_layers, self.args.plot_path)
