import numpy as np
from torch.optim import Adam

from src.exp.exp_base import Exp_base
from src.exp.exp_utils import compute_ap, plot_scores


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
        score = compute_ap(all_neurons, all_labels)

        self.logger.info("Plotting scores")
        plot_scores(score)
