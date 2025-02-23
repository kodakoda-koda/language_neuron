import json
import os

import numpy as np
import torch

from src.exp.exp_base import Exp_base
from src.exp.exp_utils import compute_ap, intervention_indices, plot_indices


class Exp_main(Exp_base):
    def detect(self):
        train_dataloader = self.get_dataloader()
        self.logger.info("Start detecting neurons")
        self.model.eval()

        all_neurons = []
        all_labels = []
        for i, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_neurons=True)
            neurons = outputs.neurons
            all_neurons.append(neurons.float().cpu().detach().numpy())
            all_labels.append(labels.float().cpu().detach().numpy())

        all_neurons = np.concatenate(all_neurons, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        self.logger.info("Start computing scores")
        indices = compute_ap(all_neurons, all_labels)

        self.logger.info("Saving outputs")
        if not os.path.exists(os.path.join(self.args.output_path, self.args.lm_name)):
            os.makedirs(os.path.join(self.args.output_path, self.args.lm_name))
        np.save(os.path.join(self.args.output_path, self.args.lm_name, "neurons.npy"), all_neurons)
        np.save(os.path.join(self.args.output_path, self.args.lm_name, "labels.npy"), all_labels)
        json.dump(indices, open(os.path.join(self.args.output_path, self.args.lm_name, "indices.json"), "w"))

        if self.args.plot:
            self.logger.info("Plotting Top-Bottom Indices")
            plot_indices(indices, self.model.config.num_layers, self.args.plot_path, self.args.lm_name)

    def inference(self):
        self.logger.info("Start intervention")
        self.model.eval()

        lang = ["en", "de", "fr", "es", "zh", "ja"]
        texts = {}
        for i, l in enumerate(lang):
            self.logger.info(f"Intervention for {l}")

            neurons = np.load(os.path.join(self.args.output_path, self.args.lm_name, "neurons.npy"))
            labels = np.load(os.path.join(self.args.output_path, self.args.lm_name, "labels.npy"))[:, i]
            indices = json.load(open(os.path.join(self.args.output_path, self.args.lm_name, "indices.json"), "r"))

            top_bottom_indices = sorted(indices[l]["top"] + indices[l]["bottom"])
            top_bottom_neurons = neurons[labels == 1][:, top_bottom_indices]
            fixed_neurons = np.median(top_bottom_neurons, axis=0)
            fixed_neurons = torch.tensor(fixed_neurons).to(device=self.device, dtype=self.dtype)

            num_layers = self.model.config.num_layers
            d_model = self.model.config.d_model
            neuron_indices, hidden_indices = intervention_indices(num_layers, d_model, top_bottom_indices)

            texts_ = []
            for _ in range(self.args.num_samples):
                outputs = self.model.generate(
                    input_ids=self.tokenizer.encode("</s>", return_tensors="pt").to(self.device),
                    fixed_neurons=torch.tensor(fixed_neurons).to(self.device),
                    neuron_indices=neuron_indices,
                    hidden_indices=hidden_indices,
                    max_length=64,
                    do_sample=True,
                    top_p=0.9,
                )
                generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                texts_.append(generated_text)

            texts[l] = texts_

        json.dump(texts, open(os.path.join(self.args.output_path, self.args.lm_name, "texts.json"), "w"))
