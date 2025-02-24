import json
import os

import numpy as np
import torch

from src.exp.exp_base import Exp_base
from src.exp.exp_utils import compute_ap


class Exp_main(Exp_base):
    def detect(self):
        train_dataloader = self.get_dataloader()
        self.model.eval()

        self.logger.info("Detecting neurons")
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

        # Compute APs and return top-middle-bottom indices
        self.logger.info("Computing Average Precision")
        indices = compute_ap(all_neurons, all_labels)

        # Save fixed neurons and indices for intervention
        self.logger.info("Saving outputs")
        output_path = os.path.join(self.args.output_path, self.args.lm_name)
        self.save_outputs(indices, all_neurons, all_labels, output_path)

        # Plot top-middle-bottom indices
        if self.args.plot:
            self.logger.info("Plotting top-bottom indices")
            plot_path = os.path.join(self.args.plot_path, self.args.lm_name)
            self.plot_indices(indices, plot_path)

    def inference(self):
        self.logger.info("Start intervention")
        self.model.eval()

        output_path = os.path.join(self.args.output_path, self.args.lm_name)
        fixed_neurons = np.load(os.path.join(output_path, "fixed_neurons.npy"))
        neuron_indices = json.load(open(os.path.join(output_path, "neuron_indices.json"), "r"))
        hidden_indices = json.load(open(os.path.join(output_path, "hidden_indices.json"), "r"))

        lang = ["en", "de", "fr", "es", "zh", "ja"]
        texts = {}
        for i, l in enumerate(lang):
            self.logger.info(f"Intervention for {l}")

            texts_ = []
            gen_kwargs = {"max_length": 64, "do_sample": True, "top_p": 0.9}
            for _ in range(self.args.num_samples):
                # ToDo: support bloom model
                outputs = self.model.generate(
                    input_ids=self.tokenizer.encode("</s>", return_tensors="pt").to(self.device),
                    fixed_neurons=torch.tensor(fixed_neurons[i]).to(device=self.device, dtype=self.dtype),
                    neuron_indices=neuron_indices[i],
                    hidden_indices=hidden_indices[i],
                    **gen_kwargs,
                )
                generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                texts_.append(generated_text)

            texts[l] = texts_

        # Save generated texts
        json.dump(texts, open(os.path.join(self.args.output_path, self.args.lm_name, "texts.json"), "w"))
