import json
import os

import numpy as np
import torch
from torch.optim import Adam

from src.exp.exp_base import Exp_base
from src.exp.exp_utils import compute_ap, plot_indices


class Exp_main(Exp_base):
    def detect(self):
        train_dataloader = self.get_dataloader()
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
        np.save(self.args.output_path + "/labels.npy", all_labels)
        json.dump(indices, open(self.args.output_path + "/indices.json", "w"))

        if self.args.plot:
            self.logger.info("Plotting Top-Bottom Indices")
            plot_indices(indices, self.model.config.num_layers, self.args.plot_path)

    def intervention(self):
        lang = ["en", "de", "fr", "es", "zh", "ja"]
        labels = np.load(self.args.output_path + "/labels.npy")
        labels = labels[:, lang.index("ja")]

        indices = json.load(open(self.args.output_path + "/indices.json", "r"))
        top_bottom_indices = indices["ja"]["top"] + indices["ja"]["bottom"]  # ToDo: change to self.args.lang
        top_bottom_indices = sorted(top_bottom_indices)

        neurons = np.load(self.args.output_path + "/neurons.npy")
        top_bottom_neurons = neurons[labels == 1][:, top_bottom_indices]
        fixed_neurons = np.median(top_bottom_neurons, axis=0)  # ToDo: change name

        # ToDo: clean
        num_layers = self.model.config.num_layers
        d_model = self.model.config.d_model
        neuron_indices = []
        hidden_indices = []
        for i in range(num_layers):
            neuron_indices_ = []
            hidden_indices_ = []

            range_ = np.arange(i * d_model * 9, (i + 1) * d_model * 9)
            q_indices = range_[:d_model]
            k_indices = range_[d_model : 2 * d_model]
            v_indices = range_[2 * d_model : 3 * d_model]
            o1_indices = range_[3 * d_model : 4 * d_model]
            f_indices = range_[4 * d_model : 8 * d_model]
            o2_indices = range_[8 * d_model : 9 * d_model]

            neuron_indices_.append([j for j, k in enumerate(top_bottom_indices) if k in q_indices])
            neuron_indices_.append([j for j, k in enumerate(top_bottom_indices) if k in k_indices])
            neuron_indices_.append([j for j, k in enumerate(top_bottom_indices) if k in v_indices])
            neuron_indices_.append([j for j, k in enumerate(top_bottom_indices) if k in o1_indices])
            neuron_indices_.append([j for j, k in enumerate(top_bottom_indices) if k in f_indices])
            neuron_indices_.append([j for j, k in enumerate(top_bottom_indices) if k in o2_indices])

            hidden_indices_.append([k % (d_model * 9) for k in top_bottom_indices if k in q_indices])
            hidden_indices_.append([k % (d_model * 9) - d_model for k in top_bottom_indices if k in k_indices])
            hidden_indices_.append([k % (d_model * 9) - 2 * d_model for k in top_bottom_indices if k in v_indices])
            hidden_indices_.append([k % (d_model * 9) - 3 * d_model for k in top_bottom_indices if k in o1_indices])
            hidden_indices_.append([k % (d_model * 9) - 4 * d_model for k in top_bottom_indices if k in f_indices])
            hidden_indices_.append([k % (d_model * 9) - 8 * d_model for k in top_bottom_indices if k in o2_indices])

            neuron_indices.append(neuron_indices_)
            hidden_indices.append(hidden_indices_)

        self.logger.info("Start intervention")
        self.model.eval()

        generate_kwargs = {
            "max_length": 64,
            "do_sample": True,
            "top_p": 0.9,
        }

        for _ in range(10):
            outputs = self.model.generate(
                input_ids=self.tokenizer.encode("</s>", return_tensors="pt").to(self.device),
                fixed_neurons=torch.tensor(fixed_neurons).to(self.device),
                neuron_indices=neuron_indices,
                hidden_indices=hidden_indices,
                **generate_kwargs,
            )

            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            print(generated_texts)
