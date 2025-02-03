from torch.optim import Adam

from src.exp_base import Exp_base


class Exp_main(Exp_base):
    def train(self):
        train_dataloader = self._get_dataloader(train_flag=True)
        optimizer = Adam(self.model.parameters(), lr=self.args.lr)

        for epoch in range(self.args.num_epochs):
            losses = []
            self.model.train()
            for i, batch in enumerate(train_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["input_ids"].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

            self.logger.info(f"epoch: {epoch + 1}, train loss: {sum(losses) / len(losses)}")

    def test(self):
        test_dataloader = self._get_dataloader(train_flag=False)
        self.model.eval()
        losses = []
        for i, batch in enumerate(test_dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["input_ids"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())

        self.logger.info(f"test loss: {sum(losses) / len(losses)}")
