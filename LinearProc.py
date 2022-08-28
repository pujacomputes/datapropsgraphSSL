import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score, accuracy_score

from GCL.eval import BaseEvaluator
from torch.utils.data import TensorDataset, DataLoader
import pdb


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        z = self.fc(x)
        return z


class LinearProc(BaseEvaluator):
    def __init__(
        self,
        num_epochs: int = 200,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        test_interval: int = 20,
    ):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(
            classifier.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(x[split["train"]], y[split["train"]])
        train_dataloader = DataLoader(dataset=dataset, batch_size=128)

        for epoch in range(self.num_epochs):
            classifier.train()
            epoch_loss = 0
            for data in train_dataloader:
                x_data = data[0]
                y_data = data[1]
                optimizer.zero_grad()
                output = classifier(x_data)
                loss = criterion(output, y_data)
                # output = classifier(x[split["train"]])
                # loss = criterion(output, y[split["train"]])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        classifier.eval()
        y_train_pred = classifier(x[split["train"]]).argmax(-1).detach().cpu().numpy()
        y_train_label = y[split["train"]].detach().cpu().numpy()

        y_valid_pred = classifier(x[split["valid"]]).argmax(-1).detach().cpu().numpy()
        y_valid_label = y[split["valid"]].detach().cpu().numpy()

        y_test_pred = classifier(x[split["test"]]).argmax(-1).detach().cpu().numpy()
        y_test_label = y[split["test"]].detach().cpu().numpy()

        train_acc = accuracy_score(y_true=y_train_label, y_pred=y_train_pred)
        valid_acc = accuracy_score(y_true=y_valid_label, y_pred=y_valid_pred)
        test_acc = accuracy_score(y_true=y_test_label, y_pred=y_test_pred)
        #
        ckpt = {
            "micro_f1": -1,
            "macro_f1": -1,
            "train_acc": train_acc,
            "valid_acc": valid_acc,
            "test_acc": test_acc,
        }
        self.classifier = classifier
        return ckpt
