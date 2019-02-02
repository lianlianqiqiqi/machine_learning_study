import torch
from torch import nn
from torch.nn import functional as f
from torch import optim
from torch.autograd import Variable


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dropout_p=0.5):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])

        layers = zip(hidden_sizes[:-1], hidden_sizes[1:])
        hidden_layers = [nn.Linear(h1, h2) for h1, h2 in layers]
        self.hidden_layers.extend(hidden_layers)

        self.output = nn.Linear(hidden_sizes[-1], output_size)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        for linear in self.hidden_layers:
            x = f.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)
        return f.log_softmax(x, dim=1)


class Model:
    def __init__(self, input_size, output_size, hidden_sizes, criterion, dropout_p=0.5):
        self.model = Network(input_size, output_size, hidden_sizes, dropout_p)
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def validation(self, test_loader):
        test_loss = 0
        accuracy = 0
        self.model.eval()
        for images, labels in iter(test_loader):
            images.resize_(images.size()[0], 28 * 28)
            with torch.no_grad():
                output = self.model.forward(images)
                test_loss += self.criterion(output, labels).item()

                equation = torch.exp(output).max(1)[1] == labels.data
                accuracy += equation.type(torch.FloatTensor).mean()
        return test_loss / len(test_loader), accuracy / len(test_loader)

    def train(self, train_loader, test_loader, print_every=40, epochs=3):
        steps = 0
        running_loss = 0
        self.model.train()
        for epoch in range(epochs):
            for images, labels in train_loader:
                steps += 1
                images.resize_(images.size()[0], 28 * 28)
                self.optimizer.zero_grad()
                output = self.model.forward(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    # test_loss, accuracy = self.validation(test_loader)
                    print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss / print_every))
                running_loss = 0
                self.model.train()
