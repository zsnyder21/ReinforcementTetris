import torch.nn as nn

class TetrisNetwork(nn.Module):
    def __init__(self):
        super(TetrisNetwork, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(64, 1), nn.ReLU(inplace=True))

        self._createWeights()

    def _createWeights(self) -> None:
        """
        Create the weights for the various layers of the network

        :return: None
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: list) -> list:
        """
        Run the network forward

        :param x: Input to the first layer of the neural net
        :return: Output of the final neural net
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
