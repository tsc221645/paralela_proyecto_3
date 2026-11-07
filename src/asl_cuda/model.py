"""CNN básica del laboratorio: mismas capas y 2,120,413 parámetros para 29 clases."""
import torch
from torch import nn

class BasicCNN(nn.Module):
    def __init__(self, classes=29):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                                      nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1),
                                      nn.ReLU(), nn.MaxPool2d(2))
        self.hidden = nn.Linear(16 * 16 * 64, 128)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(128, classes)
        # Inicialización Glorot del laboratorio, no la predeterminada de PyTorch.
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.features(x)
        # Flatten NHWC como Keras para permitir importar pesos sin permutar Dense.
        x = x.permute(0, 2, 3, 1).contiguous().flatten(1)
        return self.output(self.dropout(torch.relu(self.hidden(x))))
        # CrossEntropyLoss integra log-softmax; softmax solo para probabilidades.
