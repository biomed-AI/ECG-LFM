import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients, LayerIntegratedGradients, DeepLiftShap, GradientShap, GuidedGradCam
from pytorch_grad_cam import GradCAM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

vocab_size = 1024
embedding_dim = 32
seq_len = 128
num_classes = 1
hidden_dim = 256

class predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim 
        self.vocab_size, self.embedding_dim = vocab_size, embedding_dim

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.linear = nn.Linear(self.seq_len*self.embedding_dim, self.num_classes)
        self.linear2 = nn.Linear(self.seq_len, self.num_classes)

    def forward(self, x):
        print(x)
        #x = self.linear2(x)
        x = self.embedding(x)
        x = x.reshape(-1, self.seq_len*self.embedding_dim)
        x = F.relu(self.linear(x)) + torch.sum(x)
        return x

class wrapper_predictor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        x = self.model(x)
        x = F.softmax(x, dim=1)
        return x

model = predictor().to(device)
wrapper_model = wrapper_predictor(model).to(device)

#indexes = torch.Tensor(np.random.randint(0, vocab_size, (1, seq_len)), requires_grad=True).to(device)
indexes = torch.randint(0, vocab_size, (1, 128), dtype=torch.int).long().to(device)
baseline = torch.zeros(1, seq_len, requires_grad=True).to(device)
print(indexes.shape, model(indexes))


target_layers = [model.embedding]


# lig = LayerIntegratedGradients(model, model.embedding)
# attributions, delta = lig.attribute(inputs=indexes, target=0, n_steps=1, return_convergence_delta=True)
# print(attributions.shape)
print(indexes)
ig = IntegratedGradients(model)
attributions = ig.attribute(inputs=indexes, target=0)
print(attributions)