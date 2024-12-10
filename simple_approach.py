import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=1, num_classes=10):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, batch_first=True),
            num_layers=2
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


def load_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def token_entropy_reduction(inputs, threshold=0.5, min_tokens=1):
    """
    Apply entropy-based reduction to input tokens.
    inputs: (B, seq_len) or (B, seq_len, 1)
    """
    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(-1)

    entropy = inputs.var(dim=-1, keepdim=True) 
    mask = entropy.squeeze(-1) > threshold

    reduced_inputs = []
    for i in range(inputs.size(0)):
        high_entropy_tokens = inputs[i][mask[i]]
        if high_entropy_tokens.size(0) < min_tokens:
            high_entropy_tokens = inputs[i][:min_tokens]
        reduced_inputs.append(high_entropy_tokens)

    max_len = max(seq.size(0) for seq in reduced_inputs)
    padded_inputs = torch.zeros(len(reduced_inputs), max_len, inputs.size(-1), device=inputs.device)
    for i, seq in enumerate(reduced_inputs):
        padded_inputs[i, :seq.size(0)] = seq

    return padded_inputs


def measure_efficiency(model, loader, device):
    total_time, total_flops = 0, 0
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            inputs = inputs.view(inputs.size(0), -1, 1)
            start_time = time.time()
            _ = model(inputs)
            total_time += time.time() - start_time
            total_flops += inputs.numel() * model.classifier.in_features
    avg_time = total_time / len(loader.dataset)
    avg_flops = total_flops / len(loader.dataset)
    return avg_time, avg_flops


def train_model(model, loader, optimizer, criterion, device, input_reduction=False, threshold=0.5):
    model.train()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.view(inputs.size(0), -1)
        if input_reduction:
            inputs = token_entropy_reduction(inputs, threshold)
        else:
            inputs = inputs.unsqueeze(-1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def evaluate_model(model, loader, criterion, device, input_reduction=False, threshold=0.5):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            if input_reduction:
                inputs = token_entropy_reduction(inputs, threshold)
            else:
                inputs = inputs.unsqueeze(-1)

            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
            correct += (outputs.argmax(1) == targets).sum().item()
    accuracy = 100 * correct / len(loader.dataset)
    return total_loss / len(loader), accuracy


train_loader, test_loader = load_mnist_data()
model = SimpleTransformer(input_dim=1, num_classes=10)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
input_flag = False
for epoch in range(5):
    train_model(model, train_loader, optimizer, criterion, device=device, input_reduction=input_flag, threshold=0.5)
    loss, accuracy = evaluate_model(model, test_loader, criterion, device=device, input_reduction=input_flag, threshold=0.5)
    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

avg_time, avg_flops = measure_efficiency(model, test_loader, device)
print(f"Efficiency: Avg Time per Sample: {avg_time:.6f}s, Avg FLOPs per Sample: {avg_flops}")


