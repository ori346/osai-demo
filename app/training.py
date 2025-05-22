import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28x28 → 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                # 28x28 → 14x14

            nn.Conv2d(32, 64, 3, padding=1),  # 14x14 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 14x14 → 7x7

            nn.Flatten(),                   # 64*7*7 = 3136
            nn.Linear(3136, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    train_data = datasets.MNIST(root='.', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='.', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    model = CNNModel()
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Train
    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete")

    # 5. Save
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Model saved to mnist_model.pth")
