import rospy
import rospkg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

"""
Used for training AI models
Not launched with ROS
"""

rospy.init_node('Mymodel_graphs', anonymous=True)

# Fonction pour afficher les courbes d'apprentissage
def learning_curves(tr_losses, val_losses, tr_accs, val_accs):
    plt.figure(figsize=(10, 5))
    #plt.plot(tr_losses, label="Train Loss", c="red")
    plt.plot(val_losses, label="Validation Loss", c="blue")
    plt.xlabel("Epochs"); plt.ylabel("Loss Values")
    plt.xticks(ticks=np.arange(len(tr_losses)), labels=[i for i in range(1, len(tr_losses) + 1)])
    plt.legend(); plt.show()

    plt.figure(figsize=(10, 5))
    #plt.plot(tr_accs, label="Train Accuracy", c="orangered")
    plt.plot(val_accs, label="Validation Accuracy", c="darkgreen")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy Scores")
    plt.xticks(ticks=np.arange(len(tr_accs)), labels=[i for i in range(1, len(tr_accs) + 1)])
    plt.legend(); plt.show()

# Définir le modèle ResNet20
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):  # Adapter pour 4 classes
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def resnet20():
    return ResNet(BasicBlock, [3, 3, 3], num_classes=4)

# Préparer les données
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data_dir = rospkg.RosPack().get_path("simulation_ur3e")+"/dataset/pieces_latex_simulation/"
trainset = datasets.ImageFolder(root=os.path.join(data_dir, "Train"), transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = datasets.ImageFolder(root=os.path.join(data_dir, "Test"), transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Initialisation du modèle et de l'optimiseur
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet20().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Liste pour stocker les pertes et précisions pour chaque époque
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Boucle d'entraînement
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calcul de la précision sur le jeu d'entraînement
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    scheduler.step()
    epoch_train_loss = running_loss / len(trainloader)
    epoch_train_acc = 100 * correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # Évaluation sur les données de test
    model.eval()
    correct_test = 0
    total_test = 0
    running_val_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    epoch_val_loss = running_val_loss / len(testloader)
    epoch_val_acc = 100 * correct_test / total_test
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.2f}%")
    print(f"Test Accuracy: {epoch_val_acc:.2f}%, Test Loss: {epoch_val_loss:.4f}")

# Afficher les courbes d'apprentissage après l'entraînement
learning_curves(train_losses, val_losses, train_accuracies, val_accuracies)

# Enregistrer le modèle final
torch.save(model.state_dict(), rospkg.RosPack().get_path("simulation_ur3e")+"/AI_models/model_Resnet2_gazebo_simulation.pth")
print("Modèle enregistré correctement !")

