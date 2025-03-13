#!/usr/bin/env python3

import rospy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sensor_msgs.msg import Image
from std_msgs.msg import String  # Nouveau message pour la prédiction
from cv_bridge import CvBridge
import cv2
import sys
import numpy as np

"""
Main file for inference with AI
This file contains AI models used and publish predictions
"""

class_labels = {
    0: "viande saine",
    1: "pas de viande",
    2: "souillure fecale",
    3: "souillure d'herbiere"
}


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
    def __init__(self, block, num_blocks, num_classes=4):
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

# Charger le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet20().to(device)
model.load_state_dict(torch.load(sys.argv[1], map_location=device))
model.eval()

# Transformation d'image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

bridge = CvBridge()

# Publisher pour la prédiction
prediction_pub = rospy.Publisher('/prediction_result', String, queue_size=10)

def callback(img_msg):
    #rospy.loginfo("Image reçue, exécution de l'inférence...")

    # Convertir l'image ROS en OpenCV
    cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

    # Convertir en RGB et appliquer les transformations
    image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image_pil = transforms.ToPILImage()(image_rgb)
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Inférence
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_id = predicted.item()

    # Publier la prédiction
    prediction_pub.publish(class_labels[class_id])
    #rospy.loginfo(f"Classe prédite: {class_id} ({class_labels[class_id]})")

def inference_node():
    rospy.init_node('resnet_inference_node', anonymous=True)
    rospy.Subscriber("/camera/camera/rgb/image_raw", Image, callback)
    rospy.spin()

if __name__ == "__main__":
    inference_node()







