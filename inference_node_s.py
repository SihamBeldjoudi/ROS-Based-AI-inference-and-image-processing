#!/usr/bin/env python3

import rospy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sensor_msgs.msg import Image
from std_msgs.msg import String  # New message type for predictions
from cv_bridge import CvBridge
import cv2
import sys
import numpy as np

"""
AI Inference Script.
This script loads a deep learning model, processes images, and publishes classification results via ROS.
"""

# Dictionary mapping class IDs to labels
class_labels = {
    0: "healthy meat",
    1: "no meat",
    2: "fecal contamination",
    3: "herb contamination"
}

# Definition of a basic block for the ResNet architecture (used in ResNet20)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # First convolutional layer + batch normalization
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Second convolutional layer + batch normalization
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # If dimensions do not match, apply a shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # Forward pass through the block
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add the shortcut connection
        out = F.relu(out)
        return out

# Definition of the ResNet model
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet, self).__init__()
        self.in_channels = 16

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Adding ResNet layers
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # Adaptive pooling + fully connected layer for classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Creates a layer consisting of multiple BasicBlock units."""
        strides = [stride] + [1] * (num_blocks - 1)  # First block with specific stride
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels  # Update input channels for the next block
        return nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the network
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # Flatten feature map
        out = self.fc(out)  # Final prediction
        return out

# Function to create a ResNet20 model instance
def resnet20():
    return ResNet(BasicBlock, [3, 3, 3], num_classes=4)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet20().to(device)

# Load model weights from a file provided as a command-line argument
model.load_state_dict(torch.load(sys.argv[1], map_location=device))
model.eval()  # Set model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32 pixels
    transforms.ToTensor(),  # Convert to a PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
])

# Initialize CvBridge for converting ROS images to OpenCV format
bridge = CvBridge()

# Create a ROS publisher to send predictions as String messages
prediction_pub = rospy.Publisher('/prediction_result', String, queue_size=10)

# Callback function triggered when an image is received from the ROS topic
def callback(img_msg):
    # Convert ROS Image message to OpenCV format
    cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

    # Convert to RGB and apply necessary transformations
    image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image_pil = transforms.ToPILImage()(image_rgb)  # Convert to PIL Image
    image_tensor = transform(image_pil).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)  # Pass image through the model
        _, predicted = torch.max(outputs, 1)  # Get the predicted class index
        class_id = predicted.item()  # Convert to integer

    # Publish the inference result
    prediction_pub.publish(class_labels[class_id])

# Main function for the ROS inference node
def inference_node():
    rospy.init_node('resnet_inference_node', anonymous=True)  # Initialize the ROS node
    rospy.Subscriber("/camera/camera/rgb/image_raw", Image, callback)  # Subscribe to the image topic
    rospy.spin()  # Keep the node running to listen for incoming messages

# Run the inference node if the script is executed directly
if __name__ == "__main__":
    inference_node()
