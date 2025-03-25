# ROS-Based AI Inference & Image Processing  

This project implements a real-time AI inference and image processing pipeline using ROS and OpenCV. It utilizes a ResNet20 model trained for meat image classification and detects red zones in images.  

## Features  
### AI Model Training  
- **ResNet20 architecture** adapted for four-class classification  
- **Data augmentation** to improve generalization  
- **Adam optimizer with learning rate scheduling**  
- **Training visualization** using loss and accuracy curves
  
### AI Inference with ResNet20  
- **Classification model using inference** categorizing images into four classes:  
  - Healthy meat  
  - No meat  
  - Fecal contamination  
  - Herb contamination  
- **ROS Integration**:  
  - Subscribes to `/camera/camera/rgb/image_raw` to receive images  
  - Publishes classification results to `/prediction_result`  

### Red Zone Detection and Annotation  
- Detects **red zones** in the image (potential contamination or anomalies)  
- Publishes:  
  - The **annotated image** (`ai/processed_image`)  
  - The **coordinates of the red zone center** (`ai/red_zone_center`)  
- Integrates with inference to display classification results on the image

## Example Outputs  

### 1. AI Inference Result  
![AI Inference](images/1.png)  

### 2. Zone Detection  
![Zone detection](images/2.png)  

### 3. Training Loss & Accuracy Cu
![Zone detection](images/3.png)  

## Installation & Usage  

### Prerequisites  
- Python 3  
- ROS (Robot Operating System)  
- PyTorch  
- OpenCV  
- torchvision  

### Install Dependencies  
```bash
pip install torch torchvision opencv-python numpy rospkg
```

### Start the Pipeline  

1. **Launch ROS**  
```bash
roscore &
```

2. **Run AI Inference with ResNet20**  
```bash
python3 inference.py <path_to_model_weights.pth>
```

3. **Run Image Processing**  
```bash
python3 image_processor.py
```

### Data Flow  
```
/camera/camera/rgb/image_raw ➡ AI Inference (ResNet20) ➡ /prediction_result ➡  
Red Zone Detection & Annotation ➡ /ai/processed_image + /ai/red_zone_center
```

## Code Structure  
- **`Mymodel.py`**: Trains a ResNet20 model and saves the weights
- **`inference.py`**: Loads the model, performs inference, and publishes results  
- **`image_processor.py`**: Detects red zones, annotates the image, and publishes results  




