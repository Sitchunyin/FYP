CENG 4999 - Final Year Project
KWS2401 – Robotic system development

This project is about implementing a rodent detection model for a rodent monitoring system base on YOLOv11
This repository is the code use for the model training

This project is using a custom dataset from Roboflow, including 3507 thermal images, devided into a training set(2454 images), a testing set(351 images) and a validation set(702 images)
This is the link to the chosed dataset:
https://app.roboflow.com/fyp-mouse-dataset/3316base-custom-thermal/3

A few pretrained model is provided in the Pretrained Model file, user can adjust the model path to make use of the pre-trained model.

The basic real-time detection program(realtime_detect.py) and recording program(recording.py) is provide in the repository.

<summary>Installation:</summary>
Install the ultralytics package, including all requirements, in a Python>=3.8 environment with PyTorch>=1.8. By the following code:
```bash
pip install ultralytics
```

then clone the content of this repository to use the custom model.


<summary>Usage:</summary>

User could use the 'model training.ipynb' for custom model training and model testing

<br>
You can use Ultralytics YOLO directly from the Command Line Interface (CLI) with the `yolo` command:

```bash
# Predict using a pretrained YOLO model (e.g., YOLO11n) on an image
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
```


### Python

Ultralytics YOLO can also be integrated directly into your Python projects:

```python
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="coco8.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model
```


