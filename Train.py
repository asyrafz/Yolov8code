import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # at the beginning of the script
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # build a new model from YAML
    #model = YOLO('yolov8n-seg.yaml')

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')
    #model = YOLO('yolov8n-seg.pt')

    # Move the model to the appropriate device (e.g., GPU)
    model.to(device)

    # Train the model using the 'data.yaml' dataset. Modify the data path & other parameters accordingly
    results = model.train(data='_location_data.yaml', epochs=90, imgsz=900, plots=True)

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the mode. Replace 'path_to_image.jpg' with the actual image path or file for multiple file
    results = model('path_to_image.jpg')

    # Export the model to ONNX format
    success = model.export(format='onnx')

#The result will save in runs/detect/train[n] or runs/segment/train[n]
