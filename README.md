# YOLO-Trained-Models

This guide explains how to train your own YOLO model using the Open Images dataset.

## Step 1: Choose Classes from Open Images Dataset

You can pick from any of the 600 classes in the Open Images dataset to train your model. The list of classes can be found [here](https://github.com/dusty-mv/pytorch-ssd/blob/master/open_images_classes.txt). You can also visually browse the dataset [here](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=-%2Fm%2F031b6r).

## Step 2: Download Images

Use the following command to download images. Replace `class-name` with your desired class from the list provided above.

```bash
python3 open_images_downloader.py --max-images=3000 --class-names "Human eye" --data=data/eye '''

After downloading images its time to train the model. Using the repo of yolo https://github.com/ultralytics/ultralytics/tree/main?tab=readme-ov-file , we can train the model.
Upload files train, valid and to the roboflox. Export the model as yolo 11 (our case).

Copy the exported files to your working directory as shown:


/dataset/
    /images/
        train/
            image1.jpg
            image2.jpg
            ...
        valid/
            image1.jpg
            image2.jpg
            ...
        test/
            image1.jpg
            image2.jpg
            ...
    /labels/
        train/
            image1.txt
            image2.txt
            ...
        valid/
            image1.txt
            image2.txt
            ...
        test/
            image1.txt
            image2.txt
            ...
    /data.yaml  # Dataset configuration file



    After you have made the nesting exactly the same as above, you can now train your model:

    from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="coco8.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
