# DADS7202_Deep Learning_Hw02

# INTRODUCTION
การตรวจจับจำแนกวัตถุ (Object Detection) โดยเปรียบเทียบ 2 models ระหว่าง Faster R-CNN และ Facebook RetinaNet

## About Dataset
ชุดข้อมูลที่สร้างขึ้นเองทั้งหมดประกอบด้วย Crystal, Nestle, Aquafina ทั้งหมด 1200 รูป ปรับขนาดรูปภาพเป็น 500*500 และนำรูปไปทำการ annotation สร้างกล่องขอบเขตของวัตถุเพื่อกำหนดว่าในพื้นที่บริเวณนี้เป็นของวัตถุอะไร บันทึกไฟล์รูปออกมาเป็น .xml (Pascal Voc)

## Model 1: Faster R-CNN (two-stage model)
เทคนิค Faster R-CNN นำข้อมูลผ่านเว็บไซต์ https://app.roboflow.com เพื่อตรวจสอบ อีกทั้งมีการปรับขนาดรูปภาพเป็น 416*416 แต่ไม่ได้มีการทำ Data Augmentation เนื่องจากชุดข้อมูลที่เราสร้างมีจำนวนมากและหลากหลายในเรื่องของมุมภาพ แสง การตัดขอบ การหมุน รวมอยู่ในชุดข้อมูล และทำการแบ่งชุดข้อมูลออกเป็น train set 90% (1080 รูป) และ test set 10% (120 รูป)

### Initial model
เชื่อมต่อ google collab กับ google drive ของเรา
```
from google.colab import drive
drive.mount('/content/drive')
```
Install tensorflow
```
!pip install tensorflow-gpu
```
Import library และตรวจสอบเวอร์ชันของ tensorflow
```
import tensorflow as tf
print(tf.__version__)
```
Cloning TFOD 2.0 Github
ตั้งค่า directory และ clone github เพื่อจะได้ object detection model ออกมา
```
cd /content/drive/MyDrive
!git clone https://github.com/tensorflow/models.git
```
ตั้งค่า directory และติดตั้ง Protocal Buffet เพื่อสร้างท่อในการลำเลียงข้อมูลส่งจากอีกที่หนึ่งไปยังอีกที่หนึ่ง
```
cd /content/drive/MyDrive/models/research
!protoc object_detection/protos/*.proto --python_out=.
```
clone github เพื่อจะได้ติดตั้ง COCO API
```
!git clone https://github.com/cocodataset/cocoapi.git
```
คำสั่ง make เพื่อสร้างและเก็บกลุ่มของโปรแกรมและไฟล์จากต้นทาง
```
!make
cp -r pycocotools /content/drive/MyDrive/models/research
```
### Install the Object Detection API
ติดตั้ง the Object Detection API
```
cp object_detection/packages/tf2/setup.py .
```
ติดตั้ง Python
```
!python -m pip install .
```
```
!python object_detection/builders/model_builder_tf2_test.py
```
```
cd /content/drive/MyDrive/C_Dads7202/pre-trained-models
```
โหลด pre-model จาก Tensorflow 2 Detection Model Zoo
```
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
!tar -xvf faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
```
##### สร้าง train data:
```
!python generate_tfrecord.py -x /content/drive/MyDrive/C_Dads7202/images/train -l /content/drive/MyDrive/C_Dads7202/annotations/label_map.pbtxt -o /content/drive/MyDrive/C_Dads7202/annotations/train.record
```
##### สร้าง test data:
```
!python generate_tfrecord.py -x /content/drive/MyDrive/C_Dads7202/images/test -l /content/drive/MyDrive/C_Dads7202/annotations/label_map.pbtxt -o /content/drive/MyDrive/C_Dads7202/annotations/test.record
```
แนะนำให้ปรับให้เท่ากับขนาดรูปภาพของเราที่ใช้
min_dimension: 416
max_dimension: 416

use_bfloat16: false  # แก้จาก true ให้เป็น false เพราะใช้ GPU ในการ run

### แก้
num_steps: 10000
total_steps: 10000 (num_steps = total_steps)
warmup_steps: 1000 (แนะนำให้ใช้จำนวน 10% ของจำนวนรอบทั้งหมด)
```
!python model_main_tf2.py --model_dir=/content/drive/MyDrive/C_Dads7202/models/my_frcnn --pipeline_config_path=/content/drive/MyDrive/C_Dads7202/models/my_frcnn/pipeline.config
```
```
!python model_main_tf2.py --model_dir=/content/drive/MyDrive/C_Dads7202/models/my_frcnn --pipeline_config_path=/content/drive/MyDrive/C_Dads7202/models/my_frcnn/pipeline.config --checkpoint_dir=/content/drive/MyDrive/C_Dads7202/models/my_frcnn
```

```
%load_ext tensorboard
%tensorboard --logdir=/content/drive/MyDrive/C_Dads7202/models/my_frcnn
```
```
!python exporter_main_v2.py --input_type image_tensor --pipeline_config_path /content/drive/MyDrive/C_Dads7202/models/my_frcnn/pipeline.config --trained_checkpoint_dir /content/drive/MyDrive/C_Dads7202/models/my_frcnn --output_directory /content/drive/MyDrive/C_Dads7202/exported_models/my_model
```



จำนวนรอบของการ train = 5000
batch size = 8

### Tuned model
จำนวนรอบของการ train = 10000
batch size = 8

### Comparing between initial model and tuned model of Faster R-CNN

### Yolo V5
clone github เพื่อจะได้ติดตั้ง YOlo-V5
```
#clone YOLOv5 and 
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt # install dependencies
%pip install -q roboflow
```

```
import torch
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
```
##### ตั้งค่า environment
```
os.environ["DATASET_DIRECTORY"] = "/content/datasets"1
```
ติดตั้ง roboflow
```
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="5O37kFrhuKlGI2WuGhvK")
project = rf.workspace("national-institute-of-development-administration-no8yz").project("brands-tcgnz")
dataset = project.version(1).download("yolov5")
```
กำหนดขนาดของรูปภาพ(416),ขนาดของ batch(16),กำหนดจำนวนรอบ(150) แล้ว Train 
```
!python train.py --img 416 --batch 16 --epochs 150 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
```
## Run Inference With Trained Weights
เอาข้อมูล pre-train มาทดสอบ
```
!python detect.py --weights /content/yolov5/runs/train/exp2/weights/best.pt --img 416 --conf 0.1 --source /content/datasets/Brands-1/test/images
```

##### display inference on ALL test images
```
import glob
from IPython.display import Image, display
```
```
for imageName in glob.glob('/content/yolov5/runs/detect/exp4/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")
```


## Model 2: RetinaNet (one-stage model)
เทคนิค RatinaNet ทำการแบ่งชุดข้อมูลออกเป็น train set 70% (840 รูป) validation set 20% (240 รูป) และ test set 10% (120 รูป)

### Initial model

### Tuned model

### Comparing between initial model and tuned model of RetinaNet

## Comparing between Faster R-CNN and RetinaNet

## Reference
[1] (2022) Data preprocessing from https://app.roboflow.com

[2] (2022) Pre-trained model of Faster R-CNN and RetinaNet from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

## End Credit
งานชิ้นนี้เป็นส่วนหนึ่งของวิชา DADS7202 Deep Learning หลักสูตรวิทยาศาสตร์มหาบัณฑิต คณะสถิติประยุกต์ สถาบันบัณฑิตพัฒนบริหารศาสตร์

กลุ่ม สู้ DL แต่ DL สู้กลับ 
สมาชิก: (1) 641xxxxx03 (2) 641xxxxx06 (3) 641xxxxx13 (4) 641xxxxx20
