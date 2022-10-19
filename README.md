# DADS7202_Deep Learning_Hw02

# INTRODUCTION
การตรวจจับจำแนกวัตถุ (Object Detection) โดยเปรียบเทียบ 2 models ระหว่าง Faster R-CNN และ Facebook RetinaNet

## About Dataset
ชุดข้อมูลที่สร้างขึ้นเองทั้งหมดประกอบด้วย Crystal, Nestle, Aquafina ทั้งหมด 1200 รูป ปรับขนาดรูปภาพเป็น 500 * 500 และนำรูปไปทำการ annotation สร้างกล่องขอบเขตของวัตถุเพื่อกำหนดว่าในพื้นที่บริเวณนี้เป็นของวัตถุอะไร บันทึกไฟล์รูปออกมาเป็น .xml (Pascal Voc)

## Model 1: Faster R-CNN (two-stage model)
เทคนิค Faster R-CNN นำข้อมูลผ่านเว็บไซต์ https://app.roboflow.com เพื่อตรวจสอบ อีกทั้งมีการปรับขนาดรูปภาพเป็น 416 * 416 แต่ไม่ได้มีการทำ Data Augmentation เนื่องจากชุดข้อมูลที่เราสร้างมีจำนวนมากและหลากหลายในเรื่องของมุมภาพ แสง การตัดขอบ การหมุน รวมอยู่ในชุดข้อมูล และทำการแบ่งชุดข้อมูลออกเป็น train set 90% (1080 รูป) และ test set 10% (120 รูป)

### Initial model
เชื่อมต่อ google collab กับ google drive ของเรา
```
from google.colab import drive
drive.mount('/content/drive')
```
![1](https://user-images.githubusercontent.com/113499057/196776792-ef0de654-cda7-4817-9f79-8ffd5ac54a05.jpg)

Install tensorflow
```
!pip install tensorflow-gpu
```
Import library และตรวจสอบเวอร์ชันของ tensorflow
```
import tensorflow as tf
print(tf.__version__)
```
![3](https://user-images.githubusercontent.com/113499057/196776889-08b0d782-8da7-4adc-8ae4-ecac858076ea.jpg)

Cloning TFOD 2.0 Github
ตั้งค่า directory และ clone github เพื่อจะได้ object detection model ออกมา
```
cd /content/drive/MyDrive
!git clone https://github.com/tensorflow/models.git
```
![5](https://user-images.githubusercontent.com/113499057/196776972-08ecd031-33dd-4e18-9152-84507b31becd.jpg)

ตั้งค่า directory และติดตั้ง Protocal Buffet เพื่อสร้างท่อในการลำเลียงข้อมูลส่งจากอีกที่หนึ่งไปยังอีกที่หนึ่ง
```
cd /content/drive/MyDrive/models/research
!protoc object_detection/protos/*.proto --python_out=.
```
clone github เพื่อจะได้ติดตั้ง COCO API
```
!git clone https://github.com/cocodataset/cocoapi.git
```
![06](https://user-images.githubusercontent.com/113499057/196777068-921c6d20-ddb3-4d5e-886d-7b14d3162f9f.jpg)

คำสั่ง make เพื่อสร้างและเก็บกลุ่มของโปรแกรมและไฟล์จากต้นทาง
```
!make
cp -r pycocotools /content/drive/MyDrive/models/research
```
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
![9](https://user-images.githubusercontent.com/113499057/196779898-a90f4c60-2141-4bb3-bc52-459ba4ecb395.jpg)
```
cd /content/drive/MyDrive/C_Dads7202/pre-trained-models
```
![10](https://user-images.githubusercontent.com/113499057/196779920-e5641a4c-f2e8-4fe1-8434-be95d9c1969c.jpg)

โหลด pre-trained model จาก [Tensorflow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) และแตกไฟล์
```
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
!tar -xvf faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
```
![11](https://user-images.githubusercontent.com/113499057/196779940-32ad199d-3347-4633-afa9-56fa1cf71d9d.jpg)
![12](https://user-images.githubusercontent.com/113499057/196779955-b4f4f53f-6112-4b6e-963b-2a856088c166.jpg)

**สร้าง train data** แปลงเป็น tfrecord file จะได้ไฟล์ออกออกมาเป็น train.record
```
!python generate_tfrecord.py -x /content/drive/MyDrive/C_Dads7202/images/train -l /content/drive/MyDrive/C_Dads7202/annotations/label_map.pbtxt -o /content/drive/MyDrive/C_Dads7202/annotations/train.record
```
**สร้าง test data** แปลงเป็น tfrecord file จะได้ไฟล์ออกออกมาเป็น test.record
```
!python generate_tfrecord.py -x /content/drive/MyDrive/C_Dads7202/images/test -l /content/drive/MyDrive/C_Dads7202/annotations/label_map.pbtxt -o /content/drive/MyDrive/C_Dads7202/annotations/test.record
```
![13](https://user-images.githubusercontent.com/113499057/196779995-d3043950-e75a-4970-a7f7-3caffcdc0b9a.jpg)

**แก้ไขในไฟล์ pipline.config ใน model ของเราเอง (ดึงเฉพาะส่วน code ที่เราต้องแก้ file path)**

แนะนำให้ปรับขนาดมิติเล็กและใหญ่ที่สุดให้เท่ากับขนาดรูปภาพของเราที่ใช้ (416 * 416)
ในการปรับแก้ไข config จะใช้หลัก ๆ อยู่ 3 ไฟล์ ได้แก่ ckpt-0.index, label_map.txt, train.record, test.record
```
min_dimension: 416
max_dimension: 416

use_bfloat16: false  # แก้จาก true ให้เป็น false (เหมาะกับการใช้ TPU train) เพราะใช้ GPU ในการ run

fine_tune_checkpoint: "/content/drive/MyDrive/C_Dads7202/pre-trained-models/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/checkpoint/ckpt-0"

train_input_reader: {
  label_map_path: "/content/drive/MyDrive/C_Dads7202/annotations/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "/content/drive/MyDrive/C_Dads7202/annotations/train.record"
  }
}

eval_input_reader: {
  label_map_path: "/content/drive/MyDrive/C_Dads7202/annotations/label_map.pbtxt"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "/content/drive/MyDrive/C_Dads7202/annotations/test.record"
  }
}
```
ในการ run model ในครั้งนี้ใช้
- batch size: 4
- num_steps: 10000
- total_steps: 10000 (num_steps = total_steps)
- warmup_steps: 1000 (แนะนำให้ใช้จำนวน 10% ของจำนวนรอบทั้งหมด)


Train model
```
!python model_main_tf2.py --model_dir=/content/drive/MyDrive/C_Dads7202/models/my_frcnn --pipeline_config_path=/content/drive/MyDrive/C_Dads7202/models/my_frcnn/pipeline.config
```
![14](https://user-images.githubusercontent.com/113499057/196784178-89b1034b-883d-46fa-a1c5-7f07f17472da.jpg)

Evaluation แสดงประสิทธิภาพของ model
```
!python model_main_tf2.py --model_dir=/content/drive/MyDrive/C_Dads7202/models/my_frcnn --pipeline_config_path=/content/drive/MyDrive/C_Dads7202/models/my_frcnn/pipeline.config --checkpoint_dir=/content/drive/MyDrive/C_Dads7202/models/my_frcnn
```
![15](https://user-images.githubusercontent.com/113499057/196784195-db0281ff-b149-4d49-901d-2e29a28ba725.jpg)

ใช้ Tensorboard แสดงกราฟค่า loss (ค่าคลาดเคลื่อนในการพยากรณ์)
```
%load_ext tensorboard
%tensorboard --logdir=/content/drive/MyDrive/C_Dads7202/models/my_frcnn
```
![g2_1](https://user-images.githubusercontent.com/113499057/196784215-cb2c1676-bf88-4253-aeca-a4ad28b6b00e.jpg)
![g2_2](https://user-images.githubusercontent.com/113499057/196784235-a5ec3377-1313-403c-9c5f-de398eec4656.jpg)

ทำการบันทึก model ออกมาเก็บไว้
```
!python exporter_main_v2.py --input_type image_tensor --pipeline_config_path /content/drive/MyDrive/C_Dads7202/models/my_frcnn/pipeline.config --trained_checkpoint_dir /content/drive/MyDrive/C_Dads7202/models/my_frcnn --output_directory /content/drive/MyDrive/C_Dads7202/exported_models/my_model
```

### Tuned model
ในการ run model ในครั้งนี้ใช้
- batch size: 8
- num_steps: 20000
- total_steps: 20000 (num_steps = total_steps)
- warmup_steps: 2000 (แนะนำให้ใช้จำนวน 10% ของจำนวนรอบทั้งหมด)

### Comparing between initial model and tuned model of Faster R-CNN
#### Initial Model

#### Tuned Model

### Yolo V5
clone github เพื่อจะได้ติดตั้ง Yolo-V5
```
#clone YOLOv5 and 
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt # install dependencies
%pip install -q roboflow
```
Import library
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
**Run Inference With Trained Weights**
เอาข้อมูล pre-trained model มาทดสอบ
```
!python detect.py --weights /content/yolov5/runs/train/exp2/weights/best.pt --img 416 --conf 0.1 --source /content/datasets/Brands-1/test/images
```
ทดลองนำ model มาใช้กับข้อมูลชุด test
```
import glob
from IPython.display import Image, display
```
```
for imageName in glob.glob('/content/yolov5/runs/detect/exp4/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")
```

### ปัญหาที่พบในระหว่างการ train โดยใช้เทคนิค Faster R-CNN

## Model 2: RetinaNet (one-stage model)
เทคนิค RatinaNet ทำการแบ่งชุดข้อมูลออกเป็น train set 70% (840 รูป) validation set 20% (240 รูป) และ test set 10% (120 รูป)

### Initial model

### Tuned model

### Comparing between initial model and tuned model of RetinaNet

## Comparing between Faster R-CNN and RetinaNet
|         | Faster R-CNN | RetinaNet |
|---------|--------------|-----------|
| Chrome  | yes          | yes       |
| Firefox | yes          | yes       |
| Edge    | no           | yes       |

## Reference
[1] (2022) Data preprocessing from https://app.roboflow.com

[2] (2022) Pre-trained model of Faster R-CNN and RetinaNet from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

## End Credit
งานชิ้นนี้เป็นส่วนหนึ่งของวิชา DADS7202 Deep Learning หลักสูตรวิทยาศาสตร์มหาบัณฑิต คณะสถิติประยุกต์ สถาบันบัณฑิตพัฒนบริหารศาสตร์

กลุ่ม สู้ DL แต่ DL สู้กลับ 
สมาชิก: (1) 641xxxxx03 (2) 641xxxxx06 (3) 641xxxxx13 (4) 641xxxxx20
