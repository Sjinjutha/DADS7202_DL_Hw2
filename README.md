# DADS7202_Deep Learning_Hw02

# INTRODUCTION
การตรวจจับจำแนกวัตถุ (Object Detection) โดยเปรียบเทียบ 2 models ระหว่าง Faster R-CNN และ Facebook RetinaNet

## About Dataset
ชุดข้อมูลที่สร้างขึ้นเองทั้งหมดประกอบด้วย Crystal, Nestle, Aquafina ทั้งหมด 1200 รูป ปรับขนาดรูปภาพเป็น 500 * 500 และนำรูปไปทำการ annotation สร้างกล่องขอบเขตของวัตถุเพื่อกำหนดว่าในพื้นที่บริเวณนี้เป็นของวัตถุอะไร บันทึกไฟล์รูปออกมาเป็น .xml (Pascal Voc)

## Model 1: Faster R-CNN (two-stage model)
เทคนิค Faster R-CNN นำข้อมูลผ่านเว็บไซต์ https://app.roboflow.com เพื่อตรวจสอบ อีกทั้งมีการปรับขนาดรูปภาพเป็น 416 * 416 แต่ไม่ได้มีการทำ Data Augmentation เนื่องจากชุดข้อมูลที่เราสร้างมีจำนวนมากและหลากหลายในเรื่องของมุมภาพ แสง การตัดขอบ การหมุน รวมอยู่ในชุดข้อมูล และทำการแบ่งชุดข้อมูลออกเป็น train set 90% (1080 รูป) และ test set 10% (120 รูป)

### Initial model
เชื่อมต่อ google collab กับ google drive ของเรา เนื่องจากการ runtime ในแต่ละครั้งจะไม่ได้มีการบันทึกข้อมูลที่เราทำไว้ จึงทำการเชื่อมต่อกับ google drive ของเราเพื่อเก็บบันทึกข้อมูล
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
![p1](https://user-images.githubusercontent.com/113499057/196954087-f9475a7e-8226-41da-b050-a9a7d3957369.jpg)

![5](https://user-images.githubusercontent.com/113499057/196776972-08ecd031-33dd-4e18-9152-84507b31becd.jpg)

หลังจาก คำสั่ง clone จะได้โฟลเดอร์ models ขึ้นมาบน Drive

ตั้งค่า directory และติดตั้ง Protocal Buffet เพื่อสร้างท่อในการลำเลียงข้อมูลส่งจากอีกที่หนึ่งไปยัง research folder
```
cd /content/drive/MyDrive/models/research
!protoc object_detection/protos/*.proto --python_out=.
```
![p2](https://user-images.githubusercontent.com/113499057/196954091-b4cbf027-0165-4768-ad01-e9bc7ce71c48.jpg)

Clone github เพื่อติดตั้ง COCO API
```
!git clone https://github.com/cocodataset/cocoapi.git
```
![06](https://user-images.githubusercontent.com/113499057/196777068-921c6d20-ddb3-4d5e-886d-7b14d3162f9f.jpg)
![p3](https://user-images.githubusercontent.com/113499057/196954073-3087a72a-3641-43a0-9029-830da89807a5.jpg)

หลังจาก Clone จะได้ cocoapi folder

```
cd /content/drive/MyDrive/models/research/cocoapi/PythonAPI
```
![p4](https://user-images.githubusercontent.com/113499057/196954079-3a49f77c-f12e-4de5-8853-83086d28e74d.jpg)

เลือก directory มาที่ PythonAPI folder และ run คำสั่ง make เพื่อสร้างและเก็บกลุ่มของโปรแกรมและไฟล์จากต้นทาง และคัดลอกไฟล์ python
```
!make
cp -r pycocotools /content/drive/MyDrive/models/research
```
ติดตั้ง the Object Detection API
ตั้ง directory ตามเส้นทางข้างล่างเพื่อคัดลอกไฟล์ setup.py ลงไป
```
cd /content/drive/MyDrive/models/research
```
```
cp object_detection/packages/tf2/setup.py .
```
![p5](https://user-images.githubusercontent.com/113499057/196954082-7f65073f-a830-4222-a2b0-095e9e7eae20.jpg)

```
!python -m pip install .
```
โดย (.) นี้คือการติดตั้ง library ทั้งหมดจาก research folder
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

**สร้าง label_map.pbtxt** เพื่อเป็นการนิยาม classes ที่เราต้องการตรวจจับแล้วนำมาเก็บใน annotation folder เพื่อนำไปสร้างไฟล์ .record ในขั้นตอนต่อไป

![label](https://user-images.githubusercontent.com/113499057/196919960-06032020-f934-46ac-81bf-845b1b217190.jpg)

**นำ train data** แปลงเป็น tfrecord file เพื่อนำชุดข้อมูลรูปภาพชุด train ของเราทั้ง .jpg และ .xml บีบอัดเพื่อที่สามรถนำเข้าไปใช้ใน model ได้ และจะได้ไฟล์ออกออกมาเป็น train.record
```
!python generate_tfrecord.py -x /content/drive/MyDrive/C_Dads7202/images/train -l /content/drive/MyDrive/C_Dads7202/annotations/label_map.pbtxt -o /content/drive/MyDrive/C_Dads7202/annotations/train.record
```
**นำ test data** แปลงเป็น tfrecord file เพื่อนำชุดข้อมูลรูปภาพชุด test ของเราทั้ง .jpg และ .xml บีบอัดเพื่อที่สามรถนำเข้าไปใช้ใน model ได้ และจะได้ไฟล์ออกออกมาเป็น test.record
```
!python generate_tfrecord.py -x /content/drive/MyDrive/C_Dads7202/images/test -l /content/drive/MyDrive/C_Dads7202/annotations/label_map.pbtxt -o /content/drive/MyDrive/C_Dads7202/annotations/test.record
```
![13](https://user-images.githubusercontent.com/113499057/196779995-d3043950-e75a-4970-a7f7-3caffcdc0b9a.jpg)

**แก้ไขในไฟล์ pipline.config ใน model ของเราเอง (ดึงเฉพาะส่วน code ที่เราต้องแก้ file path)**

แนะนำให้ปรับขนาดมิติเล็กและใหญ่ที่สุดให้เท่ากับขนาดรูปภาพของเราที่ใช้ (416 * 416)
ในการปรับแก้ไข config จะใช้หลัก ๆ อยู่ 4 ไฟล์ ได้แก่
* ckpt-0.index คือไฟล์ที่ไว้เก็บค่าของ pre-trained model ที่เรานำมาใช้
* label_map.txt
* train.record
* test.record
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


**Train model**
```
!python model_main_tf2.py --model_dir=/content/drive/MyDrive/C_Dads7202/models/my_frcnn --pipeline_config_path=/content/drive/MyDrive/C_Dads7202/models/my_frcnn/pipeline.config
```
![14](https://user-images.githubusercontent.com/113499057/196784178-89b1034b-883d-46fa-a1c5-7f07f17472da.jpg)

จากการคำสั่งการ train ข้างต้นจะสั่งให้บอกค่า loss ออกมาในทุก ๆ 100 steps โดยแต่ละ step จะใช้เวลาคำนวน 0.097 วินาที หลังจากนั้นเราจะใช้ค่าที่ได้จากการ train นี้มาประเมิน model ในขั้นตอนต่อไป ซึ่งเก็บอยู่ checkpoint file

**Evaluation แสดงประสิทธิภาพของ model**
```
!python model_main_tf2.py --model_dir=/content/drive/MyDrive/C_Dads7202/models/my_frcnn --pipeline_config_path=/content/drive/MyDrive/C_Dads7202/models/my_frcnn/pipeline.config --checkpoint_dir=/content/drive/MyDrive/C_Dads7202/models/my_frcnn
```
![15](https://user-images.githubusercontent.com/113499057/196784195-db0281ff-b149-4d49-901d-2e29a28ba725.jpg)

จากรูปจะได้ว่า model ของเรานั้น
|    IoU    | Precision |
|-----------|-----------|
| 0.50:0.95 | 0.439     |
| 0.50      | 0.861     |
| 0.75      | 0.428     |

ใช้ Tensorboard แสดงกราฟค่า loss (ค่าคลาดเคลื่อนในการพยากรณ์)
```
%load_ext tensorboard
%tensorboard --logdir=/content/drive/MyDrive/C_Dads7202/models/my_frcnn
```
![t_22](https://user-images.githubusercontent.com/113499057/196929032-620c2e2f-ee82-4019-bc3a-cce5c8fe9e39.jpg)

จากกราฟเป็นค่าความผิดพลาด (loss) จากการ train เมื่อเวลาผ่านไปความผิดพลาดยิ่งน้อยลง หากได้ทําการ train มากยิ่งขึ้นโปรแกรมจะมีความแม่นยํามากขึ้น (ค่าเข้าใกล้ 0) ในการ train รอบต่อไปจึงได้มีการเพิ่มจำนวนรอบ (num_steps) ให้มากขึ้น

ทำการบันทึก model ออกมาเก็บไว้
```
!python exporter_main_v2.py --input_type image_tensor --pipeline_config_path /content/drive/MyDrive/C_Dads7202/models/my_frcnn/pipeline.config --trained_checkpoint_dir /content/drive/MyDrive/C_Dads7202/models/my_frcnn --output_directory /content/drive/MyDrive/C_Dads7202/exported_models/my_model
```
![17](https://user-images.githubusercontent.com/113499057/196785662-f4aa093d-f408-4a27-bf6a-2c452c2933e2.jpg)

ผลลัพธ์ของการ Run Model จากการทดลองใช้ข้อมูลชุด test ในรูปภาพที่ 600.jpg
- batch size: 4
- num_steps: 10000
- total_steps: 10000
- warmup_steps: 1000

จะเห็นว่า model ยังไม่ดีเท่าที่ควร เพราะไม่สามารถตรวจจับวัตถุได้ครบหมดทุกตำแหน่งที่มี จึงได้มีการปรับ hyperparameter ให้ model ในรอบต่อไป
![16](https://user-images.githubusercontent.com/113499057/196785683-8d1d8146-4c0c-43dd-9a25-76c2a882c69f.jpg)

### Tuned model
ในการ run model ในครั้งนี้ใช้ (แก้ไขที่ไฟล์ pipeline.config)
- batch size: 8
- num_steps: 20000
- total_steps: 20000 (num_steps = total_steps)
- warmup_steps: 2000 (แนะนำให้ใช้จำนวน 10% ของจำนวนรอบทั้งหมด)

แต่ยังคงใช้ไฟล์อื่น ๆ เหมือนเดิม คือ ckpt-0.index, label_map.txt, train.record, test.record

#### Train Model

![18](https://user-images.githubusercontent.com/113499057/196789197-6cc69c8f-1594-4649-a4be-fdf4b51bac9c.jpg)

#### Evaluation แสดงประสิทธิภาพของ model

![19](https://user-images.githubusercontent.com/113499057/196789205-f91017ad-e16f-4fda-90bd-a3bc4360d480.jpg)

จากรูปจะได้ว่า model ของเรานั้น
|    IoU    | Precision |
|-----------|-----------|
| 0.50:0.95 | 0.455     |
| 0.50      | 0.900     |
| 0.75      | 0.454     |


ใช้ Tensorboard แสดงกราฟค่า loss (ค่าคลาดเคลื่อนในการพยากรณ์)

![t_3](https://user-images.githubusercontent.com/113499057/196929052-65c41b69-4b18-4d97-844f-0aa38af6e0c5.jpg)

![20](https://user-images.githubusercontent.com/113499057/196789208-85f177a2-64ef-4051-a67c-7ad05917c773.jpg)


### Comparing between initial model and tuned model of Faster R-CNN
#### Initial Model

![t_22](https://user-images.githubusercontent.com/113499057/196929032-620c2e2f-ee82-4019-bc3a-cce5c8fe9e39.jpg)

![t_3](https://user-images.githubusercontent.com/113499057/196929052-65c41b69-4b18-4d97-844f-0aa38af6e0c5.jpg)

![16](https://user-images.githubusercontent.com/113499057/196785683-8d1d8146-4c0c-43dd-9a25-76c2a882c69f.jpg)

![20](https://user-images.githubusercontent.com/113499057/196789208-85f177a2-64ef-4051-a67c-7ad05917c773.jpg)

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
![y1](https://user-images.githubusercontent.com/113499057/196791465-12eb4ec9-ae57-480b-b8ca-adceeca7833c.jpg)

ตั้งค่า environment
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
![y2](https://user-images.githubusercontent.com/113499057/196791477-bb28c04b-c420-47bd-bd1b-f43f9c66f9c5.jpg)

กำหนดขนาดของรูปภาพ(416),ขนาดของ batch(16),กำหนดจำนวนรอบ(150) แล้ว Train 
```
!python train.py --img 416 --batch 16 --epochs 150 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
```
![y3](https://user-images.githubusercontent.com/113499057/196791483-74ddeb93-6805-43fc-952a-8af9e09295e1.jpg)

Evaluation แสดงประสิทธิภาพของ model
```
%load_ext tensorboard
%tensorboard --logdir runs
```
![y4](https://user-images.githubusercontent.com/113499057/196791486-57f784fc-da25-4348-87dc-5cccd6d9f8e7.jpg)

จากกราฟจะเห็นได้ว่า ค่าสัดส่วน metrics/mAP มีค่าเพิ่มขึ้น ค่ายิ่งมากยิ่งแสดงถึงความแม่นยำ

**Run Inference With Trained Weights**
เอาข้อมูล pre-trained model มาทดสอบ
```
!python detect.py --weights /content/yolov5/runs/train/exp2/weights/best.pt --img 416 --conf 0.1 --source /content/datasets/Brands-1/test/images
```

![y9](https://user-images.githubusercontent.com/113499057/196794863-971aa523-f62d-410b-8bd5-deb66ac36420.jpg)

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
![y5](https://user-images.githubusercontent.com/113499057/196791488-0b5dde15-a28d-4d9d-a882-645fba322384.jpg)

![y6](https://user-images.githubusercontent.com/113499057/196791492-7afa26aa-4392-49d5-a3a6-04f008ea16e8.jpg)

![16](https://user-images.githubusercontent.com/113499057/196785683-8d1d8146-4c0c-43dd-9a25-76c2a882c69f.jpg)

![y7](https://user-images.githubusercontent.com/113499057/196791499-3c1a43d6-df6f-442d-87ac-e3c560d5762a.jpg)


### ปัญหาที่พบในระหว่างการ train โดยใช้เทคนิค Faster R-CNN

## Model 2: RetinaNet (one-stage model)
เทคนิค RatinaNet ทำการแบ่งชุดข้อมูลออกเป็น train set 70% (840 รูป) validation set 20% (240 รูป) และ test set 10% (120 รูป)

### Initial model

### Tuned model

### Comparing between initial model and tuned model of RetinaNet

## Comparing between Faster R-CNN and RetinaNet
|                                   | batch size |  num steps |  epochs |  GPU ที่ใช้ประมวลผล  | Precision  |
|-----------------------------------|------------|------------|---------|-------------------|------------|
| Faster R-CNN ResNet50 V1 640x640  | 4          | 10000      | 1       |                   |            |
| Faster R-CNN ResNet50 V1 640x640  | 8          | 20000      | 1       |                   |            |
| ResNet50                          | 1          | 500        | 30      |                   |            |
| ResNet50                          | 1          | 1000       | 30      |                   |            |

## Reference
[1] (2022) Data preprocessing from https://app.roboflow.com

[2] (2022) Pre-trained model of Faster R-CNN from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

[3] (2022) Backbone model of ResNet50 from https://github.com/fizyr/keras-retinanet

[4] (2021) TFOD 2.0 Custom Object Detection Step By Step Tutorial from https://youtu.be/XoMiveY_1Z4

[5] (2022) How to Train Your Own Object Detector Using TensorFlow Object Detection API form https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api

[6] (2020) Object detection on Satellite Imagery using RetinaNet (Part 2) — Inference form https://medium.com/@ije_good/object-detection-on-satellite-imagery-using-retinanet-part-2-inference-aa0bf2a41eb4

[7] (2022) Use keras-retinanet to train your own dataset form https://www.programmersought.com/article/73514120308/

[8] (2022) How to build a Face Mask Detector using RetinaNet Model! from https://www.analyticsvidhya.com/blog/2020/08/how-to-build-a-face-mask-detector-using-retinanet-model/

[9] (2022) keras-retinanet form https://github.com/fizyr/keras-retinanet

[10] (2022) labelImg form https://github.com/heartexlabs/labelImg

[11] (2022) Object Detection with RetinaNet form https://keras.io/examples/vision/retinanet/

## End Credit
งานชิ้นนี้เป็นส่วนหนึ่งของวิชา DADS7202 Deep Learning หลักสูตรวิทยาศาสตร์มหาบัณฑิต คณะสถิติประยุกต์ สถาบันบัณฑิตพัฒนบริหารศาสตร์

กลุ่ม สู้ DL แต่ DL สู้กลับ 
สมาชิก: (1) 641xxxxx03 (2) 641xxxxx06 (3) 641xxxxx13 (4) 641xxxxx20
