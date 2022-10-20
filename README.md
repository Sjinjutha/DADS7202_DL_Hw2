# DADS7202_Deep Learning_Hw02

# INTRODUCTION
การตรวจจับจำแนกวัตถุ (Object Detection) โดยเปรียบเทียบ 2 models ระหว่าง Faster R-CNN และ Facebook RetinaNet กับ datasets แบรนด์น้ำดื่ม 3 แบรนด์ (Aquafina, Crystal, Nestle Purelife) เป็นภาษาไทย เพื่อระบุตำแหน่งฉลากของแบรนด์น้ำดื่มทั้ง 3 แบรนด์ จากผลการปรับแก้ไข model พบว่า 
- Faster R-CNN มีค่าผลการทดลอง mAP สูงสุดที่ 0.900 (IoU = 0.5) GPU ที่ใช้ train คือ GPU 0: Tesla T4 (UUID: GPU-7ff9c782-d6e4-d237-988a-4d7b930fd0d1)
- RetinaNet มีค่าผลการทดลอง mAP สูงสุดที่ 0.4599 (ไม่ระบุ IoU) GPU ที่ใช้ train คือ GPU 0: Tesla V100-SXM2-16GB (UUID: GPU-1829761b-5efa-4859-b8f7-667f7182fe45)

## About Dataset
ชุดข้อมูลที่สร้างขึ้นเองทั้งหมดประกอบด้วย Crystal, Nestle, Aquafina ทั้งหมด 1200 รูป ปรับขนาดรูปภาพเป็น 500 * 500 และนำรูปไปทำการ annotation สร้างกล่องขอบเขตของวัตถุเพื่อกำหนดว่าในพื้นที่บริเวณนี้เป็นของวัตถุอะไร บันทึกไฟล์รูปออกมาเป็น .xml (Pascal Voc)

## Model 1: Faster R-CNN (two-stage model)
เทคนิค Faster R-CNN นำข้อมูลผ่านเว็บไซต์ https://app.roboflow.com เพื่อตรวจสอบ อีกทั้งมีการปรับขนาดรูปภาพเป็น 416 * 416 แต่ไม่ได้มีการทำ Data Augmentation เนื่องจากชุดข้อมูลที่เราสร้างมีจำนวนมากและหลากหลายในเรื่องของมุมภาพ แสง การตัดขอบ การหมุน รวมอยู่ในชุดข้อมูล และทำการแบ่งชุดข้อมูลออกเป็น train set 90% (1080 รูป) และ test set 10% (120 รูป)

### Model 1
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
![p6](https://user-images.githubusercontent.com/113499057/196954086-558149f1-0b3a-4159-832a-274d0272fa2b.jpg)

โหลด pre-trained model จาก [Tensorflow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) และแตกไฟล์ใส่ pre-trained-models 
```
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
!tar -xvf faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
```
![11](https://user-images.githubusercontent.com/113499057/196779940-32ad199d-3347-4633-afa9-56fa1cf71d9d.jpg)
![12](https://user-images.githubusercontent.com/113499057/196779955-b4f4f53f-6112-4b6e-963b-2a856088c166.jpg)

**สร้าง label_map.pbtxt** เพื่อเป็นการนิยาม classes ที่เราต้องการตรวจจับแล้วนำมาเก็บใน annotation folder เพื่อนำไปสร้างไฟล์ .record ในขั้นตอนต่อไป

![p7](https://user-images.githubusercontent.com/113499057/196963189-413810d0-9b20-483c-a3e5-7e3679dbde7f.jpg)
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

![p8](https://user-images.githubusercontent.com/113499057/196964128-8b106643-e601-47c1-ad29-af80f7eb3f8e.jpg)

**สร้าง folder ชื่อ models ภายใน folder C_Dads7202 และสร้างไฟล์ pipline.config ใน folder model และแก้ไข filepath ให้ถูกต้อง**

แนะนำให้ปรับขนาดมิติเล็กและใหญ่ที่สุดให้เท่ากับขนาดรูปภาพของเราที่ใช้ (416 * 416)
ในการปรับแก้ไข config จะใช้หลัก ๆ อยู่ 4 ไฟล์ ได้แก่
* ckpt-0.index คือไฟล์ที่ไว้เก็บค่าของ pre-trained model ที่เรานำมาใช้
* label_map.txt (ใน annotations folder)
* train.record (ใน annotations folder)
* test.record (ใน annotations folder)
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
ตั้งค่า path ของ model directory  และ pipeline config
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
|    IoU    |    mAP    |
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

### Model 2
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
|    IoU    |    mAP    |
|-----------|-----------|
| 0.50:0.95 | 0.455     |
| 0.50      | 0.900     |
| 0.75      | 0.454     |


ใช้ Tensorboard แสดงกราฟค่า loss (ค่าคลาดเคลื่อนในการพยากรณ์)

![t_3](https://user-images.githubusercontent.com/113499057/196929052-65c41b69-4b18-4d97-844f-0aa38af6e0c5.jpg)

จากกราฟเป็นค่าความผิดพลาด (loss) จากการ train เมื่อเวลาผ่านไปความผิดพลาดยิ่งน้อยลง และค่อนข้างต่ำ

![20](https://user-images.githubusercontent.com/113499057/196789208-85f177a2-64ef-4051-a67c-7ad05917c773.jpg)

จะเห็นว่า model ยังไม่ดีเท่าที่ควร เพราะไม่สามารถตรวจจับวัตถุได้ครบหมดทุกตำแหน่งที่มี แต่ถึงแม้ว่าค่า loss จะดีขึ้นตามลำดับ ผลลัพธ์ยังไม่ดีเท่าที่ควร


### Comparing between Model 1 and Model 2 of Faster R-CNN
#### Model 1
![t_22](https://user-images.githubusercontent.com/113499057/196929032-620c2e2f-ee82-4019-bc3a-cce5c8fe9e39.jpg)
#### Model 2
![t_3](https://user-images.githubusercontent.com/113499057/196929052-65c41b69-4b18-4d97-844f-0aa38af6e0c5.jpg)

จากการปรับแก้ไขค่า parameter การเพิ่มค่าจำนวนรอบ (num_steps) จะเห็นได้ว่าค่า loss ดีขึ้น (ลดลงตามลำดับ)

#### Model 1
![16](https://user-images.githubusercontent.com/113499057/196785683-8d1d8146-4c0c-43dd-9a25-76c2a882c69f.jpg)
#### Model 2
![20](https://user-images.githubusercontent.com/113499057/196789208-85f177a2-64ef-4051-a67c-7ad05917c773.jpg)

จากการเปรียบเทียบรูปสองรูปที่ได้จากการ train model พบว่า model ยังไม่ดีเท่าที่ควร อาจมีกาปรับแก้ไขค่า hyperparameter ต่อไปแต่ไม่เป็นข้อแนะนำ เนื่องจากถือเป็นการสิ้นเปลืองเวลาและทรัพยากรโดยไม่จำเป็น เพราะค่า loss ที่แสดงในกราฟ ไม่ได้ลดลงมากเท่าที่ควร

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

**Train model**
- กำหนดขนาด 416 * 146
- batch size: 16
- num_steps: 150

```
!python train.py --img 416 --batch 16 --epochs 150 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
```
![y3](https://user-images.githubusercontent.com/113499057/196791483-74ddeb93-6805-43fc-952a-8af9e09295e1.jpg)

**Evaluation แสดงประสิทธิภาพของ model**
```
%load_ext tensorboard
%tensorboard --logdir runs
```
![y4](https://user-images.githubusercontent.com/113499057/196791486-57f784fc-da25-4348-87dc-5cccd6d9f8e7.jpg)

จากกราฟจะเห็นได้ว่า ค่าสัดส่วน metrics/mAP (Mean-Average Precision) มีค่าเพิ่มขึ้น ซึ่งค่ายิ่งมากยิ่งแสดงถึงความแม่นยำ

**Run Inference With Trained Weights**
เอาข้อมูล pre-trained model มาทดสอบ
```
!python detect.py --weights /content/yolov5/runs/train/exp2/weights/best.pt --img 416 --conf 0.1 --source /content/datasets/Brands-1/test/images
```

![y9](https://user-images.githubusercontent.com/113499057/196794863-971aa523-f62d-410b-8bd5-deb66ac36420.jpg)

ทดลองนำ model มาใช้กับข้อมูลชุด test ทั้งชุด
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

จะเห็นได้ว่า model ที่ใช้เทคนิค YOLOv5 แม่นยำกว่า ใช้เวลาและทรัพยากรน้อยกว่า อาจแนะนำให้ใช้เทคนิคนี้มากกว่า Faster R-CNN

## Model 2: RetiNet (one-stage model)
เทคนิค RatinaNet ทำการแบ่งชุดข้อมูลออกเป็น train set 70% (840 รูป) validation set 20% (240 รูป) และ test set 10% (120 รูป)

Mouth drive ไปที่ /content/drive  บน google drive
```
from google.colab import drive
drive.mount('/content/drive')
```
![n0](https://user-images.githubusercontent.com/113499057/196983454-ff2ce773-b934-4808-9f1f-aba7df701fde.jpg)

Import library ที่ต้องใช้ และเช็คเวอร์ชั่นของ library และ GPU ที่นำมารันในการเทรน
```
import sys
print( f"Python {sys.version}\n" )

import numpy as np
print( f"NumPy {np.__version__}\n" )

import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
print( f"TensorFlow {tf.__version__}" )
print( f"tf.keras.backend.image_data_format() = {tf.keras.backend.image_data_format()}" )

# Count the number of GPUs as detected by tensorflow
gpus = tf.config.list_physical_devices('GPU')
print( f"TensorFlow detected { len(gpus) } GPU(s):" )
for i, gpu in enumerate(gpus):
  print( f".... GPU No. {i}: Name = {gpu.name} , Type = {gpu.device_type}" )
```
![n2](https://user-images.githubusercontent.com/113499057/196982582-0864f318-f163-4190-a583-3659fcd4ad41.jpg)

Clone ตัว keras-retinanet model 
```
import os
print(os.getcwd())

!pip install utils

!git clone https://github.com/fizyr/keras-retinanet.git
%cd keras-retinanet/

!pip install .
!python setup.py build_ext --inplace
```
![n4](https://user-images.githubusercontent.com/113499057/196982591-7b424116-4b25-482b-b027-bb1e8ffb4ff4.jpg)
```
!pip install wget
!pip install pytz
!pip install Cython pandas tf-slim lvis
```
![n5](https://user-images.githubusercontent.com/113499057/196982594-b53a91ec-f15f-48ed-85e1-2588a84f7b13.jpg)
```
import numpy as np
import shutil
import pandas as pd
import os, sys, random
import re
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import wget
from PIL import Image # or import PIL.Image
import requests
import urllib
from keras_retinanet.utils.visualization import draw_box, draw_caption , label_color
from keras_retinanet.utils.image import preprocess_image, resize_image
from tensorflow import keras
import tensorflow_datasets as tfds
```

### Data processing
การนำข้อมูลสกุล .xml มาสร้างเป็นไฟล์ .CSV เพื่อใช้ในการนำไป run model
```
list_name = ['Train', 'Val', 'Test']

for i in list_name:
    imagePath= f"/content/drive/MyDrive/NIDA/DADS7202/DADS7202_HW2_Data/Train_Val_Test/{i}/imagePath_{i}"
    annotPath= f"/content/drive/MyDrive/NIDA/DADS7202/DADS7202_HW2_Data/Train_Val_Test/{i}/annotPath_{i}"

    names_data = 'data_' + i
    print("names_data")

    names_data = pd.DataFrame(columns=['fileName','xmin','ymin','xmax','ymax','class'])

    os.chdir(f"/content/drive/MyDrive/NIDA/DADS7202/DADS7202_HW2_Data/Train_Val_Test/{i}/")

    #read All files
    allfiles = [f for f in listdir(annotPath) if isfile(join(annotPath, f))]
    print(f"the total quantity of annotPath in {i}:", len(allfiles))

    #Read all pdf files in images and then in text and store that in temp folder
    for file in allfiles:
        #print(file)
        if (file.split(".")[1]=='xml'):
            fileName= imagePath+'/'+file.replace(".xml",'.jpg')
            tree = ET.parse(annotPath+'/'+file)
            root = tree.getroot()
            if root.find('object') :
                for obj in root.iter('object'):
                    cls_name = obj.find('name').text
                    xml_box = obj.find('bndbox')
                    xmin = xml_box.find('xmin').text
                    ymin = xml_box.find('ymin').text
                    xmax = xml_box.find('xmax').text
                    ymax = xml_box.find('ymax').text

            # if we want 0 in all elements with unbounding box image
            #else:
            #    cls_name = ''
            #    xmin = ymin = xmax = ymax = 0

            # Append rows in Empty Dataframe by adding dictionaries
            names_data = names_data.append({'fileName': fileName, 'xmin': xmin, 'ymin':ymin,'xmax':xmax,'ymax':ymax,'class':cls_name}, ignore_index=True)
            
            # names_data.to_csv(f"all_annotation_{i}.csv", index=False)
            
    print(names_data.shape)
    print(names_data)

# os.chdir("/content/drive/MyDrive/NIDA/DADS7202/DADS7202_HW2_Data/Train_Val_Test")
```
![n6-1](https://user-images.githubusercontent.com/113499057/196982598-1865d645-6b63-44c8-9307-ae90dbc5696f.jpg)

นิยาม filepath ของแต่ละ directory 
```
annot_train_dir = "/content/drive/MyDrive/NIDA/DADS7202/DADS7202_HW2_Data/Train_Val_Test/Train/annotation_Train_GoogleColab.csv"
annot_val_dir =   "/content/drive/MyDrive/NIDA/DADS7202/DADS7202_HW2_Data/Train_Val_Test/Val/annotation_Val_GoogleColab.csv"
annot_test_dir =  "/content/drive/MyDrive/NIDA/DADS7202/DADS7202_HW2_Data/Train_Val_Test/Test/annotation_Test_GoogleColab.csv"

classes_dir = "/content/drive/MyDrive/NIDA/DADS7202/DADS7202_HW2_Data/WaterBottle_Classes.csv"
```
ตรวจสอบ format ให้อยู่ในรูปแบบที่ต้องการตามภาพด้านล่าง
```
columns_ = ['fileName', 'xmin', 'ymin', 'xmax', 'ymax', 'class']

train_annot = pd.read_csv(annot_train_dir, names=columns_)
train_annot.head()
```
![n7](https://user-images.githubusercontent.com/113499057/196982602-be454a9d-31b9-4ce8-a86a-5c4df41a84f8.jpg)
```
val_annot = pd.read_csv(annot_val_dir, names=columns_)
val_annot.head()
```
![n8v](https://user-images.githubusercontent.com/113499057/196982567-28d6573a-44ca-44de-aa68-d35b2010b091.jpg)
```
test_annot = pd.read_csv(annot_test_dir, names=columns_)
test_annot.head()
```
![n9t](https://user-images.githubusercontent.com/113499057/196982574-35405297-0826-4435-a764-ff9320f66b59.jpg)
```
classes = ['Nestle','Aquafina', 'Crystal']
pd.read_csv(classes_dir, names=['class', 'index'])
```
![n10](https://user-images.githubusercontent.com/113499057/196982791-dfce7ffa-c63c-4be6-a9d6-d233543de9f0.jpg)

สุ่มข้อมูลรูปภาพจากไฟล์ CSV มา 1 แถว เพื่อดูรูปภาพ
```
# Check annotation with 1 image

data = train_annot

# pick a random image
filepath = data.sample()['fileName'].values[0]
##print(filepath)
# get all rows for this image
df2 = data[data['fileName'] == filepath]
print(df2)
im = np.array(Image.open(filepath))
print(im)

# if there's a PNG it will have alpha channel
im = im[:,:,:3]

for idx, row in df2.iterrows():
    print(idx, row)
    box = [
      row['xmin'],
      row['ymin'],
      row['xmax'],
      row['ymax'],
    ]
    print(box)
    draw_box(im, box, color=(255, 0, 0), thickness=4) #https://github.com/fizyr/keras-retinanet/blob/main/keras_retinanet/utils/visualization.py

plt.axis('off')
plt.imshow(im)
plt.show()                  
                  
#show_image_with_boxes(data)
```
![n11](https://user-images.githubusercontent.com/113499057/196982796-a8844574-0003-4eb4-8b28-5f0dcf59444e.jpg)

เปลี่ยน directory มาอยู่ที่ /content
```
os.chdir("/content/")
os.getcwd()
```
downlaod pre-trained model of 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'

ส่วนของการ train model เราจะนำ pre-trained model จากลิงค์ด้านบนมาใช้ในการปรับแก้ไข model โดยแบ่งการปรับออกเป็น 3 รูปแบบ คือการเปลี่ยน steps เป็น 100, 500, 1000 steps ซึ่งจะได้ผลลัพธ์เป็น mAP (Mean Average Precision)
```
import tensorflow as tf

if os.path.exists("/content/snapshot/") == False :
        os.mkdir("snapshot")

PRETRAINED_MODEL = "/content/drive/MyDrive/NIDA/DADS7202/DADS7202_HW2_Data/Train_Val_Test/snapshots/resnet50_coco_best_v2.1.0.h5"
path_snapshot = "/content/snapshot/"

def retina_train_(steps_=100): 
    # default 1e-5  --snapshot "/content/snapshot/resnet50_csv_01.h5" \
    !python "/content/keras-retinanet/keras_retinanet/bin/train.py" \
            --backbone "resnet50" \
            --batch-size 1 \
            --epochs 30   \
            --steps {steps_}   \
            --gpu 0  \
            --weights {PRETRAINED_MODEL} \
            --tensorboard-dir './logs' \
            --snapshot-path {path_snapshot} \
            csv {annot_train_dir}  {classes_dir} \
            --val-annotations {annot_val_dir}
```
#### Train Model 1
ในการ run ครั้งนี้ใช้
- batch size = 1
- num steps = 100
- epochs = 30
```
retina_train_(100)
```
![n12t](https://user-images.githubusercontent.com/113499057/196982800-a6395eeb-509e-4fa0-b526-44dc04b25790.jpg)
![n12t-2](https://user-images.githubusercontent.com/113499057/196982806-6ea84184-d700-4867-b936-a241abf55f13.jpg)

#### Train Model 2
ในการ run ครั้งนี้ใช้
- batch size = 1
- num steps = 500
- epochs = 30
```
retina_train_(steps_=500)
```
![n13t-1](https://user-images.githubusercontent.com/113499057/196982811-00f54af7-e65b-4ab5-90e4-db2524fc5493.jpg)

#### Train Model 3
ในการ run ครั้งนี้ใช้
- batch size = 1
- num steps = 1000
- epochs = 30
```
retina_train_(steps_=1000)
```
![n14t](https://user-images.githubusercontent.com/113499057/196982813-14e7c229-8696-4197-af05-98f383e87c27.jpg)

#### Evaluation
เตรียมข้อมูลเพื่อตรวจสอบประสิทธิภาพในการพยากรณ์ผลของ model โดยการสุ่มบางภาพจากชุด test มาเปรียบเทียบระหว่าง predicted image และ actual image
```
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

model_path = os.path.join(path_snapshot, sorted(os.listdir(path_snapshot), reverse=True)[0])
print(model_path)

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')  ## Use backbone as resnet50
model = models.convert_model(model)

# load label to names mapping for visualization purposes
labels_to_names = pd.read_csv(classes_dir,header=None).T.loc[0].to_dict()
labels_to_names
```
![n15](https://user-images.githubusercontent.com/113499057/196982926-774a71e8-cc8d-4966-91c9-044cb716103f.jpg)
```
THRES_SCOREs = 0.35  # Set Score Threshold Value

import cv2
import time

def df_plot_orinal(drawOG, img_path, df):
    df = df[df['fileName']==img_path]
    for i,r in df.iterrows():
        cv2.rectangle(drawOG, (r['xmin'], r['ymin']), (r['xmax'], r['ymax']), (255,0,0),2)
    

def img_inference(img_path, df_data, THRES_SCORE=THRES_SCOREs):
    image = read_image_bgr(img_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    drawOG = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    df_plot_orinal(drawOG, img_path, df_data)
    # correct for image scale
    boxes /= scale
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        #print(score)
        if score < THRES_SCORE:
            continue
        color = label_color(label)
        b = box.astype(int)
        draw_box(draw, b, color=color)
        caption = "{} {:.3f}%".format(labels_to_names[label], score*100)
        print(box, score, label)
        
    fig = plt.figure(figsize=(20, 20))
    ax1=fig.add_subplot(1, 2, 1)
    plt.imshow(draw)
    ax2=fig.add_subplot(1, 2, 2)
    plt.imshow(drawOG)

    ax1.title.set_text('Predicted')
    ax2.title.set_text('Actual')
    plt.show()
```

```
THRES_SCORE_Adjust = 0.4

#print(test_annot.head())

data_sample = test_annot.sample(n=5)  #Predict on Random 5 Image
for i,r in data_sample.iterrows():
    img_inference(r['fileName'], test_annot, THRES_SCORE_Adjust)
```
![n16p-1](https://user-images.githubusercontent.com/113499057/196982934-90ca3030-d309-457b-b060-cd7600c67899.jpg)

![n16p-2](https://user-images.githubusercontent.com/113499057/196982940-33d776ee-c1a8-4809-a81e-039a0c74ca15.jpg)

หลังจากนั้นนำ model ทั้ง 3 แบบ มาเปรียบเทียบประสิทธิภาพในแต่ละ model กับชุดข้อมูล test ทั้งชุด โดยใช้ค่า mAP ในการเปรียบเทียบ
```
import tensorflow as tf

for i in range(1,4):
    model_path_r = f"/content/drive/MyDrive/NIDA/DADS7202/DADS7202_HW2_Data/Snapshot/round{i}_best_resnet50.h5"

    !python "/content/keras-retinanet/keras_retinanet/bin/evaluate.py" \
        csv {annot_test_dir} {classes_dir} \
        {model_path_r} --convert-model
```
![n17](https://user-images.githubusercontent.com/113499057/196982943-f5d2eda2-aae7-4075-ae3c-3cfb21eeb6ff.jpg)

### Comparing between Model 1, Model 2 and Model 3 of RetinaNet
|                                   |    mAP   |  num steps |
|-----------------------------------|----------|------------|
| ResNet50                          |  0.1317  |    100     |
| ResNet50                          |  0.3967  |    500     |
| ResNet50                          |  0.4599  |    1000*   | 

*กำหนด num_steps=1000 แต่ในทางปฎิบัติ ในการ run 1 epoch จะ train เพียงประมาณ 700 steps โดยคาดว่ามากจากจำนวนของชุดข้อมูลชุด train มีน้อยกว่าจำนวน steps ที่กำหนด

จากการปรับค่าบน default hyperparameter ของ pre-trained model นั่นคือ num_steps ตามตารางข้างต้น โดยกำหนดค่า epochs ที่ 30 รอบ

จากผลการทดลองให้การ train model 3 เป็น model ที่ดีที่สุด เนื่องจากให้ mAP สูงที่สุด


## Comparing between Faster R-CNN and RetinaNet
|                                   | batch size |  num steps |  epochs |    mAP    |                               GPU ที่ใช้ประมวลผล                                 |
|-----------------------------------|------------|------------|---------|-----------|-------------------------------------------------------------------------------|
| Faster R-CNN ResNet50 V1 640x640  | 8          | 20000      | 1       |   0.9000  | GPU 0: Tesla T4 (UUID: GPU-7ff9c782-d6e4-d237-988a-4d7b930fd0d1)              |
| ResNet50                          | 1          | 1000       | 30      |   0.4599  | GPU 0: Tesla V100-SXM2-16GB (UUID: GPU-1829761b-5efa-4859-b8f7-667f7182fe45)  |


## Discussion
-  การ run ผ่าน local python (Jupyter Notebook) ไม่สามารถ import library เพื่อใช้ในการปรับแต่ง model แต่สามารถแก้ปัญหานี้ได้โดยการใช้ google collab แทน
-  การใช้ google collab มีทรัพยากรไม่เพียงพอในการ train model ต้อง upgrade collab pro
-  ข้อจำกัดของการดึง model ของคนอื่นมาใช้ อาจไม่สามารถปรับแก้ค่า hyperparameter ได้เท่าที่ควร ปรับได้เพียงตัวพื้นฐานเท่านั้น เช่น batch size, num_steps ในเทคนิค Faster R-CNN เราไม่สามารถแก้ไขจำนวน epochs ได้


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
