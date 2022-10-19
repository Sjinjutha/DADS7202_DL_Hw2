# DADS7202_Deep Learning_Hw02

# INTRODUCTION
การตรวจจับจำแนกวัตถุ (Object Detection) โดยเปรียบเทียบ 2 models ระหว่าง Faster R-CNN และ Facebook RetinaNet

## About Dataset
ชุดข้อมูลที่สร้างขึ้นเองทั้งหมดประกอบด้วย Crystal, Nestle, Aquafina ทั้งหมด 1200 รูป ปรับขนาดรูปภาพเป็น 500*500 และนำรูปไปทำการ annotation สร้างกล่องขอบเขตของวัตถุเพื่อกำหนดว่าในพื้นที่บริเวณนี้เป็นของวัตถุอะไร บันทึกไฟล์รูปออกมาเป็น .xml (Pascal Voc)

## Model 1: Faster R-CNN (two-stage model)
เทคนิค Faster R-CNN นำข้อมูลผ่านเว็บไซต์ https://app.roboflow.com เพื่อตรวจสอบ อีกทั้งมีการปรับขนาดรูปภาพเป็น 416*416 แต่ไม่ได้มีการทำ Data Augmentation เนื่องจากชุดข้อมูลที่เราสร้างมีจำนวนมากและหลากหลายในเรื่องของมุมภาพ แสง การตัดขอบ การหมุน รวมอยู่ในชุดข้อมูล และทำการแบ่งชุดข้อมูลออกเป็น train set 90% (1080 รูป) และ test set 10% (120 รูป)

### Initial model


จำนวนรอบของการ train = 5000
batch size = 8

### Tuned model
จำนวนรอบของการ train = 10000
batch size = 8

### Comparing between initial model and tuned model of Faster R-CNN

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
