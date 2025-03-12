# av-project1

# **2D Object Detection Project on a Video Clip**

## **Project Overview**
Object detection plays a crucial role in autonomous driving by enabling vehicles to perceive and interpret their surroundings with high accuracy. In this project, we develop a deep learning model using YOLOv11 to detect and classify cyclists, pedestrians, and vehicles in urban environments. The model will be trained and fine-tuned on the COCO (Common Objects in Context) dataset, which contains diverse real-world images with annotated objects commonly found in urban settings, such as cars, bicycles, and pedestrians.


## **Dataset**
In this project, we employed the COCO (Common Objects in Context) dataset, a prominent benchmark in computer vision for tasks like object detection, segmentation, and recognition. COCO offers a diverse array of images annotated with 80 common object categories, including pedestrians, cyclists, and vehicles, which are crucial for our urban detection application. The dataset comprises thousands of high-quality, real-world images taken in various settings, making it perfect for training deep learning models to recognize objects in dynamic, cluttered environments.

COCO's extensive annotations include bounding boxes, instance segmentation masks, and keypoints, enabling precise detection and classification. Its real-world variety ensures that our model generalizes well across different lighting conditions, occlusions, and urban scenarios. For training, we utilize the COCO dataset's structured format, converting annotations into a format compatible with YOLOv11. Furthermore, data preprocessing techniques such as resizing, normalization, and augmentation are applied to boost model performance.

## **Setup**  

To prepare the environment for this project, we install and import the necessary libraries required for YOLOv11-based object detection, data processing, and visualization. The setup includes **Ultralytics' YOLO library**, which provides a streamlined interface for training and inference, along with essential libraries like **Torch, OpenCV, NumPy, Matplotlib, and PIL** for handling image processing, visualization, and performance evaluation.  

The **COCO dataset tools (`pycocotools`)** are included to work with dataset annotations efficiently, enabling the extraction of labeled objects such as **pedestrians, cyclists, and vehicles**. Additionally, **Torchvision transforms** help preprocess images before feeding them into the model. The setup also incorporates utilities for handling file downloads and extractions, ensuring seamless access to the dataset and model weights.  


