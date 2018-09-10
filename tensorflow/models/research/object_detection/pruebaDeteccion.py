import numpy as np
import os

import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from lxml import etree

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from utils import label_map_util
from time import time

#Seleccionamos el modelo
MODEL_NAME = 'ssd_mobilenet_v1_coco'

#MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_CKPT = 'inference/four/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('training', 'label.pbtxt')

NUM_CLASSES = 90

#obtengo modelo
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#obtengo labels
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#categorias
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def list_files_ext(directory, extension):
    return list(f for f in os.listdir(directory) if f.endswith('.' + extension))
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def obtenerGroundTruth(im_name):
    box=np.zeros(4)
  #  nombre=im_name.split("/")
  #  nombre=nombre[len(nombre)-1]
    #ann=etree.parse('/home/marcelo/rcnn/py-faster-rcnn/data/citic/Annotations/'+nombre[:-4]+'.xml')
    ann=etree.parse(im_name.replace(".jpg",".xml"))
    bbox=ann.find("object/bndbox")
    box[0]= float(bbox.find('xmin').text)
    box[1] = float(bbox.find('ymin').text)
    box[2] = float(bbox.find('xmax').text)
    box[3] = float(bbox.find('ymax').text)
    return box


def obtenerIoU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou


#import glob
#im_names= [img for img in glob.glob("/home/marcelo/Escritorio/Bases_Datos/DataBase/conjSLRatCrop/negativos1/*jpg")]
id_img= list_files_ext("/home/czumelzu/images/test","jpg")
im_names=[]
for i in id_img:
    im_names.append(os.path.join("/home/czumelzu/images/test",i))

    # Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
font = cv2.FONT_HERSHEY_SIMPLEX


tp=0
tn=0
fp=0
fn=0
sumTime=0
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    sumTime=0


    for i,im_file in enumerate(im_names):
      print(i)
      image_np=cv2.imread(im_file)
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      initial_time=time()

      (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})
      final_time=time()
      ET=final_time-initial_time
      sumTime=sumTime+ET

      person=[]
      for i,pe in enumerate(classes[0]):
              if(pe==1 and scores[0][i]>0.7):
                  person.append(i)

      IoU=0
      if 'positivos' in im_file:
          if(len(person)>0):
              bbox=np.zeros(4)           
              bbox[0]=int(boxes[0][person[0]][1]*227)
              bbox[1]=int(boxes[0][person[0]][0]*227)
              bbox[2]=int(boxes[0][person[0]][3]*227)
              bbox[3]=int(boxes[0][person[0]][2]*227)
              groundTruth=obtenerGroundTruth(im_file)
              IoU=obtenerIoU(bbox,groundTruth)
          if(IoU>=0.75):
                        tp+=1
          else:
                        fn+=1
      else:
          if(len(person)>0):
                        fp+=1
          else:
                        tn+=1

acc=(tp+tn)*1.0/(tp+tn+fp+fn)
#recall
sens= tp*1.0/(tp+fn)
esp= tn*1.0/(tn+fp)
#presicion
if tp+fp == 0:
    pres=0
else:
    pres= tp*1.0/(tp+fp);

print("accuracy:"+str(acc))
print("sensibilidad"+str(sens))
print("especifidad"+str(esp))
print("precision:"+str(pres))
avgTime=sumTime/(len(im_names)*1.0)
print('Average time per image:',avgTime)
