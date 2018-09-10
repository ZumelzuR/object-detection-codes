from ctypes import *
import math
import random
import os
import numpy as np
from lxml import etree
from time import time

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("/Users/mclovin/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/czumelzu/yolo/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    #print(dets)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def list_files_ext(directory, extension):
    return list(f for f in os.listdir(directory) if f.endswith('.' + extension))
def convert(size, box):
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x,y,w,h)
def get_locations_box_from(im_name):
    box=np.zeros(4)
    ann=etree.parse(im_name.replace(".jpg",".xml"))
    w = int(ann.find("size/width").text)
    h = int(ann.find("size/height").text)
    bbox=ann.find("object/bndbox")

    b = (float(bbox.find('xmin').text), float(bbox.find('xmax').text), float(bbox.find('ymin').text),
         float(bbox.find('ymax').text))
    bb = convert((w, h), b)

    return bb
def getIoU(boxA, boxB):
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
#if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
# sobre data
#net = load_net(b"cfg/yolov2_citic.cfg", b"backup/original/yolov2_citic_2700.weights", 0)
#net = load_net(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
net = load_net(b"cfg/yolov3_citic.cfg", b"backup/tree/yolov3_citic_7000.weights", 0)
#meta = load_meta(b"data/citic.data")
#meta = load_meta(b"voc_original.data")
meta = load_meta(b"data/citic_3.data")


id_img= list_files_ext("/home/czumelzu/images/test","jpg")
im_names=[]
for i in id_img:
    im_names.append(os.path.join("/home/czumelzu/images/test",i))
#./darknet detect cfg/resnet152.cfg resnet152.weights /home/czumelzu/images/test/positivos693.jpg

#./darknet classifier predict cfg/imagenet1k.data cfg/alexnet.cfg alexnet.weights /home/czumelzu/images/test/positivos693.jpg
#./darknet detect cfg/yolov2_original.cfg yolov2.weights /home/czumelzu/images/test/positivos693.jpg

tp=0
tn=0
fp=0
fn=0
total_time = 0
avg_IoU = 0
total_ex=1;
for i,im_file in enumerate(im_names):
    #i=1
    total_ex=total_ex+1
    print(i)
    initial_time = time()
    #im_file=os.path.join(os.getcwd(), "images/test/positivos10.jpg")
    # hay que pasar todo a bytes ya que es lo que se leera por los scripts en C de darknet
    r = detect(net, meta, im_file.encode('ASCII'))
    final_time=time()
    temp_time_result = final_time - initial_time
    total_time = total_time + temp_time_result
    person_detections_indexes = []
    for i, tensor in enumerate(r):
        if (tensor[0] == b'person' and tensor[1] > 0.7):
        	person_detections_indexes.append(i)
    IoU = 0
    # arreglar, ya que pueden haber mas de una persona en la imagen

    if 'positivos' in im_file:
      	if (len(person_detections_indexes) > 0):
	        box_detected = np.zeros(4)
	        box_detected[0] = int(r[0][2][0])
	        box_detected[1] = int(r[0][2][1])
	        box_detected[2] = int(r[0][2][2])
	        box_detected[3] = int(r[0][2][3])
	        box_image = get_locations_box_from(im_file)
	        IoU = getIoU(box_detected, box_image)
	        avg_IoU=+avg_IoU
	        if (IoU >= 0.75):
	            tp += 1
	        else:
	            fn += 1
    else:
        if (len(person_detections_indexes) > 0):
            fp += 1
        else:
            tn += 1

avg_IoU=avg_IoU/total_ex
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
print("sensibilidad:"+str(sens))
print("especifidad:"+str(esp))
print("precision:"+str(pres))
print("avg IoU:"+str(avg_IoU))
avgTime=total_time/(len(im_names)*1.0)
print('Average time per image:',avgTime)
