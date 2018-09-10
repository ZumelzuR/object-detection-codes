from ctypes import *
import math
import random
import os
import numpy as np
import cv2
import sys
from lxml import etree
from time import time

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

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
    #if type(image) == bytes:
       # img = load_image(image, 0, 0)
    img=image
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, img)
    dets = get_network_boxes(net, img.w, img.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
 #   free_image(img)
    free_detections(dets, num)
    return res

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def list_files_ext(directory, extension):
    return list(f for f in os.listdir(directory) if f.endswith('.' + extension))
def convert(box):
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x,y,w,h)

def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im

def rechange_box(box,framex):
    #p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    # This recalculation is for not modify the code of the tracking, because the format of the tracking box is different
    w=box[2]-box[0]
    h=box[3]-box[1]
    if w>frame.shape[1]-box[0]:
        w=frame.shape[1]-box[0]
    if h>frame.shape[0]-box[1]:
        h = frame.shape[0] - box[1]
    return (box[0],box[1],w,h)

def format_box(box):
    x = int(box[0]) - int(box[2]) / 2
    y = int(box[1]) - int(box[3]) / 2
    w = int(box[0]) + int(box[2]) / 2
    h = int(box[1]) + int(box[3]) / 2

    return list((x-10,y-10,w+10,h+10))

def detect_box_from_frame(net,meta,inputFrame):
    #im_file=os.path.join(os.getcwd(), "positivos2101.jpg").encode('ASCII')
    r = detect(net, meta, array_to_image(inputFrame))
    person_detections_indexes = []
    for i, tensor in enumerate(r):
        if (tensor[0] == b'person' and tensor[1] >0.7):
            person_detections_indexes.append([i,tensor[1]])
        box_detected = np.zeros(4)
        box_detected[0] = int(r[person_detections_indexes[0][0]][2][0])
        box_detected[1] = int(r[person_detections_indexes[0][0]][2][1])


        box_detected[2] = int(r[person_detections_indexes[0][0]][2][2])
        box_detected[3] = int(r[person_detections_indexes[0][0]][2][3])
        bb = format_box(box_detected)
      #  if bb[2] > inputFrame.shape[1]:
      #      bb[2] =inputFrame.shape[1]
      #  if bb[3] > inputFrame.shape[0]:
      #      box_detected[3] =inputFrame.shape[0]
    return bb,person_detections_indexes[0][1]


# tracker configuration
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
#for i in tracker_types:
tracker_type = tracker_types[0]

if int(minor_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

video = cv2.VideoCapture("1.mp4")

if not video.isOpened():
    print
    "Could not open video"
    sys.exit()

ok, frame = video.read()
if not ok:
    print
    'Cannot read video file'
    sys.exit()
frame = rescale_frame(frame, percent=60)


# load net parameters and configuration
net = load_net(b"yolov3.cfg", b"yolov3.weights", 0)
meta = load_meta(b"coco.data")

#total_time = 0
#initial_time = time()

# Define an initial bounding box
bbox,confidence = detect_box_from_frame(net,meta,frame)
#bbox = (276, 423, 361, 320)
#bbox = (276, 423, 361, 320)
#cv2.imwrite("frame%d.jpg" % 1, frame)
# Uncomment the line below to select a different bounding box
#bbox = cv2.selectROI(frame, False)


#### do the tracking #####
tracker = cv2.TrackerBoosting_create()
bbox, confidence = detect_box_from_frame(net, meta, frame)
#bbox = rechange_box(bbox)
bbox=rechange_box(bbox,frame)

ok = tracker.init(frame, bbox)

count=0
bbox_previous = []
sum=0;
sum_time=0;
while True:
    start = time()
    ok, frame = video.read()
    if not ok:
        break
    frame = rescale_frame(frame, percent=60)


    timer = cv2.getTickCount()
    # update tracker
    #print(count)
    if count %150 == 0 and count!=0:
        tracker = cv2.TrackerBoosting_create()
        bbox, confidence = detect_box_from_frame(net, meta, frame)
        bbox2 = rechange_box(bbox,frame)
        print(bbox2);
        ##(406, 293, 114, 139)
        ##bbox2 = cv2.selectROI(frame, False)
       ## print(bbox2);
        ok = tracker.init(frame, bbox2)
    else:
        ok, bbox = tracker.update(frame)
    #if len(bbox_previous) != 0:
    #    change_x = abs(bbox[0] - bbox_previous[0])
    #    change_y = abs(bbox[1] - bbox_previous[1])
    #    change_w = abs(bbox[2] - bbox_previous[2])
    #    change_z = abs(bbox[3] - bbox_previous[3])
    #    if((change_x > 50 or change_y >50) or (change_x ==0 and change_y ==0 and change_w ==0 and change_z ==0)):
    #        tracker = cv2.TrackerBoosting_create()
    #        bbox, confidence = detect_box_from_frame(net, meta, frame)
    #        if(confidence > 0.7):
    #            bbox = rechange_box(bbox)
    #            ok = tracker.init(frame, bbox)
    #        else:
    #            ok=False
    #bbox_previous = bbox
    # fps
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
        #success tracking
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        # error tracking
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        tracker = cv2.TrackerBoosting_create()
        #bbox, confidence = detect_box_from_frame(net, meta, frame)
        #if(confidence > 0.7):
        #    bbox = rechange_box(bbox)
        #    ok = tracker.init(frame, bbox)
        #else:
        #    ok=False

    cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    #  FPS
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
    sum = sum+int(fps)
    final_time = time()
    temp_time_result = final_time - start
    sum_time=sum_time+temp_time_result

    cv2.imshow("Tracking", frame)
    count=count+1
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27: break
#---- end ----#
print(i)
print(sum/count)
print(sum_time/count)


