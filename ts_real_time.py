import warnings
import numpy as np
import os
import tensorflow as tf
import cv2
import sys
import argparse
from imutils.video import VideoStream
import time

def ts_detection (frame, sess, boxes, scores, classes, num_detections):

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # TODO
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image, axis=0)
    # Actual detection
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 1.5)

    return  idxs, boxes, scores, classes

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default="0",
	help="index of webcam on system")
ap.add_argument("-m", "--models", default="models",
    help="base path to model directory")
ap.add_argument("-mn", "--model_name", default="faster_rcnn_inception_resnet_v2_atrous",
    help="name of model detection")

args = vars(ap.parse_args())

# Initialization for detection objects
# Append Tensorflow object detection and darkflow directories to path
sys.path.append('PATH_TO_TENSORFLOW_OBJECT_DETECTION_FOLDER')  # ~/tensorflow/models/research/object_detection
# sys.path.append('PATH_TO_DARKFLOW_FOLDER') # ~/darkflow
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# Loading model
MODEL_PATH = os.path.join(args["models"], args["model_name"])
# Path to frozen detection graph. This is the actual model that is used for the traffic sign detection.
PATH_TO_CKPT = os.path.join(MODEL_PATH, 'inference_graph/frozen_inference_graph.pb')
# Loading list of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join("", "label_map.pbtxt")
# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `2`,
# we know that this corresponds to `mandatory`.
NUM_CLASSES = 3
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
category_index = label_map_util.create_category_index(categories)
COLORS=[(255,0,0), (0,0,255), (0,255,0), (128,255,255)]

# Load a (frozen) Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Each box represents a part of the image where a particular object was detected.
boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
scores = detection_graph.get_tensor_by_name('detection_scores:0')
classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

sess = tf.compat.v1.Session(graph=detection_graph)
# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
#vs = VideoStream("1.mp4").start()
time.sleep(1.0)
# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1
# loop over frames from the video file stream
while True:
    # read the next frame from the file
    frame = vs.read()
    #frame = imutils.resize(frame, width=450)
    (height, width, _) = frame.shape
    # detect traffic signs
    idxs, BOXES, SCORES, CLASSES = ts_detection (frame, sess, boxes, scores, classes, num_detections)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            ymin = int((BOXES[0][i][0] * height))
            xmin = int((BOXES[0][i][1] * width))
            ymax = int((BOXES[0][i][2] * height))
            xmax = int((BOXES[0][i][3] * width))
            LABEL = category_index[int(CLASSES[0][i])]
            SCORE = int(SCORES[0][i] * 100)
            if SCORE > 85:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), COLORS[int(LABEL['id'])], 2)
                y = ymin - 10 if ymin - 10 > 10 else ymin + 10
                cv2.putText(frame, str(LABEL['name']) + ": %" + str(SCORE), (xmin, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[int(LABEL['id'])], 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
