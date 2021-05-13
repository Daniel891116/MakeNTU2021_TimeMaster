from tflite_runtime.interpreter import Interpreter 
from PIL import Image
from load_labels import load_labels
from classify_image import classify_image
from cv2_init import cv2_init
from js_parser import js_parser
import cv2
import numpy as np
import time

#====================================================================================#
#init opencv2
cap = cv2_init(224, 224)

#set models and labels path
model_path = "models_and_labels./model3.tflite"
label_path = "models_and_labels./labels3.txt"

# Read class labels.
labels = load_labels(label_path)

# Load interpreter
interpreter = Interpreter(model_path)

# Allocate memory for interpreter
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']

# Set parameters
current_event = None
time_left = 0
classified_event = None
time_interval = 30      #set interval of image classification(sec)

while(True):
    print("waiting...")

    data_from_js = input()
    lineread = js_parser(data_from_js)
    [current_event, time_left] = [lineread['name'], lineread['period']]
    start_time = time.time()

    while(time_left >= 0):
        ret, frame = cap.read()
        #cv2.imshow('frame', frame )

        if time.time() - start_time >= time_interval:

            frame = frame[:,:,::-1]     #change color from BGR to RGB
            image = Image.fromarray(frame)
            image = image.resize((width, height)) # resize image to (224, 224)
            label_id, prob = classify_image(interpreter, image)
            start_time = time.time()
            time_left -= time_interval

            # Return the classification label of the image.
            classification_result = labels[label_id]
            print(classification_result)
            print('time_left:', time_left)

    if time_left < 0 :
        time_left = 0
    current_event = None    