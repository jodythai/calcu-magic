import re
import os
import time
import sys
import base64
import numpy as np
import cv2
import uuid
import pathlib
import mahotas
from sklearn.externals import joblib
from skimage.feature import hog
import imutils
from imutils import contours
from imutils import adjust_brightness_contrast
import tensorflow as tf
from flask import Blueprint, Flask, request, redirect, jsonify, current_app

upload = Blueprint('upload', __name__)

FILENAME_TEMPLATE = '{}.jpg'
PREDICT_IMAGE_WIDTH = 28
PREDICT_IMAGE_HEIGHT = 28

model_loaded = False
model_digits_svm = None
model_operators_svm = None
model_digits_operators = None
model_digits_cnn = None
model_operators_cnn = None
model_digits_ops_cnn = None

operator_dict = {
  0 : '+',
  1 : '-',
  2 : '/',
  3: '*'
}

def load_model():
  global model_loaded, model_digits_svm, model_operators_svm, model_digits_operators, model_digits_cnn, model_operators_cnn, model_digits_ops_cnn
  # model_num = tf.keras.models.load_model("static/models/" + MODEL_NUM)
  # Load the classifier
  model_digits_operators = joblib.load("static/models/digits_operator_cls.pkl")
  model_digits_svm = joblib.load("static/models/digits_cls.pkl")
  model_operators_svm = joblib.load("static/models/operators_cls.pkl")
  model_digits_cnn = tf.keras.models.load_model("static/models/handwritten_model.h5")
  model_digits_ops_cnn = tf.keras.models.load_model("static/models/mnist_operators_cnn.h5")
  model_operators_cnn = tf.keras.models.load_model("static/models/operators_cnn.h5")
  model_loaded = True

def deskew(image, width):
    (h, w) = image.shape[:2]
    moments = cv2.moments(image)

    skew = moments['mu11'] / moments['mu02']
    M = np.float32([[1, skew, -0.5*w*skew],
                    [0, 1, 0]])
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    image = imutils.resize(image, width=width)

    return image

def center_extent(image, size):
  (eW, eH) = size

  if image.shape[1] > image.shape[0]:
    image = imutils.resize(image, width=eW)
  else:
    image = imutils.resize(image, height=eH)

  extent = np.zeros((eH, eW), dtype='uint8')
  offsetX = (eW - image.shape[1]) // 2
  offsetY = (eH - image.shape[0]) // 2
  extent[offsetY:offsetY + image.shape[0], offsetX:offsetX+image.shape[1]] = image

  CM = mahotas.center_of_mass(extent)
  (cY, cX) = np.round(CM).astype("int32")
  (dX, dY) = ((size[0]//2) - cX, (size[1] // 2) - cY)
  M = np.float32([[1, 0, dX], [0, 1, dY]])
  extent = cv2.warpAffine(extent, M, size)

  return extent

def predict_operator_svm(roi):
  # Resize the image
  # roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
  # roi = cv2.dilate(roi, (3, 3))
  # Calculate the HOG features
  roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
  prediction = model_operators_svm.predict(np.array([roi_hog_fd], 'float64'))

  return prediction[0]

def predict_digits_svm(roi):
  # Resize the image
  roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
  roi = cv2.dilate(roi, (3, 3))
  # Calculate the HOG features
  roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
  prediction = model_digits_svm.predict(np.array([roi_hog_fd], 'float64'))

  return prediction[0]

def calculate(equation):
  total = 0
  s = ''
  i = 0
  while i < len(equation):
    s +=  str(equation[i])
    i += 1
  try:
    total = round(eval(s), 2)
  except Exception as e:
    print(e)
    total = 0
  return total, s 

def processing(img_name, mode):
    
  CROPPED_W = 554
  CROPPED_H = 166

  # Read the input image 
  im = cv2.imread(os.path.join(current_app.config['UPLOAD_FOLDER'], img_name))

  # create a folder to upload roi
  img_upload_path = pathlib.Path(os.path.join(current_app.config['UPLOAD_FOLDER'], img_name.rsplit('.')[0]))
  if not img_upload_path.is_dir():
    os.mkdir(img_upload_path)
  
  roi_upload_path = pathlib.Path(os.path.join(img_upload_path, 'roi'))
  if not roi_upload_path.is_dir():
    os.mkdir(roi_upload_path)

  if mode == 'camera':
    # image ratio w/h
    im_ratio = round(im.shape[1]/im.shape[0], 2)
    center_w = im.shape[1]//2
    center_h = im.shape[0]//2
    x = center_w-(CROPPED_W//2)
    y = center_h-(CROPPED_H//2)

    # crop the image
    im = im[y:y + CROPPED_H, x:x + CROPPED_W]
    cv2.imwrite(os.path.join(img_upload_path, "cropped_img.jpg"), im)
  
  if mode == 'canvas':
    im = 255 - im

  # image preprocessing
  # brightness = float(50)
  # contrast = float(1)
  # im = adjust_brightness_contrast(im, contrast=contrast, brightness=brightness)
  gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  thresh1 =cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
  _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  edged = imutils.auto_canny(thresh)
  cv2.imwrite(os.path.join(img_upload_path, "gray.jpg"), gray)
  cv2.imwrite(os.path.join(img_upload_path, "blurred.jpg"), blurred)
  cv2.imwrite(os.path.join(img_upload_path, "im_thresh.jpg"), thresh)
  cv2.imwrite(os.path.join(img_upload_path, "im_edged.jpg"), edged)

  # find contours
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)

  # sort the contours
  (cnts, boundingBoxes) = contours.sort_contours(cnts, method="left-to-right")
  # list_roi = []
  roi_operator_h_sum = 0

  im2 = im.copy()
  roi_operators = []
  roi_numbers = []
  equation = []

  c_prev = ''
  # FIND ROI, loop over the contours
  for (i, c) in enumerate(cnts):
    # compute the bounding box of the contour, then use the
    # bounding box coordinates to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    if i > 0:
      (x_prev, y_prev, w_prev, h_prev) = cv2.boundingRect(c_prev)
    
    if h > 5 and w > 10:
      cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
      roi = thresh[y-5:y+h+5, x-5:x+w+5]

      # write roi images to folder
      cv2.imwrite(os.path.join(pathlib.PurePath(roi_upload_path), "roi_" + str(i) + ".jpg"), roi) 

      roi = deskew(roi, 28)
      roi = center_extent(roi, (28,28))
      cv2.imwrite(os.path.join(pathlib.PurePath(roi_upload_path), "roi_deskew_" + str(i) + ".jpg"), roi)

      if i % 2 == 1: #operator
        roi_operators.append([roi, c])
        roi_operator_h_sum += h
        if i > 0 and h >= h_prev:
          operator = '/'
        else:
          operator = operator_dict.get(predict_operator_svm(roi))
        equation.append(operator) 
        cv2.putText(im2, str(operator), (x+int(w/2), y),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(pathlib.PurePath(roi_upload_path), "operator_" + str(i) + ".jpg"), roi)
      else: #digit
        roi_numbers.append([roi, c])
        digit = np.argmax(model_digits_cnn.predict(np.expand_dims(roi, axis=0)), axis=1)[0]
        equation.append(digit)
        cv2.putText(im2, str(digit), (x+w, y+int(h/2)),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
      c_prev = c
      # add roi into a list and calculate sum
      # list_roi.append([roi, c])
      # roi_h_sum += h
  
  # roi_operator_h_avg = roi_operator_h_sum / len(list_roi)
  # for roi, cnts in roi_operators:
    
  print(equation)    

  total, eval_string = calculate(equation)

  results = {}
  if len(roi_operators) > 0 and len(roi_numbers) >= 2:
    results['status'] = 1
    # write result to image
    (x, y, w, h) = cv2.boundingRect(roi_numbers[0][1])
    cv2.putText(im2, eval_string + '=' + str(total), (x, int(im.shape[0]*3/4)),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    im_name = img_name.split('.')[0]
    results['calculated_img'] = os.path.join(img_upload_path, im_name + "_calculated.jpg")
    results['equation'] = eval_string + '=' + str(total)
    cv2.imwrite(results['calculated_img'], im2)
  else:
    results['status'] = 0 # error

  return results

@upload.route('/upload/', methods=['POST'])
def handle_upload():
  global model_loaded, model

  if not model_loaded:
    load_model()

  if request.method == 'POST':

    data = request.get_json()

    # Preprocess the upload image
    img_raw = data['data-uri'].encode()
    img_str = re.search(b"base64,(.*)", img_raw).group(1)
    img_decode = base64.decodebytes(img_str)

    # # Write the image to the server
    # upload file to storage
    id = uuid.uuid4().hex
    filename = FILENAME_TEMPLATE.format(id)
    with open(os.path.join(current_app.config['UPLOAD_FOLDER'], FILENAME_TEMPLATE.format(id)), 'wb') as f:
      f.write(img_decode)

    print(filename, file=sys.stdout)

    results = processing(filename, data['mode'])

    if results['status'] == 1:
      with open(results['calculated_img'], "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        results['image'] = encoded_string.decode()
  return jsonify(results)

  