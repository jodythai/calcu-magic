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

# def preprocess_image(img_raw):
#   predict_img_width = PREDICT_IMAGE_WIDTH
#   predict_img_height = PREDICT_IMAGE_HEIGHT

#   img_str = re.search(b"base64,(.*)", img_raw).group(1)
#   img_decode = base64.decodebytes(img_str)

#   image = tf.image.decode_jpeg(img_decode, channels=1)
#   image = tf.image.resize(image, [predict_img_width, predict_img_height])
#   image = (255 - image) / 255.0  # normalize to [0,1] range
#   image = tf.reshape(image, (1, predict_img_width, predict_img_height, 1))

#   return image, img_decode

model_loaded = False
model_digits_svm = None
model_operators_svm = None
model_digits_operators = None
model_digits_cnn = None
model_operators_cnn = None
model_digits_ops_cnn = None

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

def calculation(img_name):
    
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

  # image ratio w/h
  im_ratio = round(im.shape[1]/im.shape[0], 2)
  center_w = im.shape[1]//2
  center_h = im.shape[0]//2
  x = center_w-(CROPPED_W//2)
  y = center_h-(CROPPED_H//2)

  # crop the image
  im = im[y:y + CROPPED_H, x:x + CROPPED_W]

  cv2.imwrite(os.path.join(img_upload_path, "cropped_img.jpg"), im)

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

  orig = im.copy()

  # sort the contours
  (cnts, boundingBoxes) = contours.sort_contours(cnts, method="left-to-right")
  list_roi = []
  roi_h_sum = 0

  im2 = im.copy()
  rows, cols = im2.shape[:2]

  print(roi_upload_path, file=sys.stdout)

  # FIND ROI, loop over the contours
  for (i, c) in enumerate(cnts):
    # compute the bounding box of the contour, then use the
    # bounding box coordinates to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    
    if h > 10 and w > 10:
      cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
      roi = thresh[y:y+h, x:x+w]
      # add roi into a list and calculate sum
      list_roi.append([roi, c])
      roi_h_sum += h
      # write roi images to folder
      cv2.imwrite(os.path.join(pathlib.PurePath(roi_upload_path), "roi_" + str(i) + ".jpg"), roi)

  roi_avg_h = roi_h_sum / len(list_roi)

  # list_operators = {}
  roi_operators = []
  roi_numbers = []

  # save roi to files
  for (i, roi) in enumerate(list_roi):
    roi = roi[0]
    # print(roi.shape)
    if roi.shape[0] < roi_avg_h:
      # list_operators['index'] = i
      # list_operators['roi'] = roi
      roi_operators.append(roi)
      cv2.imwrite(os.path.join(pathlib.PurePath(roi_upload_path), "operator_" + str(i) + ".jpg"), roi)
    else:
      roi_numbers.append(roi)
  
  print(len(roi_operators), file=sys.stdout)

  # 0 - add, 1 - subtract, 2 - divide, 3 - multiply
  operator = -1
  for roi in roi_operators:
    roi = deskew(roi, 28)
    roi = center_extent(roi, (28,28))
    operator = predict_operator_svm(roi)
    # roi_predict = np.reshape(roi, (roi.shape[0], roi.shape[1], 1))
    # prediction_cnn = model_operators_cnn.predict(np.expand_dims(roi_predict, axis=0))
    # operator = np.argmax(prediction_cnn, axis=1)
  
  print('predict operator: ' + str(operator), file=sys.stdout)

  total = 0
  for i, roi in enumerate(roi_numbers):
    roi = deskew(roi, 28)
    roi = center_extent(roi, (28,28))
    # roi_predict = np.reshape(roi, (roi.shape[0], roi.shape[1], 1))
    prediction_cnn = model_digits_cnn.predict(np.expand_dims(roi, axis=0))
    digit = np.argmax(prediction_cnn, axis=1)
    print('predict digit:' + str(digit), file=sys.stdout)

    if i == 0:
      total = digit
    else:
      if operator == 0: # add
        total += digit
      elif operator == 1: # subtract
        total -= digit
      elif operator == 3: # multiply
        total *= digit
  
  print("total: " + str(total), file=sys.stdout)

  results = {}
  if operator != -1 or len(roi_numbers) < 2:
    results['status'] = 1
    # write result to image
    (x, y, w, h) = cv2.boundingRect(list_roi[-1][1])
    cv2.putText(im2, str(int(total)), (x+w+20, y+h),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)

    im_name = img_name.split('.')[0]
    results['calculated_img'] = os.path.join(img_upload_path, im_name + "_calculated.jpg")
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

    results = calculation(filename)

    if results['status'] == 1:
      with open(results['calculated_img'], "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        results['image'] = encoded_string.decode()
  return jsonify(results)

  