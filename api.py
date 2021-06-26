import os
import io
import time
import tensorflow as tf
from tensorflow import keras 
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.models import load_model
from numpy.lib.type_check import imag
from keras.models import load_model
import cv2
import numpy as np
from flask import request
from flask import render_template
from tensorflow.keras.models import Model, Sequential
from keras import backend as K
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications.resnet50 import ResNet50, preprocess_input
from matplotlib import pyplot as plt
from tensorflow.python.client import device_lib
from datetime import datetime
from flask import Flask, request, Response
from werkzeug.utils import secure_filename
from db import db_init, db
from models import Img

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


app = Flask(__name__)
UPLOAD_FOLDER = "/Users/wangweizhong/Desktop/Medical_Treatment_System/static"
DEVICE = "cuda"
MODEL = None

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///img.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)

labels = {0: "COVID19", 1:"NORMAL"}


class Model():
	def __init__(self):
		self.model = tf.keras.models.load_model('./vgg16_weight/covid_19_model.h5')
	def returnModel(self):	
		return self.model
		
def predict(image_path, model):
	data =[]
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	data.append(image)
	data = np.array(data) / 255.0

	ans = MODEL.predict(data, batch_size=1)
	return ans.tolist()[0][0]


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(
            last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.3):
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    img = cv2.imread(img_path)
    fig, ax = plt.subplots()
    im = cv2.resize(cv2.cvtColor(cv2.imread(img_path),
                             cv2.COLOR_BGR2RGB), (img.shape[1], img.shape[0]))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # 以 0.6 透明度繪製原始影像
    ax.imshow(im, alpha=0.7)

    # 以 0.4 透明度繪製熱力圖
    ax.imshow(heatmap, cmap='jet', alpha=0.3)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False) 
    plt.savefig(cam_path,bbox_inches="tight")

def showModelDetial(img_path, file_name ):  
	preprocess_input = keras.applications.xception.preprocess_input
	decode_predictions = keras.applications.xception.decode_predictions
	last_conv_layer_name = "block5_conv3"
	# The local path to our target image
	data = []
	image = cv2.imread(img_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	data.append(image)
	data = np.array(data) / 255.0
	model = MODEL
	model.layers[-1].activation = None	
	# Print what the top predicted class is
	preds = model.predict(data, 1)
	print(preds)
	pred_class = np.argmax(preds[0])
	print(pred_class)
	# Generate class activation heatmap
	heatmap = make_gradcam_heatmap(data, model, last_conv_layer_name, pred_class)
	save_and_display_gradcam(
		img_path, heatmap, os.path.join(UPLOAD_FOLDER , "details" ,file_name))
	# return new_loc
	
	
@app.route("/predict", methods=["GET","POST"])
def upload_predictions():
	if request.method=="POST":
		image_file = request.files["image"]
		if image_file:
			image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
			image_file.save(image_location)
			pred = predict(image_location, MODEL)
			print(pred)
			print(image_location, image_file.filename)
			showModelDetial(image_location,image_file.filename )	
			return render_template("predict.html", prediction=round(pred, 4), image_loc=image_file.filename)
			# return render_template("index.html", prediction=round(pred,4), image_loc= image_file.filename) 
	return render_template("predict.html", prediction=0, image_loc =None)



@app.route('/upload', methods=['GET', 'POST'])
def upload():
	if request.method == "POST":
		pic = request.files['pic']
		if not pic:
			return 'No pic uploaded!', 400
		filename = secure_filename(pic.filename)
		mimetype = pic.mimetype
		if not filename or not mimetype:
			return 'Bad upload!', 400
		#parameter  = request.form['input name']
		name = request.form['patient_name']
		birthday = request.form['birthday']
		print(name)
		print(birthday)
		img = Img(img=pic.read(), name=name, mimetype=mimetype, birthday=birthday)
		db.session.add(img)
		db.session.commit()
	return render_template("upload.html")
	
@app.route("/", methods=["GET","POST"])
def mainPage():
	return render_template("main.html")
	

@app.route("/query", methods=["GET", "POST"])
def query():	
	if request.method == "POST":
		
		name = request.form['patient_name']
		img = Img.query.filter_by(name = name).first()
		#有這位病患
		try:
			image = img.img
			# print(img.img)
		except:
		#病患不存在 傳回錯誤
			image= None
			return 'Patient  Not Found!', 40
		#讀取byte64結構，存成圖片並傳入地址
		nparr = np.fromstring(img.img, np.uint8)
		image = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
		now = datetime.now().timestamp()
		cv2.imwrite('./static/displayDB/temp'+str(now)+'.png', image)
		time.sleep(2)
		# print(image='temp.png', name=img.name, birthday=img.birthday)
		return render_template("query.html", image='temp'+str(now)+'.png', name=img.name, birthday=img.birthday)
	return render_template("query.html", image= None )

@app.route('/<int:id>')
def get_img(id):
    img = Img.query.filter_by(id=id).first()
    if not img:
        return 'Img Not Found!', 404

    return Response(img.img, mimetype=img.mimetype)
    

if __name__ == "__main__":
	MODEL = Model().returnModel()
	app.run(port=7700, debug = True)
	
	

	
