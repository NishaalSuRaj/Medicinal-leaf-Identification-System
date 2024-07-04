from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC
import numpy as np

app = Flask(__name__)

dependencies = {
    'auc_roc': AUC
}

verbose_name = {
0: 'Alpinia Galanga (Rasna)',
1: 'Amaranthus Viridis (Arive-Dantu)', 
2: 'Artocarpus Heterophyllus (Jackfruit)',
3: 'Azadirachta Indica (Neem)',
4: 'Basella Alba (Basale)',
5: 'Brassica Juncea (Indian Mustard)',
6: 'Carissa Carandas (Karanda)',
7: 'Citrus Limon (Lemon)',
8: 'Ficus Auriculata (Roxburgh fig)',
9: 'Ficus Religiosa (Peepal Tree)',
10: 'Hibiscus Rosa-sinensis',
11: 'Jasminum (Jasmine)',
12: 'Mangifera Indica (Mango)',
13: 'Mentha (Mint)',
14: 'Moringa Oleifera (Drumstick)',
15: 'Muntingia Calabura (Jamaica Cherry-Gasagase)',
16: 'Murraya Koenigii (Curry)',
17: 'Nerium Oleander (Oleander)',
18:	'Nyctanthes Arbor-tristis (Parijata)',
19: 'Ocimum Tenuiflorum (Tulsi)',
20: 'Piper Betle (Betel)',
21: 'Plectranthus Amboinicus (Mexican Mint)',
22: 'Pongamia Pinnata (Indian Beech)',
23: 'Psidium Guajava (Guava)',
24: 'Punica Granatum (Pomegranate)',
25: 'Santalum Album (Sandalwood)',
26: 'Syzygium Cumini (Jamun)',
27: 'Syzygium Jambos (Rose Apple)',
28: 'Tabernaemontana Divaricata (Crape Jasmine)',
29: 'Trigonella Foenum-graecum (Fenugreek)'

 }
verbose_name1 = {
	0: 'Asian Holly Oak',
	1: 'bartard myrobalan',
	2: 'Clearing Nut',
	3: 'Fevernut',
	4: 'Galangal',
	5: 'Haritaki',
	6: 'Horsepurslane',
	7: 'Indian Sarsaparilla',
	8: 'IndianBirthwort',
	9: 'Jatamansi',
	10: 'nutgrass',
	11: 'Nutmeg',
	12: 'Papaya',
	13: 'Pellitory',
	14: 'Pepper'
}



model = load_model('new.h5')

#chaNGE MODEL
model1 = load_model('rawm.h5')

def predict_label(img_path):
	test_image = image.load_img(img_path, target_size=(224,224))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 224,224,3)

	predict_x=model.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	print("Verbose :  " , classes_x)
	
	return verbose_name[classes_x[0]]

def predict_label1(img_path):
	test_image = image.load_img(img_path, target_size=(224,224))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 224,224,3)

	predict_x=model1.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	print("Verbose :  " , classes_x)
	#verbose1
	return verbose_name1[classes_x[0]]

 
@app.route("/")
@app.route("/first")
def first():
	return render_template('index.html')
  
    
@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")

#Try
@app.route("/newindex", methods=['GET', 'POST'])
def index1():
	return render_template("newindex.html")

#try
@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/tests/" + img.filename	
		img.save(img_path)

		predict_result = predict_label(img_path)

	return render_template("prediction.html", prediction = predict_result, img_path = img_path)


@app.route("/submit1", methods = ['GET', 'POST'])
def get_output1():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/tests/" + img.filename	
		img.save(img_path)

		predict_result = predict_label1(img_path)

	return render_template("prediction.html", prediction = predict_result, img_path = img_path)



	
if __name__ =='__main__':
	app.run(debug = True)


	

	


