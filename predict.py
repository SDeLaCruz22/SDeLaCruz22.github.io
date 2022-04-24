import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from keras.models import load_model
from keras.preprocessing import image
IMG_SIZE = 255

'''
HAVE IT RUN A BUN OF TEST FILES AND GET THE ACCURACY
'''


#get image path
image_path= "sd2.jpg" #python Predict.py img.jpg

#Load Model
loaded_model = tf.keras.models.load_model('test.h5')

print(loaded_model.layers[0])
loaded_model.layers[0].input_shape #(None, 255, 255, 3)

img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
plt.imshow(img)
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

#result=loaded_model.predict_classes(img)
result=loaded_model.predict(img)

#get_label_name = metadata.features['label'].int2str
plt.title(result[0][0])
plt.show()

#print(loaded_model.predict_proba(img))
print(result)
#print(loaded_model.predict_proba(img))

print("this is result[0]")
print(result[0])
#need to verify which index corresponds to which label