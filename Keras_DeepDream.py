#This program uses the inception_v3 model to demonstrate Google's Deep Dream process, where the learned features of a given model are superimposed onto an image
#Resulsts in some pretty entertaining images that also help demonstrate how exactly models percieve images
from keras.applications import inception_v3
from keras.preprocessing import image
import cv2
from keras import backend as K
import numpy as np
import scipy

base_image_path = 'your/image/path'

#Necessary functions for keeping track of loss and gradients
def eval_loss_and_grads(x):
	outs = fetch_loss_and_grads([x])
	loss_value = outs[0]
	grad_values = outs[1]
	return loss_value, grad_values

def gradient_ascent(x, iterations, step, max_loss=None):
	for i in range(iterations):
		loss_value, grad_values = eval_loss_and_grads(x)
		if max_loss is not None and loss_value > max_loss:
			break
		print('...Loss value at', i, ':', loss_value)
		x += step *grad_values
	return x

#Simple functions for modifying and saving the images we'll be using
def resize_img(img, size):
	img = np.copy(img)
	factors = (1,
			   float(size[0]) / img.shape[1],
			   float(size[1]) / img.shape[2],
			   1)
	return scipy.ndimage.zoom(img, factors, order=1)

def save_img(img, fname):
	pil_img = deprocess_image(np.copy(img))
	scipy.misc.imsave(fname, pil_img)
	
def preprocess_image(image_path):
	img = image.load_img(image_path)
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = inception_v3.preprocess_input(img)
	return img

def deprocess_image(x):
	if K.image_data_format() == 'channels_first':
		x = x.reshape((3, x.shape[2], x.shape[3]))
		x = x.reshape((3, x.shape[2], x.shape[3]))
		x = x.transpose((1, 2, 0))
	else:
		x = x.reshape((x.shape[1], x.shape[2], 3))
	x/=2.
	x += 0.5
	x *= 255.
	x = np.clip(x, 0, 255).astype('uint8')
	return x

#Setting up our model's parameters, definitley experiment with these to find the features you want present in the image
K.set_learning_phase(0)
model = inception_v3.InceptionV3(weights='imagenet',
								 include_top=False)
print model.summary()
#We pick out layers from model.summary() to map onto the image, with a coefficent showing the weighting that activation will recieve 
layer_contributions = {
	'activation_18': 1.5,
	'activation_38': 1.8,
	'activation_58': .6,
	'activation_78': 1,
	'mixed7': 2.5,
	'conv2d_34': .75,
}

#Get the names of each layer
layer_dict = dict([(layer.name, layer) for layer in model.layers])

#We define the loss in respect to the given layer's coefficent above
loss = K.variable(0.)
for layer_name in layer_contributions:
	coeff = layer_contributions[layer_name]
	activation = layer_dict[layer_name].output
	scaling = K.prod(K.cast(K.shape(activation), 'float32'))
	loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling
	
#Assign our image to a variable, easier to keep track of
dream = model.input

#The core of the 'dream' process:
#steps - step size of gradient ascent
#octaves - the number of different scales we'll run the image on
#octave_scale - the scale by which the image will increase in each scaling
#iterations - number of steps per scale
#max_loss - put a cap on the total loss so as to avoid ugly artifacts
step = 0.01
num_octave = 3
octave_scale = 1.4
iterations = 20
max_loss = 100.

#Assign the gradient variable, which we'll want to maximize when shifting between octaves
grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
outputs = [loss,grads]
fetch_loss_and_grads = K.function([dream], outputs)

#We start the 'dream' process, here we make sure that the iterations will follow successfully and be of the right size
img = preprocess_image(base_image_path)
original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
	shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
	successive_shapes.append(shape)
	
successive_shapes = successive_shapes[::-1]

original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

#And now we process the images themselvess using our pre-defined variables
for shape in successive_shapes:
	print('Processing image shape', shape)
	img = resize_img(img, shape)
	img = gradient_ascent(img,
						  iterations=iterations,
						  step=step,
						  max_loss=max_loss)
	upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
	same_size_original = resize_img(original_img, shape)
	lost_detail = same_size_original - upscaled_shrunk_original_img
	
	#Prints out successive versions of the image
	img += lost_detail
	shrunk_original_img = resize_img(original_img, shape)
	save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
	
#Saves completed image to the Desktop
save_img(img, fname='your_image')