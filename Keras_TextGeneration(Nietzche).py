#This one is pretty fun. We'll be examining a text file containing the works of Nietzche and trying to automatically generate
#similar sounding text. Very difficult to say if it does this well, but it is quite educational/entertaining
import keras
import numpy as np
from keras import layers
import random
import sys

#Load the texfile
path = keras.utils.get_file(
	'nietzsche.txt',
	origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('Corpus length:', len(text))

#Set the variables for the passage we'll pick and how we'll be manipulating it
maxlen = 60
step = 3
sentences = []
next_chars = []

#We clean up the text to make our life easier and print out its total length in setences and characters
for i in range(0, len(text) - maxlen, step):
	sentences.append(text[i: i + maxlen])
	next_chars.append(text[i + maxlen])
print('Number of sequences:', len(sentences))

chars = sorted(list(set(text)))
print('Unique characters:', len(chars))

#Set up the chars to be encoded
char_indices = dict((char, chars.index(char)) for char in chars)


#One hot encode the characters into binary arryas
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		x[i, t, char_indices[char]] = 1
	y[i, char_indices[next_chars[i]]] = 1

#Very simple model, standard/suitable configuration with only two layers, as we're only dealing with text
model = keras.models.Sequential()
#LSTM is our secret weapon here - as our iterations become more accurate, LSTM will preserve the good patterns and discard the noise
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#This 'sample' function is the core part of this program
#It sets up a probability distribution for which character to select after another, with the temperature variable being used as a 
#weighting for entropy i.e. the higher the temp the less reliable and typical the character choice

#So leveragin this function, our program will try its best to crack the statistical code that lies behind Nietzsche's unique style
def sample(preds, temperature=1.0):
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

#We run the process for 60 size 1 epochs
for epoch in range(1, 60):
	print('epoch', epoch)
	model.fit(x, y,
			  batch_size=128,
			  epochs=1)
	
	#We pick a random part of the text file to use as our foundation for each epoch
	start_index = random.randint(0, len(text) - maxlen - 1)
	generated_text = text[start_index: start_index + maxlen]
	print('--- Generating with seed: "' + generated_text + '"')
	
	#We print out the resulsts, 400 characters worth, at increasing degree levels
	for temperature in [0.2, 0.5, 1.0, 1.2]:
		print('------ temperature:', temperature)
		sys.stdout.write(generated_text)

		for i in range(400):
			sampled = np.zeros((1, maxlen, len(chars)))
			for t, char in enumerate(generated_text):
				sampled[0, t, char_indices[char]] = 1.
			
			preds = model.predict(sampled, verbose=0)[0]
			next_index = sample(preds, temperature)
			next_char = chars[next_index]
			generated_text += next_char
			generated_text = generated_text[1:]
			
			sys.stdout.write(next_char)
			sys.stdout.flush()
		print()