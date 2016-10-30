import tensorflow as tf





#https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
def sdc_model(images):
	#Try to normlize inputs along batch axis
	norm = tf.nn.l2_normalize(x, 0)

	conv1 = conv_layer(norm, 'conv1', channels=24, padding='VALID')
	_activation_summary(conv1)

	conv2 = conv_layer(conv1, 'conv2', channels=36, padding='VALID')
	_activation_summary(conv2)

	conv3 = conv_layer(conv2, 'conv3', channels=48, padding='VALID')
	_activation_summary(conv3)

	conv4 = conv_layer(conv3, 'conv4', kernel=3, stride=1, channels=64, padding='VALID')
	_activation_summary(conv4)

	conv5 = conv_layer(conv4, 'conv5', kernel=3, stride=1, channels=64, padding='VALID')
	_activation_summary(conv5)

	flat = tf.reshape(conv5, [-1, 64*1*18])
	fc1 = fc_layer(flat, 'fc1', 100)
	fc2 = fc_layer(fc1, 'fc2', 50)
	fc3 = fc_layer(fc2, 'fc3', 10)
	final_output = fc_layer(fc3, 'fc4', 1, 'tanh')

	return final_output

def loss(logits, angles):
	"""Calculate the L2 loss of predicted inverse turning radius"""
	return tf.nn.l2_loss(tf.sub(logits, angles), name='loss')


def train(loss):
	#Arbitrary hyperparameter
	#Do we want to use SGD with momentum?
	#Probably not, just asking.
	train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)



#Utility functions

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def fc_layer(logits, scope, output_dim, nn_type='relu'):.
	#TODO: Verify that this is correct
	input_dim = logits.get_shape()[1]
	with tf.variable_scope(scope):
		W = weight_variable([input_dim, output_dim])
		b = bias_variable([output_dim])

		if nn_type == 'relu':
			return tf.nn.relu(tf.matmul(logits, W) + b)
		else:
			return tf.nn.tanh(tf.matmul(logits, W) + b)


def conv2d(x, W, stride=2, padding='SAME'):
	return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def conv_layer(images, scope, kernel=5, stride=2, channels=16):
	with tf.variable_scope(scope):
		#TODO: Verify that this is correct
		curr_dim = images.get_shape()[3]
		#TODO: think about adding names to variables
		W = weight_variable([kernel, kernel, curr_dim, channels])
		b = bias_variable([channels])
		return tf.nn.relu(conv2d(images, W, stride=stride) + b)


def _activation_summary(x):
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.histogram_summary(tensor_name + '/activations', x)
	tf.scaler_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

