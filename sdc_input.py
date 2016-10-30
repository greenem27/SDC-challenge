from prefetch_generator import background

import tensorflow as tf

from random import random
from bisect import bisect
from scipy import misc
import pandas as pd

#Just get static pictures and such
#Be sure to crop
#Make sure they are 66x200x3


#TODO: Put input code into ipython and check to see that the preprocessing steps
#are actually working and we are getting out what we want.


@background(max_prefetch=128)
def sample_images(training_set=True, 
				n_frames=int(10.0/0.03),
				steering_offset=1):
	"""Returns a list of n filenames for pictures 
	in consecutive order, paired with the steering commands.
	steering offset is the number of frames in the future
	that we are trying to predict."""
	folders = glob.glob('./data_dir/*')
	lengths = [len(i) for i in (glob.glob(i + '/*') for i in folders)]
	f_and_l = list(zip(folders, lengths))

	training_frac = 0.8

	while True:

		curr_folder = weighted_choice(f_and_l)
		angles = get_angles(curr_folder)

		if training_set:
			#Use first 80% of data
			file_num = random.randint(0, int(0.8 * n_files))
		else:
			#Use last 20% of data
			file_num = random.randint(int(0.8 * n_files + 1), n_files)
			# (.8 * n_files, 1) ??

		#Sorting all these files might matter, it might not.
		#Since it is asynchronous, it probably doesn't matter, but
		#if it does, we should just keep a sorted variable around.
		filename = sorted(glob.glob(curr_folder+'/*.jpg')[file_num], 
			key=lambda x: int(x.replace('frame', '').replace('.jpg', '')))

		image = open_image(curr_folder + '/' + filename)

		image = pre_process_image(image, 0)

		steering_angle = retrieve_angle(curr_folder, file_num, n_files)


		yield image, steering_angle


#Function I stole from stackoverflow
def weighted_choice(choices):
    values, weights = zip(*choices)
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = random() * total
    i = bisect(cum_weights, x)
    return values[i], weights[i]

# def retrieve_angle(folder, n, n_files):
# 	#TODO: Write code to subsample steering angles
# 	#That way we don't have to deal with that here.
# 	#so they match the number of camera images.
# 	df = pd.read_csv(folder+'_steering_angles.csv')

# 	#TODO finish this function.
# 	return

def reduce_angles(angle_list, n, n_files):
    """Take a steering wheel list and the number of images sampled from the
    camera video and reduce the angle_list to be the same size so there is a
    1 to 1 correspondance."""
    ls = []
    n = len(angle_list)
    for i in range(n_files):
        frac = 1.0 * i / n_files
        curr_spot = len(angle_list) * frac
        ind1, ind2 = int(curr_spot - 1), int(curr_spot)
        val1, val2 = angle_list[ind1], angle_list[ind2]
        ls.append(val1 + ((val2 - val1) * (frac % 1)))
    return ls[n]


def steer_to_tanh(steering_angle):
	#should return a number between -1 and 1
	#just scrunch all radians to be between 3 rotations left and right
	#TODO: Check that this makes sense
	return steering_angle / (3*3.14)

def convert_tanh_steering(tanh_angle):
	return tanh_angle * (3*3.14)

def open_image(img_filepath):
	"""Returns the resulting jpeg image as
	a numpy array."""

	#TODO: finish this function and convert it
	#TODO: Check that this returns image in row major format
	#To a numpy array. Or use tf.decode_jpeg
	return misc.imread(img_filepath)


def pre_process_image(image, px_penalty_shift):
	#TODO: Fix function so that it only acts on one image.
	#Batch size should not be a factor here
	# CENTER_PIXEL_CROP = 127
	# """Crop off top half of image.
	# Center crop resulting image, shifted
	# based off of px_penalty_shift."""
	# #We want the decisons of the algorithm to shift the window over.
	# cut_half = tf.slice(image, [0, 119 ,0], [BATCH_SIZE, 480, 640])
	# penalty_shift = tf.slice(image_cut, 
	# 	[0, 0, CENTER_PIXEL_CROP + px_penalty_shift],
	# 	[BATCH_SIZE, 120, CENTER_PIXEL_CROP + px_penalty_shift + (128*3)])

	image = image[479-210:, :, :]
	image = misc.imresize(image, (66,200,3))

	#TODO: Consider adding guassian noise
	return image

def get_angles(img_folder_name):
	#return the angles file => dataframe
	num = img_folder_name[-1]
	#TODO: Use correct data folder
	angle_file = [i for i in os.listdir('data_folder') if '.csv' in i and num in i][0]
	#TODO: Careful about paths
	df = pd.read_csv(angle_file)
	return list(df['feed.steering_angles'])

