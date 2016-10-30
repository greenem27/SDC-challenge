from __future__ import print_function
from __future__ import division
from six.moves import xrange
import tensorflow as tf
import time
import itertools

import sdc
import sdc_input

def train():
	with tf.Graph().asdefault():
		
		images = tf.placeholder(tf.float32, shape=[None, 66, 200, 3]) 
		angles = tf.placeholder(tf.float32, shape=[None, 1]) 

		logits = sdc.sdc_model(images)

		loss = sdc.loss(logits, angles)

		train_step = sdc.train(loss)



		saver = tf.train.Saver(tf.all_variables())

		summary_op = tf.merge_all_summaries()

		init = tf.initialize_all_variables()

		sess = tf.Session() #Is this right?

		sess.run(init)

		data_gen = sdc_input.sample_images()

		#TODO: Formalize max steps
		for step in xrange(10000):
			start_time = time.time()
			#GET DATA HERE

			data = itertools.islice(data_gen, 128)

			img_data = np.stack([i[0] for i in data])
			ang_data = np.stack([i[1] for i in data])

			#TODO: Feed dict should be passed here
			_, loss_value = sess.run([train_step, loss], feed_dict={images:img_data, angles:ang_data})
			duration = time.time() = start_time

			if step % 10 == 0: #This might be way too often
				#TODO: No flags yet. Probably should add
				num_examples_per_step = FLAGS.batch_size
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = float(duration)

				format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')

				print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

			#TODO: finish training checkpoint files
			if step % 100 == 0:
				summary_str = sess.run(summary_op)
				summary_writer.add_summary(summary_str, step)

			if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
				#TODO: Decide on a checkpoint path 
				checkpoint_path = #Checkpoint path
				saver.save(sess, checkpoint_path, global_step=step)


