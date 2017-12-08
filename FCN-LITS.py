'''
building FCN
'''
__author__ = 'LionZhu-List'

import trainRecords as read_mat
import tensorflow as tf 
import numpy as np
import scipy.misc as smc


tf.app.flags.DEFINE_integer("batch_size", '2', 'batch size of training')
tf.app.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.app.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.app.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.app.flags.DEFINE_bool('Debug', "False", "Debug mode: True/ False")
tf.app.flags.DEFINE_string('Mode', "Train", "Mode Train/ Test/ Visualize")

FLAGS = tf.app.flags.FLAGS


NUM_OF_CLASS = 3
IMAGE_DEPTH = 64
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNEL = 1
MAX_ITERATION = 50000 # 10 epoch
TEST_RAW = 98

recordPath = '/media/lion/000E1ACB000CD6A4/TFprojects/FCN_LITS/'
#resultPath = '/media/lion/0E450E250E450E25/TFexp/'

class  vgg16FCN:
	'''
	define the vgg layers
	'''
	def __init__(self, vgg16_path = None, trainable = True, dropout = 0.5):
		if vgg16_path is None:
			self.data_dict = {}
		else:
			pass

		self.var_dict = {}
		self.trainable = trainable
		self.dropout = dropout


	def buildVGG(self, tensor_in):
		layers = {
		'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

		'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
		
		'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
		'relu3_3', 'pool3',
		
		'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'pool4',
        
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'pool5'
		}
		print ("model starts")
		print (tensor_in.get_shape())
		#assert tensor_in.get_shape() == [1, 64, 128, 128, 1]  #[batch, in_depth, in_height, in_width, in_channels]

		with tf.variable_scope("vggNet"):
			self.conv1_1 = self.conv_layer(tensor_in, 1, 64, "conv1_1")   # 1 is in channels, 64 is out channels
			self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
			self.pool1 = self.max_pool3d(self.conv1_2, 'pool1')

			self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
			self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
			self.pool2 = self.max_pool3d(self.conv2_2, 'pool2')

			self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
			self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
			self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
			self.pool3 = self.max_pool3d(self.conv3_3, 'pool3')

			self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
			self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
			self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
			self.pool4 = self.max_pool3d(self.conv4_3, 'pool4')

			self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
			self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
			self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
			self.pool5 = self.max_pool3d(self.conv5_3, 'pool5')
	  
	  		return self.pool5, self.pool4, self.pool3

	def conv_layer(self, in_tensor, in_chnls, out_chnls, name):
		with tf.variable_scope(name):
			filt, conv_bias = self.get_conv_var(3, in_chnls, out_chnls, name)
			#filt = tf.cast(filt_a, tf.float64)
			#conv_bias = tf.cast(conv_bias_a, tf.float64)

			conv = tf.nn.conv3d(in_tensor, filt, [1, 1, 1, 1, 1], padding = 'SAME')
			bias = tf.nn.bias_add(conv, conv_bias)
			reluRes = tf.nn.relu(bias)

			return reluRes

	def get_conv_var(self, filter_size, in_chnls, out_chnls, name):
		init_value = tf.truncated_normal([filter_size, filter_size, filter_size, in_chnls, out_chnls], 0.0, 0.01)

		filters = self.get_var(init_value, name, 0, name+'_filters')

		init_value = tf.truncated_normal([out_chnls], 0.0, 0.01)

		biases = self.get_var(init_value, name, 1, name+'_biases')

		#filters = tf.cast(filters_a, tf.float64)
		#biases = tf.cast(biases_a, tf.float64)

		return filters, biases

	def get_conv_var_transpose(self, filter_size, out_chnls, in_chnls, name):
		init_value = tf.truncated_normal([filter_size, filter_size, filter_size, out_chnls, in_chnls], 0.0, 0.01)

		filters = self.get_var(init_value, name, 0, name+'_filters')

		init_value = tf.truncated_normal([out_chnls], 0.0, 0.01)

		biases = self.get_var(init_value, name, 1, name+'_biases')

		#filters = tf.cast(filters_a, tf.float64)
		#biases = tf.cast(biases_a, tf.float64)

		return filters, biases

	def get_var(self, init_value, name, indx, var_name):
		if self.data_dict is None:
			value = init_value

		if self.trainable:
			var = tf.Variable(init_value, name = var_name)

		self.var_dict[(name, indx)] = var

		return var
	
	def max_pool3d(self, in_tensor, name):
		#in_tensor_a = tf.cast(in_tensor, tf.float32)
		maxpool_res = tf.nn.max_pool3d(in_tensor, ksize = [1, 2, 2, 2, 1], strides = [1, 2, 2, 2, 1], padding = 'SAME', name = name)
		return maxpool_res

	
	def conv3d_tranpose_strided(self, x, w, b, outshape = None, stride = 2):
		if outshape is None:
			outshape = x.get_shape().as_list()

		conv = tf.nn.conv3d_transpose(x, w, outshape, strides = [1, stride, stride, stride, 1], padding = 'SAME')
		return tf.nn.bias_add(conv, b)


	def segNet(self, image, keep_prob):
		'''
		define the semantic seg net
		'''
		with tf.variable_scope("segNet"):
			self.pool_5, self.pool_4, self.pool_3 = self.buildVGG(image)
			
			w6, b6 = self.get_conv_var(7, 512, 256, name='conv6')
			self.conv6 = tf.nn.bias_add(tf.nn.conv3d(self.pool_5, w6, [1, 1, 1, 1, 1], padding = 'SAME'), b6)
			self.relu6 = tf.nn.relu(self.conv6, name = 'relu6')
			self.relu_dpout6 = tf.nn.dropout(self.relu6, keep_prob = keep_prob) 

			w7, b7 = self.get_conv_var(1, 256, 256, name='conv7')
			self.conv7 = tf.nn.bias_add(tf.nn.conv3d(self.relu_dpout6, w7, [1, 1, 1, 1, 1], padding = 'SAME'), b7)
			self.relu7 = tf.nn.relu(self.conv7, name = 'relu7')
			self.relu_dpout7 = tf.nn.dropout(self.relu7, keep_prob = keep_prob) 

			w8, b8 = self.get_conv_var(1, 256, NUM_OF_CLASS, name='conv8')
			self.conv8 = tf.nn.bias_add(tf.nn.conv3d(self.relu_dpout7, w8, [1, 1, 1, 1, 1], padding = 'SAME'), b8)


			# now to upscale to ori imagesize
			shape_deconv1 = self.pool_4.get_shape()   # shape is [batch, depth, height, width, channels]
			w_deconv1, b_deconv1 = self.get_conv_var_transpose(4, shape_deconv1[4].value, NUM_OF_CLASS, name='deconv1')
			deconv1 = self.conv3d_tranpose_strided(self.conv8, w_deconv1, b_deconv1, outshape = tf.shape(self.pool_4))
			self.fuse_1 = tf.add(deconv1, self.pool_4, name = "fuse_1")

			shape_deconv2 = self.pool_3.get_shape()  # shape_deconv1,2 is a tensor while shape_deconv3 is a tuple
			w_deconv2, b_deconv2 = self.get_conv_var_transpose(4, shape_deconv2[4].value, shape_deconv1[4].value, name = 'deconv2')
			deconv2 = self.conv3d_tranpose_strided(self.fuse_1, w_deconv2, b_deconv2, outshape = tf.shape(self.pool_3))
			self.fuse_2 = tf.add(deconv2, self.pool_3, name = 'fuse_2')

			shape = tf.shape(image)    # image shape is [batch, depth, height, width, channels]
			shape_deconv3 = tf.stack([shape[0], shape[1], shape[2], shape[3], NUM_OF_CLASS])  # channels to classNUM
			w_deconv3, b_deconv3 = self.get_conv_var_transpose(16, NUM_OF_CLASS, shape_deconv2[4].value, name = 'deconv3')
			deconv3 = self.conv3d_tranpose_strided(self.fuse_2, w_deconv3, b_deconv3, outshape = shape_deconv3, stride = 8)

			#pred is the position of argmax in channels, is 3-D
			# dimension = 4 that is argmax at the 4-axis
			annotation_pred = tf.argmax(deconv3, dimension = 4, name = 'prediction') 

		return tf.expand_dims(annotation_pred, dim = 4), deconv3

	def fcn_run(self):
		keep_probability = tf.placeholder(tf.float32, name = "keep_prob")
		image = tf.placeholder(tf.float32, shape = [None, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name = 'input_img')

		annotation = tf.placeholder(tf.int32, shape = [None, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name = 'annotation')
		
		pred_annotation, logits = self.segNet(image, keep_probability)
		print (type(image))
		#tf.summary.image("image", image, max_outputs = 3)
		#tf.summary.image("groud_truth", tf.cast(annotation, tf.uint8), max_outputs = 3)
		#tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs = 3)

		loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,
																			labels = tf.squeeze(annotation, squeeze_dims=[4]),
																			name = 'entropy_loss')))
		#loss = tf.cast(loss_a, tf.float32)
		tf.summary.scalar('entropy_loss', loss)

		trainable_var = tf.trainable_variables()
		if FLAGS.Debug:
			for var in trainable_var:
				tf.summary.histogram(var.op.name, var)
				tf.add_to_collection('reg_loss', tf.nn.l2_loss(var))
		train_op = self.training(loss, trainable_var)

		print("setting up summary op...")
		summary_op = tf.summary.merge_all()
		
		print("setting up image reader...")
		img_data_a, label_data_a = read_mat.read_decode(recordPath + "train.tfrecords")
		img_data = tf.cast(img_data_a, tf.float32)
		label_data = tf.cast(label_data_a, tf.float32)

		img_train_batch_a, label_train_batch_a = tf.train.shuffle_batch([img_data, label_data],
																	batch_size=4, capacity=8, min_after_dequeue=4)
		img_train_batch = tf.expand_dims(img_train_batch_a, dim = 4)
		label_train_batch = tf.expand_dims(label_train_batch_a, dim = 4)
		print (img_train_batch.shape)
		print ('setup session...')
		sess = tf.Session()
		saver = tf.train.Saver()
		summary_writer = tf.summary.FileWriter(FLAGS.logs_dir,sess.graph)

		sess.run(tf.global_variables_initializer())
		threads = tf.train.start_queue_runners(sess)

		if FLAGS.Mode == 'Train':
			for itr in xrange (MAX_ITERATION):
				#img_batch shape:[batch_size, depth, height, width, chanl], type: ndarray
				img_batch, l_batch = sess.run([img_train_batch, label_train_batch ])

				print (itr, img_batch.shape)
				feed = {image: img_batch, annotation: l_batch, keep_probability: 0.85}
				sess.run(train_op, feed_dict = feed)
				train_loss_print, summary_str = sess.run([loss, summary_op],  feed_dict = feed)
				print (train_loss_print)
				summary_writer.add_summary(summary_str, itr)

				if (itr+1) % 5000 == 0:
					saver.save(sess,recordPath + '/netCKPT/modle', global_step = itr)
				#.......................... test test test test test..............................................#
				if itr == (MAX_ITERATION - 1):
					test_img_data_a, test_label_data_a = read_mat.read_decode(recordPath + "test.tfrecords")
					test_img_data = tf.cast(test_img_data_a, tf.float32)
					test_label_data = tf.cast(test_label_data_a, tf.float32)

					test_img_train_batch_a, test_label_train_batch_a = tf.train.batch([test_img_data, test_label_data],
																					batch_size=1, capacity=2)

					test_img_train_batch = tf.expand_dims(test_img_train_batch_a, dim=4)
					test_label_train_batch = tf.expand_dims(test_label_train_batch_a, dim=4)
					threads = tf.train.start_queue_runners(sess)
					for test_itr in range (TEST_RAW):
						test_img_batch, test_l_batch = sess.run([test_img_train_batch, test_label_train_batch])
						test_feed = {image: test_img_batch, annotation: test_l_batch, keep_probability: 0.85}

						test_loss, test_pred_annotation = sess.run([loss, pred_annotation], feed_dict = test_feed)
						#tep  = test_pred_annotation[0, 30, :, :, 0]
						#print (tep.shape)
						#np.savetxt('pred30.csv', tep, delimiter=',')
						#label_tosave = tf.cast(tf.reshape(test_l_batch, [1, 128, 128, 64, 1]), tf.int16)
						#pred_tosave = tf.cast(tf.reshape(test_pred_annotation, [1, 128, 128, 64, 1]), tf.int16)
						self.save_imgs(test_itr, test_l_batch, test_pred_annotation)


		elif FLAGS.Mode == 'Visualize':
			pass


	def training(self, loss_val, va_list):
		optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
		grads = optimizer.compute_gradients(loss_val, var_list = va_list)
		if FLAGS.Debug:
			for grad, var in grads:
				tf.summary.histogram(grad, var)

		return optimizer.apply_gradients(grads)

	def save_imgs(self, test_num, label_batch, pred_batch):
		for dept in range (IMAGE_DEPTH):
			label_img_mat = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 3))
			pred_img_mat = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 3))
			for i in range (IMAGE_HEIGHT):
				for j in range(IMAGE_WIDTH):
					if label_batch[0, dept, i, j,  0] == 0:
						label_img_mat[i, j, 0] = label_img_mat[i, j, 1] =  label_img_mat[i, j, 2] = 128  # backgroud Gray ##R G B
					if label_batch[0, dept, i, j,  0] == 1:
						label_img_mat[i, j, 0] = 255
						label_img_mat[i, j, 1] = 69
						label_img_mat[i, j, 2] = 0     # liver Red
					if label_batch[0, dept, i, j, 0] == 2:
						label_img_mat[i, j, 0] = 64
						label_img_mat[i, j, 1] = 0
						label_img_mat[i, j, 2] = 128	#lesion Purple

					if pred_batch[0, dept, i, j,  0] == 0:
						pred_img_mat[i, j, 0] = pred_img_mat[i, j, 1] =  pred_img_mat[i, j, 2] = 128  # backgroud Gray
					if pred_batch[0, dept, i, j,  0] == 1:
						pred_img_mat[i, j, 0] = 255
						pred_img_mat[i, j, 1] = 69
						pred_img_mat[i, j, 2] = 0    # liver Red
					if pred_batch[0, dept, i, j, 0] == 2:
						pred_img_mat[i, j, 0] = 64
						pred_img_mat[i, j, 1] = 0
						pred_img_mat[i, j, 2] = 128  #lesion Purple

			#scipy.misc.imsave(recordPath + 'imgs/' + '%d-%d-mask.png' % (test_num, dept), label_img_mat)
			#scipy.misc.imsave(recordPath + 'imgs/' + '%d-%d-pred.png' % (test_num, dept), pred_img_mat)
			smc.toimage(label_img_mat, cmin=0.0, cmax=255).save(recordPath + 'imgs/' + '%d-%d-mask.png' % (test_num, dept))
			smc.toimage(pred_img_mat, cmin=0.0, cmax=255).save(recordPath + 'imgs/' + '%d-%d-pred.png' % (test_num, dept))
			# label_img_mat = None
			# pred_img_mat = None


def main(agrv = None):
	#FLAGS.para_name
	fcn = vgg16FCN()
	fcn.fcn_run()
	print ('finished!')


if __name__ == '__main__':
	tf.app.run()
































		

