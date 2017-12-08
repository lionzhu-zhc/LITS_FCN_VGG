'''
read from .raws
'''

__author__ = 'LionZhu-List'
import os
import scipy.io as sio
import tensorflow as tf
import numpy as np
from PIL import Image

cwd = os.getcwd()
print cwd

recordPath = '/media/lion/000E1ACB000CD6A4/TFprojects/FCN_LITS/'
matPath = './trainDir.txt'
matSet = open(matPath, 'r').read().splitlines()
matNum = len(matSet)

def createTrainRecord():
	writer = tf.python_io.TFRecordWriter(recordPath + "train.tfrecords")
	for i in xrange(matNum):

		img_fid = open(matSet[i]+'data.raw', 'rb')
		msk_fid = open(matSet[i]+'mask.raw', 'rb')
		img = img_fid.read()
		msk = msk_fid.read()
		#idata = Image.frombytes('F', (128,128,64), img, 'raw', 'F;64F')

		example = tf.train.Example(features = tf.train.Features(feature = {
			"imgdata": tf.train.Feature(bytes_list = tf.train.BytesList(value = [img])),
			"labels": tf.train.Feature(bytes_list = tf.train.BytesList(value = [msk]))
			}))

		writer.write(example.SerializeToString())
	writer.close()


def read_decode(filename):
	file_queue = tf.train.string_input_producer([filename ])
	reader = tf.TFRecordReader()

	_, serialized_example = reader.read(file_queue)

	features = tf.parse_single_example(serialized_example,
										features = {
										'imgdata':tf.FixedLenFeature([], tf.string),
										'labels':tf.FixedLenFeature([], tf.string)
										})
	imgdata = tf.decode_raw(features['imgdata'],tf.float64)
	imgdata = tf.reshape(imgdata, [64,128, 128])
	imgdata_tranp = tf.transpose(imgdata, perm=[0,2,1])
	labels = tf.decode_raw(features['labels'],tf.float64)
	labels = tf.reshape(labels, [64,128, 128])
	labels_tranp = tf.transpose(labels, perm=[0,2,1])

	return imgdata_tranp, labels_tranp


if __name__ == '__main__':
	createTrainRecord()

	# img_data shape: [batch_size, depths, row(height), cols(width)] ,
	# need to add channels =1 at the end of cols(width) to 5D tensor
	img_data_a, label_data_a = read_decode(recordPath + "train.tfrecords")
	img_data = tf.cast(img_data_a, tf.float32)
	label_data = tf.cast(label_data_a, tf.float32)
	img_batch, label_batch = tf.train.shuffle_batch([img_data, label_data], 
													batch_size = 1, capacity = 10, min_after_dequeue = 5)
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		threads = tf.train.start_queue_runners(sess)

		for i in range(10):
			x, y = sess.run([img_batch, label_batch])
			print(type(x))
			print(y.shape)
			print (x[0, 39, 50, 50])
			print (y[0, 39, 50, 50])

		#np.savetxt('xx32.csv', x[0, 39, :, :], delimiter=',', fmt = '%.4f' )











