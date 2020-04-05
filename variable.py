import tensorflow as tf
def forward_pass(w,x):
	return tf.matmul(w,x)
def train_loop(x, niter = 5):
	with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
		w = tf.get_variable("weights",shape=(1,2),initializer=tf.truncated_normal_initializer(),trainable=True)
	preds = []
	for k in xrange(niter):
		preds.append(forward_pass(w,x))
		w = w + 0.1
	return preds

with tf.Session() as sess:
	preds = train_loop(tf.constant([[3.2,5.1,7.2],[4.3,6.2,8.3]]))
	tf.global_variables_initializer().run()
	for i in xrange(len(preds)):
		print "{}:{}".format(i,preds[i].eval())

