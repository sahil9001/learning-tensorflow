import tensorflow as tf

x = tf.constant([3,5,7],name = "x")
y = tf.constant([1,2,3],name = "y")
z1 = tf.add(x,y,name="z1")
z2 = x * y
z3 = z2-z1
with tf.Session() as sess:
	with tf.summary.FileWriter('summaries', sess.graph) as writer:
		a1, a3 = sess.run([z1,z3])
