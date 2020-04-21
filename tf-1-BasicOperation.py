import tensorflow as tf

# Define tensor constant
a = tf.constant(3)
b = tf.constant(4)
c = tf.constant(5)

# Operation
add = tf.add(a, c)
sub = tf.subtract(a, b)
mul = tf.multiply(b, c)
div = tf.divide(b, c)

sum = tf.reduce_sum([a, b, c])
mean = tf.reduce_mean([a, b, c])
min = tf.reduce_min([a, b, c])
max = tf.reduce_max([a, b, c])

# Print value
print("a = ", a.numpy())
print("b = ", b.numpy())
print("c = ", c.numpy())

print("add = ", add.numpy())
print("sub = ", sub.numpy())
print("mul = ", mul.numpy())
print("div = ", div.numpy())

print("sum = ", sum.numpy())
print("mean = ", mean.numpy())
print("min = ", min.numpy())
print("max = ", max.numpy())

# Matrix
m1 = tf.constant([[1, 2], [3, 4]])
m2 = tf.constant([[5, 6], [7, 8]])
m1xm2 = tf.matmul(m1, m2)

print("m1 x m2 = ", m1xm2.numpy())