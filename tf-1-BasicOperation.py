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

# ==================== Matrix ====================
m1 = tf.constant([[1, 2], [3, 4]])
m2 = tf.constant([[5, 6], [7, 8]])
m1xm2 = tf.matmul(m1, m2)

print("m1 x m2 = ", m1xm2.numpy())

# Special Matrix
# Matrix with all elements are zero
m3 = tf.zeros([3, 4], tf.int8)
m3
input_tensor = m1
# z2 has shape of input_tensor 
m4 = tf.zeros_like(input_tensor)
m4

# Matrix with all elements are 1
m5 = tf.ones([2, 7])
m5
m6 = tf.ones_like(m1)
m6

# Tensor filled with scalar value
m7 = tf.fill([4, 8], 9)
m7

# ==================== Sequence ====================
# tf.linspace(start, stop, num), step = (stop - start)/(num - 1)
z1 = tf.linspace(1.0, 21.0, 6)
z1

# tf.range([start], limit, delta)
start = 20
limit = 5
delta = -2

z2 = tf.range(start, limit, delta)
z2
z3 = tf.range(limit)
z3

# ==================== random ====================
# tf.random.normal(shape, mean, deviation)
r1 = tf.random.normal([3, 4], 0.0, 0.5)
r1


