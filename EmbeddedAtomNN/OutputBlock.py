import tensorflow as tf
from tensorflow.keras.layers import Layer

class OutputBlock(Layer):
	def __str__(self):
		return "output"+super().__str__()

	def __init__(self, num_nodes, n_out, num_layers, activation_fn=None, seed=None, drop_rate=0.0, dtype=tf.float32, name="OutputBlock"):
		super().__init__(dtype=dtype,name=name)

		self._activation_fn = activation_fn

		if dtype==tf.float64 :
			tf.keras.backend.set_floatx('float64')

		self._drop_rate = drop_rate
		initializer = tf.keras.initializers.GlorotNormal(seed=seed)

		self._layers = []
		for i in range(num_layers):
			self._layers.append( 
				tf.keras.layers.Dense(num_nodes, activation=activation_fn, 
				kernel_initializer=initializer, bias_initializer='zeros',name=name+"/Hidden_"+str(i)+"/", use_bias=True, dtype=dtype)
				)

		#initializer = tf.keras.initializers.GlorotNormal(seed=seed)
		initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=seed)
		self._dense = tf.keras.layers.Dense(n_out, activation=None, 
				kernel_initializer=initializer, bias_initializer='zeros',
				#kernel_initializer='zeros', bias_initializer='zeros',
				use_bias=False, dtype=dtype,name=name+'/Dense')

	@property
	def activation_fn(self):
		return self._activation_fn
    
	@property
	def layers(self):
		return self._layers

	@property
	def dense(self):
		return self._dense

	@property
	def drop_rate(self):
		return self._drop_rate

	def __call__(self, x):
		for i in range(len(self.layers)):
			x = self.layers[i](x)
			if self.drop_rate is not None:
				x = tf.nn.dropout(x, self.drop_rate)
		return self.dense(x)
