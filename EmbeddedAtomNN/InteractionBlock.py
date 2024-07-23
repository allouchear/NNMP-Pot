import tensorflow as tf
from tensorflow.keras.layers import Layer

class InteractionBlock(Layer):
	def __str__(self):
		return "interaction_block"+super().__str__()

	def __init__(self, num_nodes, num_layers, activation_fn=None, seed=None, drop_rate=0.0, dtype=tf.float32, name="InteractionBlock"):
		super().__init__(dtype=dtype,name=name)
            #interaction layer
		self._drop_rate = drop_rate
		initializer = tf.keras.initializers.GlorotNormal(seed=seed)
		#initializer2 = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=seed)
		initializer2 = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed)
		self._layers = []
		self._activation_fn = activation_fn
		for i in range(num_layers):
			self._layers.append( 
				tf.keras.layers.Dense(num_nodes, activation=activation_fn, 
				#kernel_initializer=initializer, bias_initializer='zeros',name=name+"/Hidden_"+str(i)+"/", use_bias=True, dtype=dtype)
				#kernel_initializer=initializer2, bias_initializer='zeros',name=name+"/Hidden_"+str(i)+"/", use_bias=True, dtype=dtype)
				kernel_initializer=initializer, bias_initializer=initializer2,name=name+"/Hidden_"+str(i)+"/", use_bias=True, dtype=dtype)
				)

	@property
	def activation_fn(self):
		return self._activation_fn
    
	@property
	def layers(self):
		return self._layers

	@property
	def drop_rate(self):
		return self._drop_rate

	def __call__(self, x):
		#if self.activation_fn is not None:
		#	x = self.activation_fn(x)
		for i in range(len(self.layers)):
			#print("i=",i)
			#print("xav=",x)
			x = self.layers[i](x)
			#print("xap=",x)
			if self.drop_rate is not None:
				x = tf.nn.dropout(x, self.drop_rate)
		return x
