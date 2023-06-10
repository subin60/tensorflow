

import tensorflow as tf
import numpy as np



xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
ys = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=float)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

model.compile(loss='mse',
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

model.fit(xs,ys, epochs=500)



new_x = 7.0
prediction = model.predict([new_x])[0]
print(prediction)