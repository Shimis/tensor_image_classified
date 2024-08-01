import tensorflow as tf

# Загрузите предобученную модель MobileNet V2 из TensorFlow Keras Applications
mobilenet_v2_model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(350, 350, 3))

# Преобразуйте модель в формат TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(mobilenet_v2_model)
tflite_model = converter.convert()

# Сохраните модель в файл
with open('mobilenet_v2_350.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted and saved as mobilenet_v2.tflite")
