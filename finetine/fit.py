import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Параметры обучения
img_size = (450, 450)
batch_size = 32
epochs = 10

# Загрузка данных с использованием TFDS
dataset, info = tfds.load('open_images_v4', split='train', with_info=True, as_supervised=True)
val_dataset = tfds.load('open_images_v4', split='validation', as_supervised=True)

# Предобработка данных
def preprocess(image, label):
    image = tf.image.resize(image, img_size)
    image = image / 255.0  # Нормализация
    return image, label

train_dataset = dataset.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Загрузка предобученной модели MobileNet V2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))

# Добавление новых слоев
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(info.features['labels'].num_classes, activation='softmax')(x)

# Создание новой модели
model = Model(inputs=base_model.input, outputs=predictions)

# Заморозка слоев базовой модели
for layer in base_model.layers:
    layer.trainable = False

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# Разморозка слоев базовой модели для дообучения
for layer in base_model.layers:
    layer.trainable = True

# Компиляция модели с более низким learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Дообучение модели
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# Сохранение модели
model.save('mobilenet_v2_300x300.h5')