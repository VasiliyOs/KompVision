import time
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Categorical Crossentropy)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r*-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Загрузка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Размерность обучающих изображений: {x_train.shape}")
print(f"Размерность обучающих меток: {y_train.shape}")
print(f"Количество тестовых примеров: {x_test.shape[0]}")

# Нормализация данных
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Преобразование меток в One-Hot Encoding
num_classes = 10
y_train_ohe = to_categorical(y_train, num_classes=num_classes)
y_test_ohe = to_categorical(y_test, num_classes=num_classes)

# Построение модели MLP
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu', name='Hidden_1'),
    Dense(128, activation='relu', name='Hidden_2'),
    Dense(num_classes, activation='softmax', name='Output_Layer')
])
model.summary()

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
start = time.time()
history_extended = model.fit(x_train, y_train_ohe,
                            epochs=20,
                            batch_size=32,
                            validation_data=(x_test, y_test_ohe),
                            verbose=1)
finish = time.time() - start
print(f"Время обучения: {finish:.2f} секунд")

# Визуализация истории обучения
plot_history(history_extended)