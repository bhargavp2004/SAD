from keras.layers import *
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import tensorflow as tf
import seaborn as sns

seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

# /media/b/DATA/B.TECH/6TH SEMESTER/Machine Learning/Project/Suspicious Activity Detection/Suspicious_Activity_Detection_ProjectML/Data_Extracted/Datasets/Datasets/Peliculas/Explosion
SEQUENCE_LENGTH = 20
total_frames = 0
DATASET_DIR = r'D:\B.TECH\6TH SEMESTER\Machine Learning\Project\Suspicious Activity Detection\Suspicious_Activity_Detection_ProjectML\Data_Extracted\Datasets\Datasets\Peliculas2'

CLASSES_LIST = ["Fight", "Normal", "Shooting"]
 
class PrintValuesCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("Printing values at the end of epoch:", epoch)
        for i in range(len(features_test)):
            x = features_test[i:i+1]
            y_true = labels_test[i:i+1]
            y_pred = self.model.predict(x)
            print("Sequence:", i)
            print("Input vector size:", x.shape)
            print("Sequence length:", SEQUENCE_LENGTH)
            print("Image size (height x width):", IMAGE_HEIGHT, "x", IMAGE_WIDTH)
            print("True label:", y_true)
            print("Predicted label:", y_pred)
            print("Input vector dimensions:", x.shape[1:], "\n")

def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):

        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()

        if not success:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        normalized_frame = resized_frame / 255

        frames_list.append(normalized_frame)

    video_reader.release()
    return frames_list

def create_dataset():
    global total_frames
    features = []
    labels = []
    video_files_paths = []

    for class_index, class_name in enumerate(CLASSES_LIST):

        print(f'Extracting Data of Class: {class_name}')

        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))

        class_frames = 0
        for file_name in files_list:
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
            frames = frames_extraction(video_file_path)
            total_frames = total_frames + len(frames)
            class_frames += len(frames)
            if len(frames) == SEQUENCE_LENGTH:

                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)
        print(f'Total frames for {class_name}: {class_frames}')
        total_frames += class_frames
        print("Total frames : ", total_frames)

    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels, video_files_paths


features, labels, video_files_paths = create_dataset()
print("Total frames : ", )
print("Feature's shape : ", features.shape)
one_hot_encoded_labels = to_categorical(labels)
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size = 0.25, shuffle=True, random_state = 10)
features = None
labels = None

def create_model():
    model = Sequential()
    
    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation = 'relu'),
                              input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    print("After Conv2D layer 1:", model.output_shape)

    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    print("After MaxPooling2D layer 1:", model.output_shape)
    
    model.add(TimeDistributed(Dropout(0.20)))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu')))
    print("After Conv2D layer 2:", model.output_shape)

    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    print("After MaxPooling2D layer 2:", model.output_shape)
    
    model.add(TimeDistributed(Dropout(0.20)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    print("After Conv2D layer 3:", model.output_shape)

    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    print("After MaxPooling2D layer 3:", model.output_shape)
    
    model.add(TimeDistributed(Dropout(0.20)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    print("After Conv2D layer 4:", model.output_shape)

    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    print("After MaxPooling2D layer 4:", model.output_shape)

    model.add(TimeDistributed(Flatten()))
    print("After Flatten layer:", model.output_shape)

    model.add(LSTM(32))
    print("After LSTM layer:", model.output_shape)

    model.add(Dense(len(CLASSES_LIST), activation = 'softmax'))
    print("After Dense layer:", model.output_shape)

    model.summary()

    return model



model = create_model()

early_stopping_callback = EarlyStopping(monitor = 'accuracy', patience = 10, mode = 'max', restore_best_weights = True)
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])
model_training_history = model.fit(features_train, labels_train, epochs=70, batch_size=4, shuffle=True, validation_split=0.25, callbacks=[early_stopping_callback])

plt.plot(model_training_history.history['accuracy'], label='Training Accuracy')
plt.plot(model_training_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy for LRCN ')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(model_training_history.history['loss'], label='Training Loss')
plt.plot(model_training_history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss for LRCN ')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_loss, test_accuracy = model.evaluate(features_test, labels_test)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy (LRCN): {test_accuracy:.4f}')
predictions = model.predict(features_test)
predictions_classes = np.argmax(predictions, axis=1)
labels_test_classes = np.argmax(labels_test, axis=1)

precision = precision_score(labels_test_classes, predictions_classes, average='weighted')
recall = recall_score(labels_test_classes, predictions_classes, average='weighted')

print(f'Weighted Precision: {precision:.4f}')
print(f'Weighted Recall: {recall:.4f}')
conf_matrix = confusion_matrix(labels_test_classes, predictions_classes)
f1_score_value = f1_score(labels_test_classes, predictions_classes, average='weighted')
print("\nF1 Score (weighted):", f1_score_value)
print("Confusion Matrix:")
print(conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.title('Confusion Matrix For LRCN')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
model.save("Suspicious_Human_Activity_Detection_LRCN.keras")
