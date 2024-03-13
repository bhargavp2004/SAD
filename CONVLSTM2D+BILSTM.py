import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import os
import random
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import seaborn as sns

# Set seed for reproducibility
seed_constant = 7

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
DATASET_DIR = r'D:\B.TECH\6TH SEMESTER\Machine Learning\Project\Suspicious Activity Detection\Suspicious_Activity_Detection_ProjectML\Data_Extracted\Datasets\Datasets\Peliculas2'
CLASSES_LIST = ["Fight", "Normal", "Shooting"]

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
    features = []
    labels = []
    video_files_paths = []

    for class_index, class_name in enumerate(CLASSES_LIST):
        print(f'Extracting Data of Class: {class_name}')
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))

        for file_name in files_list:
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
            frames = frames_extraction(video_file_path)

            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels

features, labels = create_dataset()
print("Features shape:", features.shape)

one_hot_encoded_labels = to_categorical(labels)
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size=0.20, random_state=10)

def create_convlstm_model():
    model = Sequential()

    model.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True, input_shape=(SEQUENCE_LENGTH,
                                                                                      IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    
    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    
    model.add(ConvLSTM2D(filters=14, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    
    model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    
    model.add(TimeDistributed(Flatten()))
    
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(len(CLASSES_LIST), activation="softmax"))
     
    model.summary()
    return model

model = create_convlstm_model()

early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_training_history = model.fit(x=features_train, y=labels_train, epochs=50, batch_size=4, shuffle=True, validation_split=0.20, callbacks=[early_stopping_callback])

# Plot training and validation accuracy
plt.plot(model_training_history.history['accuracy'], label='Training Accuracy')
plt.plot(model_training_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy for CONVLSTM2D_BILSTM')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(model_training_history.history['loss'], label='Training Loss')
plt.plot(model_training_history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss for CONVLSTM2D_BILSTM')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(features_test, labels_test)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy (CONVLSTM2D_BILSTM): {test_accuracy:.4f}')

# Calculate precision, recall, F1 score, and confusion matrix
predictions = model.predict(features_test)
predictions_classes = np.argmax(predictions, axis=1)
labels_test_classes = np.argmax(labels_test, axis=1)

precision = precision_score(labels_test_classes, predictions_classes, average='weighted')
recall = recall_score(labels_test_classes, predictions_classes, average='weighted')
f1_score_value = f1_score(labels_test_classes, predictions_classes, average='weighted')
conf_matrix = confusion_matrix(labels_test_classes, predictions_classes)

print(f'Weighted Precision: {precision:.4f}')
print(f'Weighted Recall: {recall:.4f}')
print(f'Weighted F1 Score: {f1_score_value:.4f}')
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.title('Confusion Matrix For CONVLSTM2D_BILSTM')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

model.save("Suspicious_Human_Activity_Detection_CONVLSTM2D_BILSTM.keras")
