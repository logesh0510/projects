from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


imagePaths = list(paths.list_images('/content/drive/MyDrive/Drowsiness detection/Dataset'))
data = []
labels = []
# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]
    # load the input image (224x224) and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)
# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_shape=(224, 224, 3))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(124, activation="relu")(headModel)
headModel = Dense(224, activation="relu")(headModel)
headModel = Dropout(0.25)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)
# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
INIT_LR = 1e-4
EPOCHS = 5
BS = 8
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
model.save('/content/drive/MyDrive/Drowsiness detection/drowsi_detect 1.h5')

# test the model
import cv2
import os
import tensorflow
from google.colab.patches import cv2_imshow
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import jupyter_beeper

 
cascPath =  "/content/drive/MyDrive/Drowsiness detection/haarcascade_eye.xml"
eyeCascade = cv2.CascadeClassifier(cascPath)
model = tensorflow.keras.models.load_model("/content/drive/MyDrive/Drowsiness detection/drowsi_detect 1.h5")
 
video_capture = cv2.VideoCapture("/content/drive/MyDrive/Drowsiness detection/drowsi.mp4")
counter=0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    eyes = eyeCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(30, 30),
                                        )
    eyes_list=[]
    preds=[]
    for (x, y, w, h) in eyes:
        eye_frame = frame[y:y+h,x:x+w]
        eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2RGB)
        eye_frame = cv2.resize(eye_frame, (224, 224))
        eye_frame = img_to_array(eye_frame)
        eye_frame = np.expand_dims(eye_frame, axis=0)
        eye_frame =  preprocess_input(eye_frame)
        eyes_list.append(eye_frame)
        if len(eyes_list)>0:
          try:
            preds = model.predict(eyes_list)  
          except ValueError:
            print('expected 1 input but received two input tensors')
        for pred in preds:
            (close, open) = pred
        if open > close:
             label = "openedeye"       
        else:
             counter+=1
             label= "closedeye"
             if counter>5:
               b=jupyter_beeper.Beeper()
               b.beep()
               counter=0
        color = (0, 255, 0) if label == "openedeye" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(close,open) * 100)
        cv2.putText(frame, label, (x, y- 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 5)
 
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0,0,255), 2)
        # Display the resulting frame
        cv2_imshow(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
