import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Dropout, BatchNormalization
import keras.applications.mobilenet_v2 as mobilenetv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocessing
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix

base_path = './garbage_classification_subset/'

categories = os.listdir(base_path)
print('El conjunto de datos tiene ' + str(len(categories)) + ' categorías: ' + str(categories))


def add_class_name_prefix(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: x[:re.search("\d",x).start()] + '/' + x)
    return df

filenames_list = []
categories_list = []
for i in range(len(categories)):
    filenames = os.listdir(base_path + categories[i])
    filenames_list = filenames_list  +filenames
    categories_list = categories_list + [i] * len(filenames)

df = pd.DataFrame({
    'filename': filenames_list,
    'category': categories_list
})

df = add_class_name_prefix(df, 'filename')
df = df.sample(frac=1).reset_index(drop=True)

print('number of elements = ' , len(df))

df.head()

df_visualization = df.copy()

df_visualization['category'] = df_visualization['category'].apply(lambda x:categories[x] )

df_visualization['category'].value_counts().plot.bar(x = 'count', y = 'category' )

plt.xlabel("Garbage Classes", labelpad=14)
plt.ylabel("Images Count", labelpad=14)
plt.title("Count of images per class", y=1.02)


IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

mobilenetv2_layer = mobilenetv2.MobileNetV2(include_top = False, 
                       input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS),
                       weights = 'imagenet')

# Unfreeze the last few layers for fine-tuning
mobilenetv2_layer.trainable = True
for layer in mobilenetv2_layer.layers[:-30]:
    layer.trainable = False

# Rebuild the model with dropout and batch norm
model = Sequential()
model.add(keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

# Apply preprocessing and add base model
model.add(Lambda(mobilenetv2_preprocessing))
model.add(mobilenetv2_layer)

# Add custom layers
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(len(categories), activation='softmax'))

# Compile with a low learning rate for fine-tuning
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-5),
    metrics=['categorical_accuracy']
)

model.summary()

early_stop = EarlyStopping(
    patience=8,
    verbose=1,
    monitor='val_categorical_accuracy',
    mode='max',
    min_delta=0.0005,
    restore_best_weights=True
)
callbacks = [early_stop]

print('call back defined!')

#Change the categories from numbers to names
df["category"] = df["category"].apply(lambda x:categories[x] ) # Convert to category names


# We first split the data into two sets and then split the validate_df to two sets
train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
validate_df, test_df = train_test_split(validate_df, test_size=0.5, random_state=42)

train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

print('train size = ', total_train , 'validate size = ',
      total_validate, 'test size = ', test_df.shape[0])

batch_size=32

train_datagen = image.ImageDataGenerator(
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    base_path,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
    )

validation_datagen = image.ImageDataGenerator()

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    base_path,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# Encode the labels as integers
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['category'])

# Calculate class weights using the encoded values
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['label_encoded']),
    y=df['label_encoded']
)

# Create the dictionary with integer keys (0, 1, 2, ...)
class_weights_dict = dict(zip(np.unique(df['label_encoded']), class_weights))

EPOCHS = 20
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weights_dict
)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_yticks(np.arange(0, 0.7, 0.1))
ax1.legend()

ax2.plot(history.history['categorical_accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_categorical_accuracy'], color='r',label="Validation accuracy")
ax2.legend()

legend = plt.legend(loc='best')
plt.tight_layout()
plt.show()

test_datagen = image.ImageDataGenerator()

test_generator = test_datagen.flow_from_dataframe(
    dataframe= test_df,
    directory=base_path,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=1,
    shuffle=False
)


loss, accuracy = model.evaluate(test_generator)

print('Accuracy on test set =', round(accuracy * 100, 2), '%')



gen_label_map = test_generator.class_indices
gen_label_map = dict((v,k) for k,v in gen_label_map.items())
print(gen_label_map)

# Get the model's predictions for the test set
preds = model.predict(test_generator)

# Get the category with the highest predicted probability
preds = preds.argmax(1)

# Convert the predicted category's number to name
preds = [gen_label_map[item] for item in preds]

# Convert the pandas dataframe to a numpy matrix
labels = test_df['category'].to_numpy()

print(classification_report(labels, preds))


cm = confusion_matrix(labels, preds, labels=categories)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=categories, yticklabels=categories, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

model.save("models/garbage_classifier.keras")
print("Model saved successfully.")