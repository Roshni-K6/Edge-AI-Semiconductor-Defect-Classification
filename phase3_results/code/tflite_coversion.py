import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Convert ONNX → SavedModel (run this manually before script)
# onnx2tf -i mobilenet_defect.onnx

IMG_SIZE = 128

def representative_dataset():
    dataset_path = "phase3_dataset/train"

    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path)[:30]:
            img_path = os.path.join(class_path, img_name)

            img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb")
            img = image.img_to_array(img)
            img = img / 255.0
            img = np.expand_dims(img, axis=0).astype(np.float32)

            yield [img]

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

with open("mobilenet_defect_int8.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("INT8 TFLite model created successfully!")
print("Model Size (MB):", os.path.getsize("mobilenet_defect_int8.tflite")/(1024*1024))
