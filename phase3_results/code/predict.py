import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# ==============================
# CONFIGURATION
# ==============================
IMG_SIZE = 128
MODEL_PATH = "mobilenet_defect_int8.tflite"
PREDICTION_FOLDER = "Hackathon_phase3_prediction_dataset"
LOG_FILE = "prediction_log.txt"

class_names = [
    'BRIDGE', 'CLEAN_CRACK', 'CLEAN_LAYER', 'CLEAN_VIA',
    'CMP', 'CRACK', 'LER', 'OPEN',
    'OTHERS', 'PARTICLE', 'VIA'
]

# ==============================
# LOAD MODEL
# ==============================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

print("Model Loaded Successfully")

# ==============================
# OPEN LOG FILE
# ==============================
log_file = open(LOG_FILE, "w")
log_file.write("Image_Name --> Prediction\n")
log_file.write("--------------------------------\n")

# ==============================
# PREDICTION LOOP
# ==============================
image_count = 0

for img_name in sorted(os.listdir(PREDICTION_FOLDER)):

    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(PREDICTION_FOLDER, img_name)

    # Load image (same preprocessing as training)
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb")
    img = image.img_to_array(img)

    # Normalize (same as ToTensor)
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Quantize for INT8 model
    img_quant = img / input_scale + input_zero_point
    img_quant = np.clip(img_quant, -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], img_quant)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize output
    output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

    pred_index = np.argmax(output_data)
    predicted_class = class_names[pred_index]

    result_line = f"{img_name} --> {predicted_class}"
    
    print(result_line)
    log_file.write(result_line + "\n")

    image_count += 1

log_file.close()

print("\n===================================")
print(f"Total Images Processed: {image_count}")
print(f"Results saved in: {LOG_FILE}")
print("Prediction completed successfully!")
