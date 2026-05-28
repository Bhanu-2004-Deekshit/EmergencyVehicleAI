import os
import urllib.request

# Ensure the models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# The official URL for MobileNetV2 Weights (No Top)
url = "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
output_path = "models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"

print(f"Downloading weights to {output_path}...")

try:
    urllib.request.urlretrieve(url, output_path)
    print("Download complete!")
except Exception as e:
    print(f"Error: {e}")
    print("If this failed, try downloading manually from: https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")