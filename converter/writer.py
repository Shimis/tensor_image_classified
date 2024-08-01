from tflite_support.metadata_writers import writer_utils
from tflite_support.metadata_writers import image_classifier
from tflite_support.metadata_writers import metadata_info

# Specify the path to your TFLite model
model_path = "mobilenet_v2.tflite"

# Load your model
with open(model_path, "rb") as f:
    model_content = f.read()

# Create ImageClassifier writer
writer = image_classifier.MetadataWriter.create_for_inference(
    model_content, 
    input_norm_mean=[127.5],   # Mean value for normalization
    input_norm_std=[127.5],    # Standard deviation for normalization
    input_tensor_type=metadata_info.TensorType.FLOAT32
)

# Populate metadata into the model
tflite_model_with_metadata = writer.populate()

# Save the TFLite model with metadata
output_model_path = "mobilenet_v2_with_metadata.tflite"
writer_utils.save_file(output_model_path, tflite_model_with_metadata)