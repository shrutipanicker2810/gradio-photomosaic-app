import kagglehub
from app import quantize_image
import os

# I have used the below dataset to generate tiles
dataset_path = kagglehub.dataset_download("flash10042/abstract-paintings-dataset")

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
image_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if os.path.splitext(file.lower())[1] in image_extensions:
            image_files.append(os.path.join(root, file))

print(f"Found {len(image_files)} images in the dataset.")

# quantization parameters
levels = 8   
output_dir = os.path.join(os.getcwd(), "quantized_images")

# process each image
for img_path in image_files:
    try:
        # Quantize the image
        quantized_img = quantize_image(img_path, levels)        
        # output filename ("image.jpg" -> "image_quantized.jpg")
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        original_ext = os.path.splitext(img_path)[1]
        output_filename = f"{base_name}_quantized.jpg" 
        output_path = os.path.join(output_dir, output_filename)
        quantized_img.save(output_path, 'JPEG', quality=95)  # high quality to preserve details
        
        print(f"Processed and saved: {output_filename}")
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

print("Batch quantization complete!")