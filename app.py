from PIL import Image
import numpy as np
import os
from scipy.spatial.distance import cdist
import tempfile
import time

def quantize_image(image_path, levels):
    '''
    Loads an image and applies uniform color quantization to reduce color variations.
    Args: image_path (str), levels (int)
    Returns: PIL.Image
    '''
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Could not load image: {e}")
    
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_arr = np.array(image)
    step = 255 // levels
    quantized_array = (img_arr // step) * step
    quantized_array = quantized_array + step // 2
    quantized_array = np.clip(quantized_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(quantized_array)

def create_image_grid(image, rows=32, cols=32):
    '''
    Divide an image into a grid of smaller rectagular tiles for mosaic generation.
    Args: image (PIL.Image), rows (int), cols (int)
    Returns: tuple of cropped_image (PIL.Image), grid_array (numpy.ndarray)
    '''
    if image.mode != 'RGB':
        image = image.convert('RGB')

    width, height = image.size
    tile_width = width // cols
    tile_height = height // rows

    crop_width = cols * tile_width
    crop_height = rows * tile_height
    cropped_image = image.crop((0, 0, crop_width, crop_height))

    img_arr = np.array(cropped_image)
    grid_array = img_arr.reshape(
        rows, tile_height, cols, tile_width, 3
    ).transpose(0, 2, 1, 3, 4)

    return cropped_image, grid_array

def calculate_mse(original, mosaic):
    '''
    Calculate Mean Squared Error between original and mosaic images
    Args: original (PIL.Image), mosaic (PIL.Image)
    Returns: mse (float)
    '''
    if original.size != mosaic.size:
        mosaic = mosaic.resize(original.size, Image.LANCZOS)
    
    orig_arr = np.array(original).astype(np.float32)
    mosaic_arr = np.array(mosaic).astype(np.float32)
    mse = np.mean((orig_arr - mosaic_arr) ** 2)
    return mse

def calculate_simple_ssim(original, mosaic):
    '''
    Calculate simplified SSIM using basic statistical measures
    Args: original (PIL.Image), mosaic (PIL.Image)
    Returns: ssim (float)
    '''
    if original.size != mosaic.size:
        mosaic = mosaic.resize(original.size, Image.LANCZOS)
    
    orig_arr = np.array(original.convert('L')).astype(np.float32)
    mosaic_arr = np.array(mosaic.convert('L')).astype(np.float32)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = np.mean(orig_arr)
    mu2 = np.mean(mosaic_arr)
    var1 = np.var(orig_arr)
    var2 = np.var(mosaic_arr)
    cov = np.mean((orig_arr - mu1) * (mosaic_arr - mu2))
    
    numerator = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (var1 + var2 + C2)
    ssim = numerator / denominator
    
    return ssim

def create_mosaic(input_path, tile_dir, levels=16, rows=32, cols=32, output_path="mosaic.jpg"):
    '''
    Creates the mosaic by replacing grid cells with best matching tile using vectorized operations.
    - Input image is quantized to reduce color complexity
    - Input image is then divided into grid of specified dimensions
    - Average color for eacg grid cell is computed
    - Each grid cell is matched to the closest time based on color similarity
    - Final mosaic is constructed using vectorized operations

    Args: input_path (str), tile_dir (str), levels (int), rows (int), cols (int), output_path (str)
    Returns: tuple of mosaic (PIL.Image), mse (float), ssim (float), total_time (float)
    '''
    start_time = time.time()
    
    # Quantize input image
    quantized_img = quantize_image(input_path, levels)
    
    # Create grid and compute averages
    _, grid_array = create_image_grid(quantized_img, rows, cols)
    grid_averages = np.mean(grid_array, axis=(2, 3))
    tile_height, tile_width = grid_array.shape[2:4]
    
    # Load tile library
    tile_paths = [os.path.join(tile_dir, f) for f in os.listdir(tile_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    if not tile_paths:
        raise ValueError(f"No tile images found in {tile_dir}")
    
    tile_arrays = []
    tile_avgs = []
    
    for p in tile_paths:
        try:
            img = Image.open(p).resize((tile_width, tile_height))
            arr = np.array(img.convert('RGB'))
            avg = np.mean(arr, axis=(0, 1))
            tile_arrays.append(arr)
            tile_avgs.append(avg)
        except:
            continue
    
    if not tile_avgs:
        raise ValueError("No valid tiles loaded")
    
    tile_avgs = np.array(tile_avgs)
    tile_arrays = np.array(tile_arrays)
    
    # Vectorized matching
    flat_grid_avgs = grid_averages.reshape(-1, 3)
    distances = cdist(flat_grid_avgs, tile_avgs, metric='euclidean')
    best_indices = np.argmin(distances, axis=1).astype(np.int32)
    
    # Construct mosaic
    crop_width = cols * tile_width
    crop_height = rows * tile_height
    mosaic_array = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
    grid_indices = best_indices.reshape(rows, cols)
    
    for i in range(rows):
        for j in range(cols):
            start_h = i * tile_height
            end_h = (i + 1) * tile_height
            start_w = j * tile_width
            end_w = (j + 1) * tile_width
            mosaic_array[start_h:end_h, start_w:end_w] = tile_arrays[grid_indices[i, j]]
    
    # Save result
    mosaic = Image.fromarray(mosaic_array)
    mosaic.save(output_path, 'JPEG', quality=95)
    
    total_time = time.time() - start_time
    
    # Calculate quality metrics
    original = Image.open(input_path)
    if original.mode != 'RGB':
        original = original.convert('RGB')
    
    mse = calculate_mse(original, mosaic)
    ssim = calculate_simple_ssim(original, mosaic)
    
    return mosaic, mse, ssim, total_time

def create_mosaic_loop_based(input_path, tile_dir, levels=16, rows=32, cols=32):
    '''
    Creates the mosaic by replacing grid cells with best matching tile using loop-based operations.
    We use this function to compare with the vectorized operation based function above
    and highlight benefits.

    Args: input_path (str), tile_dir (str), levels (int), rows (int), cols (int)
    Returns: total_time (float)
    '''
    start_time = time.time()
    
    quantized_img = quantize_image(input_path, levels)
    _, grid_array = create_image_grid(quantized_img, rows, cols)
    tile_height, tile_width = grid_array.shape[2:4]
    
    # Load limited tiles for fair comparison
    tile_paths = [os.path.join(tile_dir, f) for f in os.listdir(tile_dir) if f.lower().endswith(('.jpg', '.jpeg'))][:100]
    
    tile_arrays = []
    tile_avgs = []
    
    for p in tile_paths:
        try:
            img = Image.open(p).resize((tile_width, tile_height))
            arr = np.array(img.convert('RGB'))
            avg = np.mean(arr, axis=(0, 1))
            tile_arrays.append(arr)
            tile_avgs.append(avg)
        except:
            continue
    
    tile_arrays = np.array(tile_arrays)
    tile_avgs = np.array(tile_avgs)
    
    # LOOP-BASED MATCHING
    crop_width = cols * tile_width
    crop_height = rows * tile_height
    mosaic_array = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            cell_avg = np.mean(grid_array[i, j], axis=(0, 1))
            
            # Find best tile using loops
            best_distance = float('inf')
            best_tile_idx = 0
            
            for tile_idx, tile_avg in enumerate(tile_avgs):
                distance = np.sum((cell_avg - tile_avg) ** 2)
                if distance < best_distance:
                    best_distance = distance
                    best_tile_idx = tile_idx
            
            # Place tile
            start_h = i * tile_height
            end_h = (i + 1) * tile_height
            start_w = j * tile_width
            end_w = (j + 1) * tile_width
            mosaic_array[start_h:end_h, start_w:end_w] = tile_arrays[best_tile_idx]
    
    total_time = time.time() - start_time
    return total_time

def process_image(input_image, rows, cols, levels):
    '''
    Main processing function that generates photomosaics for all grid sizes and performs comprehensive analysis.
    Refer Gradio interface for final mosaic generated and terminal for performance analysis done.
    '''
    print("=" * 60, flush=True)
    print("PHOTOMOSAIC GENERATION STARTED", flush=True)
    print("=" * 60, flush=True)
    
    if input_image is None:
        print("ERROR: No image provided!", flush=True)
        return None, None, None, None
    
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            input_image.save(temp_path, 'JPEG', quality=95)
        
        tile_dir = os.path.join(os.getcwd(), "quantized_images")
        
        if not os.path.exists(tile_dir):
            raise ValueError(f"Tile directory not found: {tile_dir}")
        
        tile_count = len([f for f in os.listdir(tile_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
        print(f"Input: {input_image.size}, Tiles: {tile_count}, Quantization: {int(levels)} levels", flush=True)
        
        # Generate quantized image for display
        quantized_image = quantize_image(temp_path, int(levels))
        
        # Generate all three grid sizes
        grid_sizes = [(16, 16), (32, 32), (64, 64)]
        results = []
        mosaics = {}
        
        for rows_grid, cols_grid in grid_sizes:
            mosaic, mse, ssim, proc_time = create_mosaic(
                input_path=temp_path,
                tile_dir=tile_dir,
                levels=int(levels),
                rows=rows_grid,
                cols=cols_grid,
                output_path=f"mosaic_{rows_grid}x{cols_grid}.jpg"
            )
            
            results.append({
                'grid_size': f"{rows_grid}x{cols_grid}",
                'total_tiles': rows_grid * cols_grid,
                'processing_time': proc_time,
                'mse': mse,
                'ssim': ssim
            })
            
            mosaics[f"{rows_grid}x{cols_grid}"] = mosaic
            print(f"  {rows_grid}x{cols_grid}: {proc_time:.3f}s, MSE: {mse:.1f}, SSIM: {ssim:.3f}")
        
        # STEP 5: PERFORMANCE METRICS
        print(f"\n{'='*60}", flush=True)
        print("STEP 5: PERFORMANCE METRICS", flush=True)
        print("="*60, flush=True)
        print(f"{'Grid Size':<12} {'MSE':<10} {'SSIM':<8}")
        print("-"*32, flush=True)
        for result in results:
            print(f"{result['grid_size']:<12} {result['mse']:<10.1f} {result['ssim']:<8.3f}")
        
        # STEP 6: PERFORMANCE ANALYSIS
        print(f"\n{'='*60}", flush=True)
        print("STEP 6: PERFORMANCE ANALYSIS", flush=True)
        print("="*60, flush=True)
        
        # 6a. Processing time for different grid sizes
        print("6a. Processing Time for Different Grid Sizes:")
        print(f"{'Grid Size':<12} {'Time(s)':<10} {'Tiles/sec':<12}")
        print("-"*36, flush=True)
        for result in results:
            tiles_per_sec = result['total_tiles'] / result['processing_time']
            print(f"{result['grid_size']:<12} {result['processing_time']:<10.3f} {tiles_per_sec:<12.1f}")
        
        # 6b. Performance scaling analysis
        print("\n6b. Performance Scaling Analysis:")
        for i in range(1, len(results)):
            prev = results[i-1]
            curr = results[i]
            
            tile_ratio = curr['total_tiles'] / prev['total_tiles']
            time_ratio = curr['processing_time'] / prev['processing_time']
            efficiency = tile_ratio / time_ratio
            
            scaling_type = "Super-linear" if efficiency > 2.0 else "Linear" if efficiency > 0.8 else "Sub-linear"
            print(f"  {prev['grid_size']} → {curr['grid_size']}: {tile_ratio:.1f}x tiles, {time_ratio:.2f}x time ({scaling_type})")
        
        # 6c. Vectorized vs Loop-based comparison (FIXED)
        print("\n6c. Vectorized vs Loop-based Implementation:")
        comparison_sizes = [(16, 16), (32, 32)]
        
        for rows_test, cols_test in comparison_sizes:
            # Setup test data for fair comparison
            quantized_test = quantize_image(temp_path, int(levels))
            _, grid_array_test = create_image_grid(quantized_test, rows_test, cols_test)
            grid_averages_test = np.mean(grid_array_test, axis=(2, 3))
            flat_grid_avgs = grid_averages_test.reshape(-1, 3)
            
            # Load limited tiles for fair comparison (same for both methods)
            tile_paths_limited = [os.path.join(tile_dir, f) for f in os.listdir(tile_dir) if f.lower().endswith(('.jpg', '.jpeg'))][:100]
            tile_avgs_limited = []
            for p in tile_paths_limited:
                try:
                    img = Image.open(p).resize((20, 20))
                    arr = np.array(img.convert('RGB'))
                    tile_avgs_limited.append(np.mean(arr, axis=(0, 1)))
                except:
                    continue
            tile_avgs_limited = np.array(tile_avgs_limited[:50])
            
            # Test VECTORIZED approach (using scipy cdist)
            vec_start = time.time()
            distances = cdist(flat_grid_avgs, tile_avgs_limited, metric='euclidean')
            best_indices_vec = np.argmin(distances, axis=1)
            vec_time = time.time() - vec_start
            
            # Test LOOP-BASED approach (manual distance calculation)
            loop_start = time.time()
            best_indices_loop = []
            for grid_avg in flat_grid_avgs:
                best_dist = float('inf')
                best_idx = 0
                for i, tile_avg in enumerate(tile_avgs_limited):
                    dist = np.sqrt(np.sum((grid_avg - tile_avg) ** 2))
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                best_indices_loop.append(best_idx)
            loop_time = time.time() - loop_start
            
            speedup = loop_time / vec_time if vec_time > 0 else 0
            print(f"  {rows_test}x{cols_test}: Vectorized {vec_time:.4f}s, Loop {loop_time:.4f}s, Speedup: {speedup:.1f}x")
        
        print("="*60, flush=True)
        
        # Clean up
        os.unlink(temp_path)
        
        # Return all images for display
        return (quantized_image, 
                mosaics.get("16x16"), 
                mosaics.get("32x32"), 
                mosaics.get("64x64"))
        
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        # Clean up on error
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        return None, None, None, None

# GRADIO INTERFACE
import gradio as gr

print("Creating streamlined mosaic generator interface...")

with gr.Blocks(title="Photomosaic Generator") as app:
    gr.Markdown("# Photomosaic Generator")
    gr.Markdown("Upload an image to generate photomosaics with performance analysis.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Original Image")
            
            with gr.Row():
                rows_input = gr.Slider(8, 64, value=32, step=8, label="Grid Rows")
                cols_input = gr.Slider(8, 64, value=32, step=8, label="Grid Columns")
            
            levels_input = gr.Slider(4, 16, value=16, step=4, label="Quantization Levels")
            
            generate_btn = gr.Button("Generate Mosaics", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Processing Pipeline")
            
            # Show quantized (segmented) image
            quantized_display = gr.Image(type="pil", label="Quantized Image (Segmented)")
            
            gr.Markdown("### Generated Mosaics")
            
            with gr.Row():
                mosaic_16x16 = gr.Image(type="pil", label="16×16 Grid Mosaic")
                mosaic_32x32 = gr.Image(type="pil", label="32×32 Grid Mosaic")
            
            mosaic_64x64 = gr.Image(type="pil", label="64×64 Grid Mosaic")
    
    gr.Markdown("- **Complete Pipeline**: Original → Quantized → Mosaic results")
    
    def on_generate_click(img, r, c, l):
        if img is None:
            return None, None, None, None
        return process_image(img, r, c, l)
    
    generate_btn.click(
        fn=on_generate_click,
        inputs=[image_input, rows_input, cols_input, levels_input],
        outputs=[quantized_display, mosaic_16x16, mosaic_32x32, mosaic_64x64]
    )

print("Launching streamlined app...")
app.launch(share=True, debug=True)