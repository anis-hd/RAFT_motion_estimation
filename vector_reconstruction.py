import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
# import matplotlib.pyplot as plt # No longer needed for output video
import os
import cv2 # Use OpenCV for image/video reading/writing and drawing
import time
from tqdm import tqdm # For progress bar

# --- Configuration ---
# Target height for resizing frames before processing
TARGET_HEIGHT = 480
# TARGET_WIDTH will be calculated based on input video aspect ratio

NUM_ITERATIONS = 8 # Must match the trained model's iterations
CHECKPOINT_DIR = './raft_checkpoints' # Directory where weights are saved
FINAL_WEIGHTS_FILE = 'raft_final_weights.weights.h5' # Name of the saved weights file
INPUT_VIDEO_PATH = './input.mp4'
OUTPUT_VIDEO_PATH = './output_vectors.mp4' # Changed output filename

VECTOR_STEP = 14       # Draw a vector every 8 pixels (More points)
VECTOR_SCALE = 5.0    # Multiplier for vector length (Longer vectors)
VECTOR_COLOR = (0, 255, 0) # Red in BGR (Changed color)
VECTOR_THICKNESS = 1

# --- Dynamic Model Input Size (Will be set after reading video) ---
IMG_HEIGHT = -1 # Placeholder
IMG_WIDTH = -1  # Placeholder

# --- Mixed Precision Setup (Important if trained with it) ---
POLICY = 'mixed_float16' # Or 'float32' if not trained with mixed precision
print(f"Setting global policy to: {POLICY}")
tf.keras.mixed_precision.set_global_policy(POLICY)

# --- 1. Re-define Model Architecture (Identical to original script) ---
# Basic Residual Block
def BasicBlock(filters, stride=1):
    # ... (keep the definition as in the original script)
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(dtype=tf.float32), # Specify dtype for norm layers
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(dtype=tf.float32),
    ])

def DownsampleBlock(filters, stride=2):
     # ... (keep the definition as in the original script)
     return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(dtype=tf.float32),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(dtype=tf.float32),
        tf.keras.layers.ReLU(), # Add ReLU after second conv too
    ])

# Feature Encoder (f-net)
class FeatureEncoder(tf.keras.Model):
    # ... (keep the definition as in the original script)
    def __init__(self, name='feature_encoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.norm1 = tfa.layers.InstanceNormalization(dtype=tf.float32)
        self.relu1 = tf.keras.layers.ReLU()
        self.layer1 = DownsampleBlock(64, stride=1) # Output: H/2, W/2
        self.layer2 = DownsampleBlock(96, stride=2) # Output: H/4, W/4
        self.layer3 = DownsampleBlock(128, stride=2) # Output: H/8, W/8
        self.conv_out = tf.keras.layers.Conv2D(256, kernel_size=1) # Final projection
    def call(self, x):
        x = self.conv1(x); x = self.norm1(x); x = self.relu1(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.conv_out(x)
        return x

# Context Encoder (c-net)
class ContextEncoder(tf.keras.Model):
    # ... (keep the definition as in the original script)
    def __init__(self, name='context_encoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.norm1 = tfa.layers.InstanceNormalization(dtype=tf.float32)
        self.relu1 = tf.keras.layers.ReLU()
        self.layer1 = DownsampleBlock(64, stride=1)
        self.layer2 = DownsampleBlock(96, stride=2)
        self.layer3 = DownsampleBlock(128, stride=2)
        self.conv_out = tf.keras.layers.Conv2D(128, kernel_size=1)
    def call(self, x):
        x = self.conv1(x); x = self.norm1(x); x = self.relu1(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.conv_out(x)
        return x

# Convolutional GRU Cell
class ConvGRUCell(tf.keras.layers.Layer):
    # ... (keep the definition as in the original script)
    def __init__(self, hidden_filters, input_filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_filters = hidden_filters
        self.input_filters = input_filters
        self.kernel_size = kernel_size
        self.state_size = tf.TensorShape([None, None, hidden_filters])
        self.conv_update = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', activation='sigmoid', kernel_initializer='glorot_uniform')
        self.conv_reset = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', activation='sigmoid', kernel_initializer='glorot_uniform')
        self.conv_candidate = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', activation='tanh', kernel_initializer='glorot_uniform')
    def build(self, input_shape): pass
    def call(self, inputs, states):
        h_prev = states[0]
        combined_input_h = tf.concat([inputs, h_prev], axis=-1)
        update_gate = self.conv_update(combined_input_h)
        reset_gate = self.conv_reset(combined_input_h)
        combined_input_reset_h = tf.concat([inputs, reset_gate * h_prev], axis=-1)
        candidate_h = self.conv_candidate(combined_input_reset_h)
        new_h = (1. - update_gate) * h_prev + update_gate * candidate_h
        return new_h, [new_h]

# Motion Encoder and Update Block
class UpdateBlock(tf.keras.Model):
    # ... (keep the definition as in the original script)
    def __init__(self, iterations, hidden_dim=128, context_dim=128, corr_levels=1, corr_radius=4, name='update_block', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iterations = iterations
        self.hidden_dim = hidden_dim
        corr_feature_dim = (2 * corr_radius + 1)**2 * corr_levels
        motion_encoder_input_dim = corr_feature_dim + 2
        motion_encoder_output_dim = 32
        inp_dim = max(0, context_dim - hidden_dim)
        gru_input_total_dim = motion_encoder_output_dim + inp_dim
        self.motion_encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, 1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(motion_encoder_output_dim, 3, padding='same', activation='relu')
            ], name='motion_encoder')
        self.gru_cell = ConvGRUCell(hidden_filters=hidden_dim, input_filters=gru_input_total_dim)
        self.flow_head = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(2, 3, padding='same')
            ], name='flow_head')
    def call(self, net, inp, corr_features, flow_init=None):
        shape_tensor = tf.shape(net); b, h, w = shape_tensor[0], shape_tensor[1], shape_tensor[2]
        flow = tf.zeros([b, h, w, 2], dtype=tf.float32) if flow_init is None else tf.cast(flow_init, tf.float32)
        hidden_state = net
        flow_predictions = []
        for iter in range(self.iterations):
            flow = tf.stop_gradient(flow)
            motion_input = tf.concat([corr_features, flow], axis=-1)
            motion_features = self.motion_encoder(motion_input)
            gru_input = tf.concat([motion_features, inp], axis=-1)
            hidden_state, [hidden_state] = self.gru_cell(gru_input, [hidden_state])
            delta_flow = self.flow_head(hidden_state)
            delta_flow = tf.cast(delta_flow, tf.float32)
            flow = flow + delta_flow
            flow_predictions.append(flow)
        return flow_predictions

# --- Helper for Correlation ---
def build_correlation_volume(fmap1, fmap2, radius=4):
    # ... (keep the definition as in the original script)
    compute_dtype = fmap1.dtype
    fmap2 = tf.cast(fmap2, compute_dtype)
    batch_size, h, w, c = tf.shape(fmap1)[0], tf.shape(fmap1)[1], tf.shape(fmap1)[2], tf.shape(fmap1)[3]
    pad_size = radius
    fmap2_padded = tf.pad(fmap2, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='CONSTANT')
    gy, gx = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')
    coords_base = tf.stack([gy, gx], axis=-1); coords_base = tf.cast(coords_base, tf.int32)
    coords_base = tf.expand_dims(tf.expand_dims(coords_base, 0), -2)
    coords_base = tf.tile(coords_base, [batch_size, 1, 1, 1, 1])
    dy, dx = tf.meshgrid(tf.range(-radius, radius + 1), tf.range(-radius, radius + 1), indexing='ij')
    delta = tf.stack([dy, dx], axis=-1); num_neighbors = (2*radius+1)**2
    delta = tf.reshape(delta, [1, 1, 1, num_neighbors, 2]); delta = tf.cast(delta, tf.int32)
    lookup_coords = coords_base + delta + pad_size
    batch_indices = tf.range(batch_size); batch_indices = tf.reshape(batch_indices, [batch_size, 1, 1, 1])
    batch_indices = tf.tile(batch_indices, [1, h, w, num_neighbors])
    lookup_indices = tf.stack([batch_indices, lookup_coords[..., 0], lookup_coords[..., 1]], axis=-1)
    fmap2_neighbors = tf.gather_nd(fmap2_padded, lookup_indices)
    fmap1_expanded = tf.expand_dims(fmap1, axis=3)
    correlation = tf.reduce_sum(fmap1_expanded * fmap2_neighbors, axis=-1)
    correlation_float32 = tf.cast(correlation, tf.float32)
    correlation_normalized = correlation_float32 / tf.cast(c, tf.float32)
    return correlation_normalized

# --- RAFT Model ---
class RAFT(tf.keras.Model):
    # ... (keep the definition as in the previous script)
    def __init__(self, img_height, img_width, num_iterations=NUM_ITERATIONS, hidden_dim=128, context_dim=128, corr_levels=1, corr_radius=4, name='raft', **kwargs):
        super().__init__(name=name, **kwargs)
        print(f"Initializing RAFT with H={img_height}, W={img_width}")
        self.num_iterations = num_iterations; self.hidden_dim = hidden_dim; self.context_dim = context_dim
        self.corr_levels = corr_levels; self.corr_radius = corr_radius
        self.feature_encoder = FeatureEncoder()
        self.context_encoder = ContextEncoder()
        self.update_block = UpdateBlock(iterations=num_iterations, hidden_dim=hidden_dim, context_dim=context_dim, corr_levels=corr_levels, corr_radius=corr_radius)

    @tf.function
    def upsample_flow(self, flow, target_height, target_width):
        # ... (keep the definition as in the previous script)
        flow = tf.cast(flow, tf.float32)
        shape_tensor = tf.shape(flow); b, h_low, w_low = shape_tensor[0], shape_tensor[1], shape_tensor[2]
        scale_factor_h = tf.cast(target_height, tf.float32) / tf.cast(h_low, tf.float32)
        scale_factor_w = tf.cast(target_width, tf.float32) / tf.cast(w_low, tf.float32)
        flow_upsampled = tf.image.resize(flow, [target_height, target_width], method='bilinear')
        u = flow_upsampled[..., 0] * scale_factor_w; v = flow_upsampled[..., 1] * scale_factor_h
        flow_scaled = tf.stack([u, v], axis=-1)
        return flow_scaled

    def call(self, inputs, training=False):
        image1, image2 = inputs
        # Target height/width for upsampling is taken from input images directly
        target_height = tf.shape(image1)[1]; target_width = tf.shape(image1)[2]
        fmap1 = self.feature_encoder(image1); fmap2 = self.feature_encoder(image2)
        context_fmap = self.context_encoder(image1)
        split_sizes = [self.hidden_dim, max(0, self.context_dim - self.hidden_dim)]
        if sum(split_sizes) != self.context_dim: raise ValueError("Context split error")
        net, inp = tf.split(context_fmap, split_sizes, axis=-1)
        net = tf.tanh(net); inp = tf.nn.relu(inp)
        corr_features = build_correlation_volume(fmap1, fmap2, radius=self.corr_radius)
        flow_predictions_low_res = self.update_block(net, inp, corr_features, flow_init=None)
        flow_predictions_upsampled = [self.upsample_flow(flow_lr, target_height, target_width) for flow_lr in flow_predictions_low_res]
        return flow_predictions_upsampled

# --- 2. Preprocessing Function (Modified for OpenCV frames) ---
def preprocess_frame(frame_np, target_h, target_w):
    """
    Preprocesses a single frame (NumPy array from OpenCV BGR) for RAFT.
    Converts color, normalizes, and returns TF tensor.
    Note: Resizing is now handled separately *before* calling this if needed
          for the visualization background.
    """
    # frame_np is expected to be HxWxC BGR uint8 (already resized if necessary)
    img_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)

    img_tf = tf.convert_to_tensor(img_rgb, dtype=tf.uint8) # Keep uint8 here
    img_tf = tf.image.convert_image_dtype(img_tf, tf.float32) # Normalizes to [0, 1]
    img_tf.set_shape([target_h, target_w, 3]) # Ensure shape is known
    return img_tf

# --- 3. Visualization Function (NEW - Vector Field) ---
def visualize_flow_vectors(background_frame_bgr, flow_field_np, step=16, scale=1.0, color=(0, 255, 0), thickness=1):
    """
    Draws flow vectors (arrows) onto a background frame.

    Args:
        background_frame_bgr (np.ndarray): The background image (H, W, 3) in BGR uint8 format.
                                           This should be the *first* frame of the pair.
        flow_field_np (np.ndarray): The optical flow field (H, W, 2) float32.
        step (int): Grid spacing for drawing vectors.
        scale (float): Multiplier for vector length for visualization.
        color (tuple): BGR color for the vectors.
        thickness (int): Thickness of the vector lines.

    Returns:
        np.ndarray: A *new* frame (BGR uint8) with vectors drawn on it.
    """
    if background_frame_bgr is None or flow_field_np is None:
        print("Warning: Cannot visualize vectors, background or flow is None.")
        return background_frame_bgr # Return original or None

    if background_frame_bgr.shape[:2] != flow_field_np.shape[:2]:
        print(f"Warning: Background shape {background_frame_bgr.shape[:2]} doesn't match flow shape {flow_field_np.shape[:2]}. Resizing background.")
        background_frame_bgr = cv2.resize(background_frame_bgr, (flow_field_np.shape[1], flow_field_np.shape[0]))

    # Make a copy to avoid drawing on the original frame if it's reused
    output_frame = background_frame_bgr.copy()
    h, w = flow_field_np.shape[:2]

    # Generate grid points
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)

    # Get flow vectors at grid points
    fx, fy = flow_field_np[y, x].T * scale # Apply scaling here

    # Create lines array (start_x, start_y, end_x, end_y)
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5) # Round to nearest integer

    # Draw arrows
    for (x1, y1), (x2, y2) in lines:
        # Draw the arrowed line
        cv2.arrowedLine(output_frame, (x1, y1), (x2, y2), color, thickness, tipLength=0.3)

    return output_frame


# --- 4. Main Inference Logic (Modified for Video & Vector Viz) ---
if __name__ == "__main__":

    # --- Input Video Check and Dimension Calculation ---
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"Error: Input video not found at {INPUT_VIDEO_PATH}")
        exit()

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {INPUT_VIDEO_PATH}")
        exit()

    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input video: {original_width}x{original_height} @ {fps:.2f} FPS, {frame_count} frames")

    # Calculate target width, maintaining aspect ratio
    aspect_ratio = original_width / original_height
    TARGET_WIDTH = int(TARGET_HEIGHT * aspect_ratio)

    # Ensure width is divisible by 8
    if TARGET_WIDTH % 8 != 0:
        TARGET_WIDTH = (TARGET_WIDTH // 8) * 8
        print(f"Adjusted TARGET_WIDTH to be divisible by 8: {TARGET_WIDTH}")

    # Update global model input dimensions
    IMG_HEIGHT = TARGET_HEIGHT
    IMG_WIDTH = TARGET_WIDTH
    print(f"Processing frames resized to: {IMG_WIDTH}x{IMG_HEIGHT}")

    # --- Model Instantiation and Weight Loading ---
    weights_path = os.path.join(CHECKPOINT_DIR, FINAL_WEIGHTS_FILE)
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        cap.release()
        exit()

    print("Instantiating RAFT model...")
    raft_model = RAFT(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_iterations=NUM_ITERATIONS)

    # Build the model
    input_dtype = tf.float16 if POLICY == 'mixed_float16' else tf.float32
    print(f"Building model with dummy data (dtype: {input_dtype})...")
    dummy_img1 = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3], dtype=input_dtype)
    dummy_img2 = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3], dtype=input_dtype)
    try:
        _ = raft_model([dummy_img1, dummy_img2], training=False)
        print("Model built successfully.")
    except Exception as e:
        print(f"Warning: Error during initial model build: {e}")

    # Load the final weights
    print(f"Loading final weights from: {weights_path}")
    try:
        status = raft_model.load_weights(weights_path)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        cap.release()
        exit()

    # --- Video Processing Loop ---
    print(f"Starting video processing: {INPUT_VIDEO_PATH} -> {OUTPUT_VIDEO_PATH}")

    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (IMG_WIDTH, IMG_HEIGHT))
    if not out.isOpened():
       print(f"Error: Could not open video writer for {OUTPUT_VIDEO_PATH}")
       cap.release()
       exit()

    prev_frame_processed_tf = None
    prev_frame_resized_bgr = None # Store the resized BGR frame for visualization background
    frame_num = 0
    total_inf_time = 0

    pbar = tqdm(total=frame_count, desc="Processing Frames", unit="frame")

    while True:
        ret, frame_current_raw = cap.read()
        if not ret:
            print("\nEnd of video reached.")
            # If there's a pending last frame, write it without flow
            if prev_frame_resized_bgr is not None:
                 print("Writing last frame without flow.")
                 out.write(prev_frame_resized_bgr)
            break # End of video

        # Resize the current raw frame ONCE to the target size (for both model and viz background)
        current_frame_resized_bgr = cv2.resize(frame_current_raw, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # Preprocess the *resized* frame for the RAFT model -> TF Tensor [0,1]
        current_frame_processed_tf = preprocess_frame(current_frame_resized_bgr, IMG_HEIGHT, IMG_WIDTH)

        # --- Process Pair and Write Output ---
        if prev_frame_processed_tf is not None and prev_frame_resized_bgr is not None:
            # Prepare batch for inference
            img1_batch = tf.expand_dims(prev_frame_processed_tf, 0)
            img2_batch = tf.expand_dims(current_frame_processed_tf, 0)

            # Cast to model's compute dtype
            img1_batch = tf.cast(img1_batch, raft_model.compute_dtype)
            img2_batch = tf.cast(img2_batch, raft_model.compute_dtype)

            # Run inference
            start_inf = time.time()
            predicted_flow_list = raft_model([img1_batch, img2_batch], training=False)
            inf_time = time.time() - start_inf
            total_inf_time += inf_time

            output_viz_frame = None # Initialize frame to write

            if predicted_flow_list:
                # Get final flow, ensure float32, convert to NumPy
                predicted_flow_final = tf.cast(predicted_flow_list[-1], tf.float32)
                predicted_flow_np = predicted_flow_final[0].numpy() # Shape (H, W, 2)

                # Visualize flow vectors onto the *previous* resized BGR frame
                output_viz_frame = visualize_flow_vectors(
                    prev_frame_resized_bgr, # Use the stored BGR frame as background
                    predicted_flow_np,
                    step=VECTOR_STEP,
                    scale=VECTOR_SCALE,
                    color=VECTOR_COLOR,
                    thickness=VECTOR_THICKNESS
                )

            else:
                print(f"Warning: Model returned no flow for frame pair {frame_num-1}-{frame_num}. Writing previous frame without vectors.")
                # Write the previous frame without vectors if flow fails
                output_viz_frame = prev_frame_resized_bgr

            # Write the visualization frame to the output video
            if output_viz_frame is not None:
                out.write(output_viz_frame)
            else:
                # Fallback: write a black frame if something went wrong
                print(f"Error: Visualization frame is None for frame {frame_num-1}. Writing black frame.")
                black_frame = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
                out.write(black_frame)

        # --- Update state for next iteration ---
        prev_frame_processed_tf = current_frame_processed_tf
        prev_frame_resized_bgr = current_frame_resized_bgr # Store the resized BGR frame

        frame_num += 1
        pbar.update(1) # Update progress bar


    # --- Cleanup ---
    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    avg_inf_time = total_inf_time / (frame_num -1) if frame_num > 1 else 0
    print(f"\nVideo processing complete. Output saved to: {OUTPUT_VIDEO_PATH}")
    print(f"Processed {frame_num} frames.")
    if frame_num > 1:
        print(f"Average inference time per frame pair: {avg_inf_time:.4f}s")