import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 # Use OpenCV for image reading/writing if needed, and visualization background
import time

# --- Configuration (Match Training Parameters) ---
IMG_HEIGHT = 384
IMG_WIDTH = 512
NUM_ITERATIONS = 8 # Must match the trained model's iterations
CHECKPOINT_DIR = './raft_checkpoints' # Directory where weights are saved
DATA_DIR = './data' # Directory containing frame0.png and frame1.png
FINAL_WEIGHTS_FILE = 'raft_final_weights.weights.h5' # Name of the saved weights file

# --- Mixed Precision Setup (Important if trained with it) ---
# Set this policy BEFORE model instantiation and weight loading
# Use the same policy as during training (e.g., 'mixed_float16' or 'float32')
POLICY = 'mixed_float16' # Or 'float32' if not trained with mixed precision
print(f"Setting global policy to: {POLICY}")
tf.keras.mixed_precision.set_global_policy(POLICY)

# --- 1. Re-define Model Architecture (Must be identical to training) ---
# Copy all necessary model classes and helper functions from the training script:
# BasicBlock, DownsampleBlock, FeatureEncoder, ContextEncoder, ConvGRUCell,
# UpdateBlock, build_correlation_volume, upsample_flow, RAFT

# Basic Residual Block
def BasicBlock(filters, stride=1):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(dtype=tf.float32), # Specify dtype for norm layers
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(dtype=tf.float32),
    ])

def DownsampleBlock(filters, stride=2):
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
    def __init__(self, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_iterations=NUM_ITERATIONS, hidden_dim=128, context_dim=128, corr_levels=1, corr_radius=4, name='raft', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_iterations = num_iterations; self.hidden_dim = hidden_dim; self.context_dim = context_dim
        self.corr_levels = corr_levels; self.corr_radius = corr_radius
        self.feature_encoder = FeatureEncoder()
        self.context_encoder = ContextEncoder()
        self.update_block = UpdateBlock(iterations=num_iterations, hidden_dim=hidden_dim, context_dim=context_dim, corr_levels=corr_levels, corr_radius=corr_radius)

    @tf.function
    def upsample_flow(self, flow, target_height, target_width):
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

# --- 2. Preprocessing Function ---
def load_and_preprocess_image(path):
    print(f"Loading and preprocessing image: {path}")
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method='bilinear')
    img = tf.image.convert_image_dtype(img, tf.float32) # Normalizes to [0, 1]
    img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3]) # Ensure shape is known
    return img

# --- 3. Visualization Functions (Copied from training script) ---
# Helper function to visualize flow (Middlebury color code) - Needed by visualize_flow
def make_color_wheel():
    RY = 15; YG = 6; GC = 4; CB = 11; BM = 13; MR = 6; ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0; colorwheel[0:RY, 0] = 255; colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY)/RY); col += RY
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0, YG)/YG); colorwheel[col:col+YG, 1] = 255; col += YG
    colorwheel[col:col+GC, 1] = 255; colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0, GC)/GC); col += GC
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(0, CB)/CB); colorwheel[col:col+CB, 2] = 255; col += CB
    colorwheel[col:col+BM, 2] = 255; colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0, BM)/BM); col += BM
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(0, MR)/MR); colorwheel[col:col+MR, 0] = 255
    return colorwheel.astype(np.uint8)

def flow_to_color(flow, convert_to_bgr=False):
    """Converts optical flow (u, v) to a color image using Middlebury color scheme."""
    if flow is None: return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    UNKNOWN_FLOW_THRESH = 1e7; SMALL_FLOW = 1e-9
    if not isinstance(flow, np.ndarray) or flow.ndim != 3 or flow.shape[2] != 2:
        print(f"Warning: Invalid flow shape in flow_to_color: {flow.shape if isinstance(flow, np.ndarray) else type(flow)}")
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    height, width, _ = flow.shape; img = np.zeros((height, width, 3), dtype=np.uint8)
    colorwheel = make_color_wheel(); ncols = colorwheel.shape[0]
    u, v = np.nan_to_num(flow[..., 0]), np.nan_to_num(flow[..., 1])
    mag = np.sqrt(u**2 + v**2); ang = np.arctan2(-v, -u) / np.pi; ang = (ang + 1.0) / 2.0
    valid_mag = mag[np.abs(mag) < UNKNOWN_FLOW_THRESH]; mag_max = np.max(valid_mag) if valid_mag.size > 0 else 0.0
    mag_norm = np.clip(mag / mag_max, 0, 1) if mag_max > SMALL_FLOW else np.zeros_like(mag)
    fk = (ang * (ncols - 1)); k0 = np.floor(fk).astype(np.int32); k1 = (k0 + 1) % ncols; f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]; col0 = tmp[k0] / 255.0; col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1; col = 1 - mag_norm * (1 - col); img[:, :, i] = np.floor(255.0 * col)
    idx_unknown = (np.abs(u) > UNKNOWN_FLOW_THRESH) | (np.abs(v) > UNKNOWN_FLOW_THRESH) | (mag > mag_max * 1.1 if mag_max > SMALL_FLOW else False)
    img[idx_unknown] = 0
    if convert_to_bgr: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def visualize_flow(image1_np, flow_pred_np, filename_prefix="flow_vis"):
    """Generates and saves flow visualizations (magnitude and vector field)."""
    if image1_np is None or flow_pred_np is None:
        print("Warning: Cannot visualize flow, input image or flow is None.")
        return

    image1_np = np.asarray(image1_np); flow_pred_np = np.asarray(flow_pred_np)
    if image1_np.dtype == np.float32 or image1_np.dtype == np.float64: image1_np = np.clip(image1_np, 0, 1)
    if flow_pred_np.ndim != 3 or flow_pred_np.shape[-1] != 2:
        print(f"Warning: Invalid flow shape for visualization: {flow_pred_np.shape}. Skipping.")
        return
    h, w, _ = flow_pred_np.shape; img_h, img_w, _ = image1_np.shape
    if img_h != h or img_w != w:
        print(f"Warning: Resizing image ({img_h}x{img_w}) to match flow ({h}x{w}) for display.")
        image1_np = cv2.resize(image1_np, (w, h))

    # 1. Flow Magnitude Heatmap
    try:
        plt.figure(figsize=(12, 9))
        magnitude = np.sqrt(np.sum(flow_pred_np**2, axis=-1))
        im = plt.imshow(magnitude, cmap='viridis')
        plt.colorbar(im, label='Flow Magnitude (pixels)')
        plt.title(f'{os.path.basename(filename_prefix)} - Flow Magnitude')
        plt.axis('off'); plt.tight_layout()
        plt.savefig(f"{filename_prefix}_magnitude.png"); plt.close()
        print(f"Saved flow magnitude heatmap to {filename_prefix}_magnitude.png")
    except Exception as e: print(f"Error generating magnitude plot: {e}"); plt.close()

    # 2. Vector Field Overlay (Quiver Plot)
    try:
        step = max(1, min(h, w) // 32)
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        # Ensure indices are within bounds AFTER calculating them
        y = np.clip(y, 0, h - 1)
        x = np.clip(x, 0, w - 1)
        fx, fy = flow_pred_np[y, x].T

        plt.figure(figsize=(12, 9))
        plt.imshow(image1_np)
        plt.quiver(x, y, fx, fy, color='red', scale=None, scale_units='xy', angles='xy', headwidth=5, headlength=6, width=0.0015)
        plt.title(f'{os.path.basename(filename_prefix)} - Flow Vectors (Overlay)')
        plt.axis('off'); plt.tight_layout()
        plt.savefig(f"{filename_prefix}_vectors.png"); plt.close()
        print(f"Saved flow vector field to {filename_prefix}_vectors.png")
    except Exception as e: print(f"Error generating vector plot: {e}"); plt.close()

    # 3. Middlebury Color Visualization (Optional but often useful)
    try:
        flow_color_img = flow_to_color(flow_pred_np)
        plt.figure(figsize=(12, 9))
        plt.imshow(flow_color_img)
        plt.title(f'{os.path.basename(filename_prefix)} - Flow (Color)')
        plt.axis('off'); plt.tight_layout()
        plt.savefig(f"{filename_prefix}_color.png"); plt.close()
        print(f"Saved color flow visualization to {filename_prefix}_color.png")
    except Exception as e: print(f"Error generating color flow plot: {e}"); plt.close()


# --- 4. Main Inference Logic ---
if __name__ == "__main__":
    # Define paths
    frame0_path = os.path.join(DATA_DIR, 'frame0.png')
    frame1_path = os.path.join(DATA_DIR, 'frame1.png')
    weights_path = os.path.join(CHECKPOINT_DIR, FINAL_WEIGHTS_FILE)
    output_prefix = "inference_flow_0_1" # Prefix for saved visualization files

    # Check if files exist
    if not os.path.exists(frame0_path):
        print(f"Error: Input file not found at {frame0_path}")
        exit()
    if not os.path.exists(frame1_path):
        print(f"Error: Input file not found at {frame1_path}")
        exit()
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        exit()

    # Instantiate the model
    print("Instantiating RAFT model...")
    raft_model = RAFT(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_iterations=NUM_ITERATIONS)

    # Build the model by calling it with dummy data (ensures layers are created before loading weights)
    # Use correct dtype based on policy
    input_dtype = tf.float16 if POLICY == 'mixed_float16' else tf.float32
    print(f"Building model with dummy data (dtype: {input_dtype})...")
    dummy_img1 = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3], dtype=input_dtype)
    dummy_img2 = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3], dtype=input_dtype)
    try:
        _ = raft_model([dummy_img1, dummy_img2], training=False)
        print("Model built successfully.")
    except Exception as e:
        print(f"Warning: Error during initial model build (might be ok if layers already exist): {e}")


    # Load the final weights
    print(f"Loading final weights from: {weights_path}")
    try:
        # Use expect_partial() if you only saved model weights and not optimizer state etc.
        status = raft_model.load_weights(weights_path)#.expect_partial() # Use .expect_partial() if only model weights are saved
        # status.assert_existing_objects_matched() # Optional: Verify all weights were loaded
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        exit()

    # Load and preprocess the input frames
    img1_inf = load_and_preprocess_image(frame0_path)
    img2_inf = load_and_preprocess_image(frame1_path)

    # Add batch dimension
    img1_inf_batch = tf.expand_dims(img1_inf, 0)
    img2_inf_batch = tf.expand_dims(img2_inf, 0)

    # Cast to model's compute dtype if necessary (usually handled by layers, but explicit cast can help debugging)
    img1_inf_batch = tf.cast(img1_inf_batch, raft_model.compute_dtype)
    img2_inf_batch = tf.cast(img2_inf_batch, raft_model.compute_dtype)
    print(f"Input tensor dtype for inference: {img1_inf_batch.dtype}")

    # Run inference
    print("Running inference...")
    start_inf = time.time()
    predicted_flow_list = raft_model([img1_inf_batch, img2_inf_batch], training=False)
    inf_time = time.time() - start_inf
    print(f"Inference completed in {inf_time:.3f}s")

    # Extract the final flow prediction (usually the last one in the list)
    if not predicted_flow_list:
        print("Error: Model did not return any flow predictions.")
        exit()

    # --- IMPORTANT: Ensure final flow is float32 for visualization/numpy conversion ---
    predicted_flow_final = tf.cast(predicted_flow_list[-1], tf.float32)

    # Remove batch dimension and convert to NumPy
    predicted_flow_np = predicted_flow_final[0].numpy()
    img1_np = img1_inf.numpy() # Use the original preprocessed image (float32 0-1 range) for visualization

    print(f"Predicted flow shape: {predicted_flow_np.shape}, dtype: {predicted_flow_np.dtype}")
    print(f"Min/Max flow values (u): {np.min(predicted_flow_np[..., 0]):.2f}/{np.max(predicted_flow_np[..., 0]):.2f}")
    print(f"Min/Max flow values (v): {np.min(predicted_flow_np[..., 1]):.2f}/{np.max(predicted_flow_np[..., 1]):.2f}")


    # Visualize the flow
    print(f"Generating visualizations with prefix '{output_prefix}'...")
    visualize_flow(img1_np, predicted_flow_np, filename_prefix=output_prefix)

    print("\nInference and visualization complete.")