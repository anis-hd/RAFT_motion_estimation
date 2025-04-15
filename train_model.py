import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import time
from glob import glob
from tqdm import tqdm
import cv2 # <-- Added import
import datetime # For TensorBoard logging
tf.keras.mixed_precision.set_global_policy('mixed_float16')
# --- Configuration ---
IMG_HEIGHT = 384
IMG_WIDTH = 512
BATCH_SIZE = 2
EPOCHS = 1000 # Total desired epochs
LEARNING_RATE = 1e-4
NUM_ITERATIONS = 8
DATA_DIR = './train'
CHECKPOINT_DIR = './raft_checkpoints'
SAVED_MODEL_DIR = './raft_saved_model'
LOG_DIR = './raft_logs'
VISUALIZATION_SCENE = 'market_2'

# --- Frequency for saving checkpoints (optional, default is per epoch) ---
# Set SAVE_FREQ > 0 to save every N steps instead of every epoch
# Set SAVE_FREQ = None or 0 to save only at the end of each epoch
SAVE_FREQ = None # Save per epoch by default

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Mixed Precision Setup (Optional) ---
# ... (keep your mixed precision setup if using) ...

# --- 1. Dataset Structure and Preprocessing ---
# ... (keep your dataset functions: read_flo_file, parse_frame_num, etc.) ...
# Function to parse .flo files (standard Middlebury format)
def read_flo_file(filename_tensor):
    """
    Read a .flo file (Middlebury format). Expects filename as TF tensor.
    Returns tuple: (flow_data: np.float32[H, W, 2], height: np.int32, width: np.int32)
    """
    filename_bytes = filename_tensor.numpy()
    try:
        with open(filename_bytes, 'rb') as f:
            magic = np.frombuffer(f.read(4), np.float32, count=1)
            if not np.isclose(magic[0], 202021.25):
                fname_str = filename_bytes.decode('utf-8', errors='replace')
                raise ValueError(f'Magic number incorrect ({magic[0]}). Invalid .flo file: {fname_str}')

            width = np.frombuffer(f.read(4), np.int32, count=1)[0]
            height = np.frombuffer(f.read(4), np.int32, count=1)[0]

            if width <= 0 or height <= 0 or width * height > 5000*5000:
                 fname_str = filename_bytes.decode('utf-8', errors='replace')
                 raise ValueError(f"Invalid dimensions ({width}x{height}) read from file: {fname_str}")

            data = np.frombuffer(f.read(), np.float32, count=-1)
            expected_elements = height * width * 2
            if data.size != expected_elements:
                fname_str = filename_bytes.decode('utf-8', errors='replace')
                raise ValueError(f"Incorrect data size read from {fname_str}. Expected {expected_elements} floats, got {data.size}")

            flow = data.reshape((height, width, 2))
        # Return flow data AND original dimensions
        return flow.astype(np.float32), np.int32(height), np.int32(width)
    except FileNotFoundError:
        fname_str = filename_bytes.decode('utf-8', errors='replace')
        print(f"Error: File not found: {fname_str}")
        raise
    except Exception as e:
        fname_str = filename_bytes.decode('utf-8', errors='replace')
        print(f"Error processing file {fname_str}: {e}")
        raise

def parse_frame_num(filename):
    """Extracts frame number from filename like frame_XXXX.png/flo"""
    match = re.search(r'frame_(\d+)\.(png|flo)', os.path.basename(filename))
    return int(match.group(1)) if match else -1

def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method='bilinear')
    img = tf.image.convert_image_dtype(img, tf.float32) # Normalizes to [0, 1]
    img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3]) # Ensure shape is known
    return img

def load_and_preprocess_flow(path_tensor):
    """
    Loads, resizes, and SCALES flow from a .flo file path tensor.
    Uses tf.py_function to wrap read_flo_file.
    """
    # Use tf.py_function to wrap the numpy .flo reader
    # Specify input tensor and output types (flow data, original height, original width)
    flow, h_orig, w_orig = tf.py_function(
        read_flo_file,
        [path_tensor],
        [tf.float32, tf.int32, tf.int32]
    )

    # Set shape hint for flow data *before* resizing (important for graph construction)
    flow.set_shape([None, None, 2])
    # Original height/width are scalars
    h_orig.set_shape([])
    w_orig.set_shape([])

    # Resize flow map to target dimensions
    flow_resized = tf.image.resize(flow, [IMG_HEIGHT, IMG_WIDTH], method='bilinear')

    # --- Scale Flow Vectors ---
    # Calculate scaling factors based on original vs target size
    # Cast original dimensions to float32 for division
    h_orig_f = tf.cast(h_orig, tf.float32)
    w_orig_f = tf.cast(w_orig, tf.float32)

    # Avoid division by zero if original dimension was somehow 0 (should be caught by read_flo)
    scale_h = tf.cast(IMG_HEIGHT, tf.float32) / tf.maximum(h_orig_f, 1.0)
    scale_w = tf.cast(IMG_WIDTH, tf.float32) / tf.maximum(w_orig_f, 1.0)

    # Apply scaling to the u and v components of the resized flow
    u = flow_resized[..., 0] * scale_w
    v = flow_resized[..., 1] * scale_h
    flow_scaled = tf.stack([u, v], axis=-1)
    # --- End Scaling ---

    # Set the final static shape for the processed flow tensor
    flow_scaled.set_shape([IMG_HEIGHT, IMG_WIDTH, 2])

    return flow_scaled

def configure_dataset(data_dir, is_training=True):
    clean_dir = os.path.join(data_dir, 'clean')
    flow_dir = os.path.join(data_dir, 'flow')
    image_pairs = []
    flow_paths = []

    scene_dirs = sorted([d for d in os.listdir(clean_dir) if os.path.isdir(os.path.join(clean_dir, d))])

    for scene in scene_dirs:
        scene_clean_path = os.path.join(clean_dir, scene)
        scene_flow_path = os.path.join(flow_dir, scene)

        if not os.path.isdir(scene_flow_path):
            print(f"Warning: Flow directory not found for scene {scene}, skipping.")
            continue

        frames = sorted(glob(os.path.join(scene_clean_path, 'frame_*.png')), key=parse_frame_num)

        for i in range(len(frames) - 1):
            frame1_path = frames[i]
            frame2_path = frames[i+1]
            frame1_num = parse_frame_num(frame1_path)
            frame2_num = parse_frame_num(frame2_path)

            if frame1_num != -1 and frame2_num == frame1_num + 1:
                flow_filename = f"frame_{frame1_num:04d}.flo"
                flow_path = os.path.join(scene_flow_path, flow_filename)

                if os.path.exists(frame1_path) and os.path.exists(frame2_path) and os.path.exists(flow_path):
                    image_pairs.append((frame1_path, frame2_path))
                    flow_paths.append(flow_path)
                # else:
                #     print(f"Warning: Missing file for pair {frame1_path}, {frame2_path}, {flow_path}")


    print(f"Found {len(image_pairs)} image pairs and flows.")
    if not image_pairs:
        raise ValueError("No image pairs found. Check dataset structure and paths.")

    img_path_ds = tf.data.Dataset.from_tensor_slices([p[0] for p in image_pairs])
    img2_path_ds = tf.data.Dataset.from_tensor_slices([p[1] for p in image_pairs])
    flow_path_ds = tf.data.Dataset.from_tensor_slices(flow_paths)

    image1_ds = img_path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    image2_ds = img2_path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    flow_ds = flow_path_ds.map(load_and_preprocess_flow, num_parallel_calls=tf.data.AUTOTUNE) # Use updated function

    # Combine into a single dataset
    dataset = tf.data.Dataset.zip(((image1_ds, image2_ds), flow_ds))

    # --- Optional Augmentations --- (Apply only during training)
    if is_training:
        # Basic spatial augmentation example (random horizontal flip)
        @tf.function
        def augment(image1, image2, flow):
            do_flip = tf.random.uniform(()) > 0.5
            if do_flip:
                image1 = tf.image.flip_left_right(image1)
                image2 = tf.image.flip_left_right(image2)
                # Flip flow horizontally and negate u component
                flow = tf.image.flip_left_right(flow)
                flow = tf.stack([-flow[..., 0], flow[..., 1]], axis=-1)
            return image1, image2, flow

        # Define a mapping function that accepts the unpacked elements
        # This function explicitly takes TWO arguments, matching how map seems to unpack
        def map_augment_unpacked(images_tuple, flow_gt):
            # Unpack the images tuple
            img1, img2 = images_tuple

            # Call the decorated augment function
            img1_aug, img2_aug, flow_aug = augment(img1, img2, flow_gt)

            # Repack into the desired structure
            return (img1_aug, img2_aug), flow_aug

        # Apply this new mapping function using map
        dataset = dataset.map(map_augment_unpacked, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(image_pairs), 100)) # Shuffle

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

# --- 2. RAFT Model Implementation ---
# ... (keep your model classes: BasicBlock, DownsampleBlock, FeatureEncoder, etc.) ...
# Basic Residual Block
def BasicBlock(filters, stride=1):
    # Layer normalization/instance normalization can be float32 even in mixed precision
    # Usually okay, but check TF docs if issues arise.
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
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) # Shape: (B, H/8, W/8, 128)

        x = self.conv_out(x) # Shape: (B, H/8, W/8, 256)
        # Cast output to compute dtype if using mixed precision
        # return tf.cast(x, self.compute_dtype) # No, conv_out does this implicitly
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
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) # Shape: (B, H/8, W/8, 128)

        x = self.conv_out(x) # Shape: (B, H/8, W/8, 128)
        return x

# Convolutional GRU Cell
class ConvGRUCell(tf.keras.layers.Layer):
    def __init__(self, hidden_filters, input_filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_filters = hidden_filters
        self.input_filters = input_filters
        self.kernel_size = kernel_size
        # TF will infer spatial dimensions, state needs hidden_filters channels
        self.state_size = tf.TensorShape([None, None, hidden_filters])

        # Gates and candidate calculation. Use sigmoid/tanh which handle mixed precision well.
        self.conv_update = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', activation='sigmoid', kernel_initializer='glorot_uniform')
        self.conv_reset = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', activation='sigmoid', kernel_initializer='glorot_uniform')
        self.conv_candidate = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', activation='tanh', kernel_initializer='glorot_uniform')

    def build(self, input_shape):
        # input_shape is [(batch, H, W, input_filters), (batch, H, W, hidden_filters)] (inputs, states)
        # Conv2D layers create weights in __init__ or on first call.
        pass

    def call(self, inputs, states):
        h_prev = states[0] # GRU has one state tensor

        # Ensure inputs and state are compatible types (e.g., float16 or float32)
        # This should happen automatically with mixed precision policy
        # tf.print("GRU Input dtype:", inputs.dtype, "State dtype:", h_prev.dtype)

        # Concatenate previous hidden state and input along the channel dimension
        combined_input_h = tf.concat([inputs, h_prev], axis=-1)

        # Calculate gates
        update_gate = self.conv_update(combined_input_h)
        reset_gate = self.conv_reset(combined_input_h)

        # Calculate candidate hidden state
        combined_input_reset_h = tf.concat([inputs, reset_gate * h_prev], axis=-1)
        candidate_h = self.conv_candidate(combined_input_reset_h)

        # Calculate new hidden state
        new_h = (1. - update_gate) * h_prev + update_gate * candidate_h

        return new_h, [new_h] # Output and new state (must be a list)

# Motion Encoder and Update Block
class UpdateBlock(tf.keras.Model):
    def __init__(self, iterations, hidden_dim=128, context_dim=128, corr_levels=1, corr_radius=4, name='update_block', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iterations = iterations
        self.hidden_dim = hidden_dim

        # Calculate dimensions based on inputs
        corr_feature_dim = (2 * corr_radius + 1)**2 * corr_levels
        motion_encoder_input_dim = corr_feature_dim + 2 # Correlation features + flow (u,v)
        motion_encoder_output_dim = 32 # Chosen dimension for motion features

        # Determine GRU input features dimension
        # GRU input = motion_features (32) + context 'inp' features
        # 'inp' features dim = max(0, total_context_dim - hidden_dim_for_gru_state)
        inp_dim = max(0, context_dim - hidden_dim)
        gru_input_total_dim = motion_encoder_output_dim + inp_dim

        # Motion Encoder
        self.motion_encoder = tf.keras.Sequential([
            # Input shape not explicitly needed here, inferred on first call
            tf.keras.layers.Conv2D(128, 1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(motion_encoder_output_dim, 3, padding='same', activation='relu') # Output 32 dims
            ], name='motion_encoder')

        # GRU Cell
        self.gru_cell = ConvGRUCell(hidden_filters=hidden_dim, input_filters=gru_input_total_dim)

        # Flow Head: Predicts flow *update* from GRU hidden state
        self.flow_head = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(2, 3, padding='same') # Output delta_flow (dx, dy)
            # Final Conv2D layer should output float32, even in mixed precision,
            # as flow represents precise offsets. Keras handles this automatically.
            ], name='flow_head')

    def call(self, net, inp, corr_features, flow_init=None):
        # net: Initial hidden state (B, H/8, W/8, hidden_dim) - tanh applied
        # inp: Context features for GRU input (B, H/8, W/8, inp_dim) - relu applied
        # corr_features: Correlation features (B, H/8, W/8, corr_feature_dim)
        # flow_init: Initial flow guess (optional)

        shape_tensor = tf.shape(net)
        b, h, w = shape_tensor[0], shape_tensor[1], shape_tensor[2]

        if flow_init is None:
            flow = tf.zeros([b, h, w, 2], dtype=tf.float32) # Flow is always float32
        else:
            # Ensure init flow is float32
            flow = tf.cast(flow_init, tf.float32)

        hidden_state = net # Initialize GRU hidden state

        flow_predictions = []

        # Manual GRU loop
        for iter in range(self.iterations):
            # Detach flow estimate from gradient computation for stability
            flow = tf.stop_gradient(flow)

            # Concatenate correlation features and current flow estimate
            motion_input = tf.concat([corr_features, flow], axis=-1)
            # Cast motion input if needed by motion encoder (depends on mixed precision policy)
            # motion_input_casted = tf.cast(motion_input, self.compute_dtype) # Keras layers handle this
            motion_features = self.motion_encoder(motion_input) # Output: (B, H/8, W/8, 32)

            # Prepare GRU input: concatenate motion features and 'inp' context features
            gru_input = tf.concat([motion_features, inp], axis=-1)
            # tf.print("GRU Input shape:", tf.shape(gru_input), "dtype:", gru_input.dtype, "State dtype:", hidden_state.dtype)

            # Pass through GRU Cell (manages its own state list internally)
            hidden_state, [hidden_state] = self.gru_cell(gru_input, [hidden_state])

            # Predict flow update (delta_flow) from hidden state
            # Cast hidden_state? No, flow_head input layer will handle casting if needed.
            delta_flow = self.flow_head(hidden_state)
            # Ensure delta_flow is float32 for the update
            delta_flow = tf.cast(delta_flow, tf.float32)

            # Update flow estimate
            flow = flow + delta_flow

            # Store intermediate prediction (at H/8 resolution)
            flow_predictions.append(flow)

        return flow_predictions # List of flow fields at H/8 resolution

# --- Helper for Correlation ---
def build_correlation_volume(fmap1, fmap2, radius=4):
    """
    Builds a correlation volume using manual lookup.

    NOTE: This manual implementation using gather is often a performance
    bottleneck compared to optimized CUDA kernels or potentially alternatives
    like tfa.layers.CorrelationCost (if its behavior matches).
    Consider optimizing this part for better performance if needed.

    Args:
        fmap1: Features from image 1 (B, H/8, W/8, C)
        fmap2: Features from image 2 (B, H/8, W/8, C)
        radius: Lookup radius.

    Returns:
        Correlation volume (B, H/8, W/8, (2*radius+1)**2)
    """
    # Ensure inputs are in the compute dtype (e.g., float16 if mixed precision active)
    compute_dtype = fmap1.dtype
    fmap2 = tf.cast(fmap2, compute_dtype)

    batch_size, h, w, c = tf.shape(fmap1)[0], tf.shape(fmap1)[1], tf.shape(fmap1)[2], tf.shape(fmap1)[3]

    # Pad fmap2 for lookups near boundaries
    pad_size = radius
    fmap2_padded = tf.pad(fmap2, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='CONSTANT')

    # Create grid of coordinates for fmap1
    gy, gx = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij') # H, W
    coords_base = tf.stack([gy, gx], axis=-1) # H, W, 2
    coords_base = tf.cast(coords_base, tf.int32)
    # Expand dims for batch and neighborhood dimensions
    coords_base = tf.expand_dims(tf.expand_dims(coords_base, 0), -2) # 1, H, W, 1, 2
    coords_base = tf.tile(coords_base, [batch_size, 1, 1, 1, 1]) # B, H, W, 1, 2

    # Create offsets for the neighborhood
    dy, dx = tf.meshgrid(tf.range(-radius, radius + 1), tf.range(-radius, radius + 1), indexing='ij')
    delta = tf.stack([dy, dx], axis=-1) # (2R+1), (2R+1), 2
    num_neighbors = (2*radius+1)**2
    delta = tf.reshape(delta, [1, 1, 1, num_neighbors, 2]) # 1, 1, 1, K*K, 2
    delta = tf.cast(delta, tf.int32)

    # Calculate lookup coordinates in padded fmap2
    lookup_coords = coords_base + delta + pad_size # Broadcasting: B, H, W, K*K, 2

    # Gather features from fmap2_padded using batch indices
    batch_indices = tf.range(batch_size)
    batch_indices = tf.reshape(batch_indices, [batch_size, 1, 1, 1])
    batch_indices = tf.tile(batch_indices, [1, h, w, num_neighbors]) # B, H, W, K*K

    lookup_indices = tf.stack([
        batch_indices,
        lookup_coords[..., 0], # y coordinates
        lookup_coords[..., 1]  # x coordinates
    ], axis=-1) # Shape: B, H, W, K*K, 3 (batch, y, x)

    # Gather neighboring features from fmap2
    fmap2_neighbors = tf.gather_nd(fmap2_padded, lookup_indices) # B, H, W, K*K, C

    # Expand fmap1 for broadcasting dot product
    fmap1_expanded = tf.expand_dims(fmap1, axis=3) # B, H, W, 1, C

    # Compute dot product correlation: (B,H,W,1,C) * (B,H,W,K*K,C) -> (B,H,W,K*K,C)
    # Sum over channel dim C -> (B, H, W, K*K)
    correlation = tf.reduce_sum(fmap1_expanded * fmap2_neighbors, axis=-1) # B, H, W, K*K

    # --- MODIFICATION START ---
    # Cast correlation to float32 before dividing to ensure type compatibility
    correlation_float32 = tf.cast(correlation, tf.float32)
    # Normalize by feature dimension C (use float32 for potentially small C)
    correlation_normalized = correlation_float32 / tf.cast(c, tf.float32)
    # --- MODIFICATION END ---

    # Cast correlation back to compute_dtype if needed (usually done by next layer)
    # correlation = tf.cast(correlation, compute_dtype) # Commented out - let subsequent layers handle casting if necessary

    # Shape: B, H/8, W/8, (2R+1)*(2R+1)
    # Return the normalized correlation (which is now float32)
    return correlation_normalized

# --- RAFT Model ---
class RAFT(tf.keras.Model):
    def __init__(self, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_iterations=NUM_ITERATIONS, hidden_dim=128, context_dim=128, corr_levels=1, corr_radius=4, name='raft', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_iterations = num_iterations
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.corr_levels = corr_levels # Using only 1 level for simplicity
        self.corr_radius = corr_radius

        # Shared Feature Encoder
        self.feature_encoder = FeatureEncoder()
        # Context Encoder (only for image1)
        self.context_encoder = ContextEncoder()
        # Update Block
        self.update_block = UpdateBlock(iterations=num_iterations,
                                         hidden_dim=hidden_dim,
                                         context_dim=context_dim, # Pass total context dim
                                         corr_levels=corr_levels,
                                         corr_radius=corr_radius)

    @tf.function # Decorate upsample for potential graph optimization
    def upsample_flow(self, flow, target_height, target_width):
        """Upsamples flow field to target resolution using bilinear interp + scaling."""
        # flow shape: (B, H_low, W_low, 2)
        # Ensure flow input is float32 for precise calculations
        flow = tf.cast(flow, tf.float32)

        shape_tensor = tf.shape(flow)
        b, h_low, w_low = shape_tensor[0], shape_tensor[1], shape_tensor[2]

        # Calculate scaling factor dynamically
        scale_factor_h = tf.cast(target_height, tf.float32) / tf.cast(h_low, tf.float32)
        scale_factor_w = tf.cast(target_width, tf.float32) / tf.cast(w_low, tf.float32)

        # Resize using bilinear interpolation
        flow_upsampled = tf.image.resize(flow, [target_height, target_width], method='bilinear')

        # Scale flow vectors correctly
        u = flow_upsampled[..., 0] * scale_factor_w
        v = flow_upsampled[..., 1] * scale_factor_h
        flow_scaled = tf.stack([u, v], axis=-1)

        return flow_scaled # Output is float32

    def call(self, inputs, training=False):
        image1, image2 = inputs # Pair of images (B, H, W, 3)

        # Get target H/W from input image1 shape dynamically
        target_height = tf.shape(image1)[1]
        target_width = tf.shape(image1)[2]

        # 1. Feature Extraction (Shared weights)
        # Cast inputs if using mixed precision
        # image1 = tf.cast(image1, self.compute_dtype)
        # image2 = tf.cast(image2, self.compute_dtype)
        fmap1 = self.feature_encoder(image1) # (B, H/8, W/8, 256)
        fmap2 = self.feature_encoder(image2) # (B, H/8, W/8, 256)

        # 2. Context Extraction (from image1 only)
        context_fmap = self.context_encoder(image1) # (B, H/8, W/8, context_dim=128)

        # 3. Context Split Logic (Standard RAFT practice)
        # Split context features into initial hidden state ('net') and GRU input features ('inp')
        # Ensure split sizes sum correctly
        split_sizes = [self.hidden_dim, max(0, self.context_dim - self.hidden_dim)]
        if sum(split_sizes) != self.context_dim:
             raise ValueError(f"Context split sizes {split_sizes} do not sum to context dimension {self.context_dim}")

        net, inp = tf.split(context_fmap, split_sizes, axis=-1)

        # Apply activations for GRU init/input
        net = tf.tanh(net) # Initial hidden state for GRU
        inp = tf.nn.relu(inp) # Input features for GRU updates

        # net and inp should be in the model's compute dtype (e.g., float16)
        # tf.print("Net dtype:", net.dtype, "Inp dtype:", inp.dtype)

        # 4. Correlation Volume / Lookup
        # Correlation uses feature maps (potentially float16)
        corr_features = build_correlation_volume(fmap1, fmap2, radius=self.corr_radius)
        # tf.print("Corr features dtype:", corr_features.dtype)

        # 5. Recurrent Flow Updates using UpdateBlock
        # Update block handles internal types, takes net/inp/corr
        flow_predictions_low_res = self.update_block(net, inp, corr_features, flow_init=None)

        # 6. Upsample all predicted flows to target resolution for loss/output
        flow_predictions_upsampled = []
        for flow_lr in flow_predictions_low_res:
            # Upsampling uses float32 internally for precision, returns float32
            flow_up = self.upsample_flow(flow_lr, target_height, target_width)
            flow_predictions_upsampled.append(flow_up)

        # Return the list of *upsampled* float32 flow predictions
        return flow_predictions_upsampled

# --- 3. Training Pipeline ---

# Loss Function: End-Point Error (EPE)
@tf.function
def endpoint_error(flow_gt, flow_pred):
    """Calculates average End-Point Error (EPE)."""
    # Ensure inputs are float32 for precise distance calculation
    flow_gt = tf.cast(flow_gt, tf.float32)
    flow_pred = tf.cast(flow_pred, tf.float32)

    # Ensure shapes match before calculation
    tf.debugging.assert_equal(tf.shape(flow_gt), tf.shape(flow_pred),
                              message="Shapes of flow_gt and flow_pred must match in EPE.")
    # flow_gt, flow_pred: shape (B, H, W, 2)
    sq_diff = tf.square(flow_gt - flow_pred) # (B, H, W, 2)
    epe_map = tf.sqrt(tf.reduce_sum(sq_diff, axis=-1) + 1e-8) # Add epsilon for stability, (B, H, W)
    # Average EPE over batch and spatial dimensions
    avg_epe = tf.reduce_mean(epe_map)
    return avg_epe

# RAFT Loss: Weighted sum of EPEs over iterations
@tf.function
def raft_loss(flow_gt, flow_predictions, gamma=0.8):
    """Calculates the weighted sum of EPEs for RAFT predictions."""
    # flow_gt: Ground truth flow (B, H, W, 2) - float32
    # flow_predictions: List of predicted flows [(B, H, W, 2), ...] from RAFT (float32)
    # gamma: Weight decay factor

    flow_gt = tf.cast(flow_gt, tf.float32) # Ensure GT is float32

    if not flow_predictions:
         tf.print("Warning: flow_predictions list is empty in raft_loss.")
         return tf.constant(0.0, dtype=tf.float32)

    n_predictions = len(flow_predictions)
    total_loss = tf.constant(0.0, dtype=tf.float32)

    for i in range(n_predictions):
        flow_pred = tf.cast(flow_predictions[i], tf.float32) # Ensure pred is float32
        i_weight = tf.cast(gamma**(n_predictions - 1 - i), dtype=tf.float32)
        i_epe = endpoint_error(flow_gt, flow_pred)
        total_loss += i_weight * i_epe

    return total_loss

# --- Optimizer and Model Instantiation ---
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# If using mixed precision float16, wrap the optimizer:
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
print("Using LossScaleOptimizer for mixed precision.")

# Instantiate the model *after* setting mixed precision policy if used
raft_model = RAFT(num_iterations=NUM_ITERATIONS)

# --- Checkpoint Setup ---
# Create trackable variables for epoch and global step
ckpt_epoch = tf.Variable(0, trainable=False, dtype=tf.int64, name='checkpoint_epoch')
# global_step will track total batches processed across all epochs
global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')

# Create the checkpoint object, including the new variables
ckpt = tf.train.Checkpoint(optimizer=optimizer,
                           model=raft_model,
                           epoch=ckpt_epoch,
                           step=global_step) # Add epoch and step

# Create the checkpoint manager
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=3)

# --- Restore Checkpoint ---
start_epoch = 0
if ckpt_manager.latest_checkpoint:
    print(f"Restoring checkpoint from {ckpt_manager.latest_checkpoint}...")
    # Restore state; this loads weights, optimizer state, epoch, and step
    status = ckpt.restore(ckpt_manager.latest_checkpoint)

    # Optional: Check if all variables were restored successfully
    try:
        status.assert_existing_objects_matched()
        print("Checkpoint restored successfully.")
    except AssertionError as e:
        print(f"Warning: Checkpoint restoration issue: {e}")
        print("Proceeding, but state might be incomplete.")

    # Get the epoch number to resume from
    start_epoch = ckpt_epoch.numpy()
    print(f"Resuming training from Epoch {start_epoch} (will start next epoch: {start_epoch + 1})")
    print(f"Global step restored to: {global_step.numpy()}")
else:
    print("No checkpoint found, initializing from scratch.")
    # Ensure global_step starts at 0 if not restoring
    global_step.assign(0)
    ckpt_epoch.assign(0)

# --- Build Model (if necessary) ---
# Build model by processing a dummy input batch (AFTER potential restore)
# This is important so layers are created before potential weight loading
print("Building model...")
dummy_img1 = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)
dummy_img2 = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)
try:
    _ = raft_model([dummy_img1, dummy_img2], training=False)
    print("Model built successfully.")
    # raft_model.summary() # Optional: Print summary
except Exception as e:
    print(f"Error during initial model build: {e}")
    # If restoration happened, the model might already be built.
    if not ckpt_manager.latest_checkpoint:
         raise # Re-raise if it failed on initial build


@tf.function # Compile training step for speed
def train_step(images, flow_gt):
    image1, image2 = images

    with tf.GradientTape() as tape:
        flow_predictions = raft_model([image1, image2], training=True)
        loss = raft_loss(flow_gt, flow_predictions)
        # scaled_loss = loss # Modify if using LossScaleOptimizer
        # --- MODIFICATION START ---
        # Scale the loss using the optimizer before computing gradients
        scaled_loss = optimizer.get_scaled_loss(loss) if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer) else loss
        # --- MODIFICATION END ---

    # Compute gradients using the scaled loss
    gradients = tape.gradient(scaled_loss, raft_model.trainable_variables)
    # --- MODIFICATION START ---
    # Unscale the gradients before clipping or applying them
    gradients = optimizer.get_unscaled_gradients(gradients) if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer) else gradients
    # --- MODIFICATION END ---
    gradients = [(tf.clip_by_norm(g, 1.0) if g is not None else None) for g in gradients]

    trainable_vars = raft_model.trainable_variables
    valid_grads_and_vars = [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]

    # Apply the unscaled gradients
    optimizer.apply_gradients(valid_grads_and_vars)

    # Calculate EPE on the final prediction for monitoring
    final_pred = tf.cast(flow_predictions[-1], tf.float32) if flow_predictions else None
    if final_pred is not None:
      final_epe = endpoint_error(flow_gt, final_pred)
    else:
      final_epe = tf.constant(0.0, dtype=tf.float32)

    # Return the original (unscaled) loss for logging
    return loss, final_epe


# --- Training Loop ---
print(f"Starting training from epoch {start_epoch + 1} up to {EPOCHS}...")
train_dataset = configure_dataset(DATA_DIR, is_training=True)

# Set up TensorBoard logging
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = os.path.join(LOG_DIR, current_time + f"_resume_from_epoch_{start_epoch}")
summary_writer = tf.summary.create_file_writer(train_log_dir)
print(f"Logging TensorBoard data to: {train_log_dir}")

# Calculate total batches if possible for tqdm
total_batches_per_epoch = tf.data.experimental.cardinality(train_dataset)
if total_batches_per_epoch == tf.data.experimental.UNKNOWN_CARDINALITY:
     print("Could not determine dataset size per epoch. Progress bar length may be inaccurate.")
     total_batches_per_epoch = None # Set to None if size unknown
else:
     total_batches_per_epoch = total_batches_per_epoch.numpy()

# --- Modified Epoch Loop ---
# Start from the epoch *after* the last saved one
for epoch in range(start_epoch, EPOCHS):
    print(f"\n--- Starting Epoch {epoch + 1}/{EPOCHS} ---")
    start_time = time.time()
    epoch_loss = 0.0
    epoch_epe = 0.0
    step_in_epoch = 0

    # Use tqdm for progress bar
    pbar = tqdm(train_dataset, total=total_batches_per_epoch, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

    for images, flow_gt in pbar:
        loss, final_epe = train_step(images, flow_gt)
        loss_np = loss.numpy()
        epe_np = final_epe.numpy()

        epoch_loss += loss_np
        epoch_epe += epe_np
        step_in_epoch += 1
        global_step.assign_add(1) # Increment the global step variable

        current_step = global_step.numpy() # Get current global step value

        # Update progress bar description with current stats and global step
        pbar.set_postfix({'Loss': f"{loss_np:.4f}", 'EPE': f"{epe_np:.4f}", 'Step': current_step})

        # Log to TensorBoard using the global step
        with summary_writer.as_default(step=current_step): # Set step context for summaries
            tf.summary.scalar('batch_loss', loss_np)
            tf.summary.scalar('batch_final_epe', epe_np)
            # Log learning rate
            if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                current_lr = optimizer.learning_rate(global_step) # Pass global_step
            else:
                current_lr = optimizer.learning_rate # Constant LR
            # Ensure current_lr is logged correctly
            tf.summary.scalar('learning_rate', current_lr if isinstance(current_lr, (float, int)) else current_lr.numpy())

        # --- Optional: Save checkpoint every N steps ---
        if SAVE_FREQ and SAVE_FREQ > 0 and current_step % SAVE_FREQ == 0:
            # Note: We don't update ckpt_epoch here, only at the end of the epoch
            save_path_step = ckpt_manager.save(checkpoint_number=current_step)
            print(f"\nSaved step checkpoint at global step {current_step}: {save_path_step}")

    # --- End of Epoch ---
    if step_in_epoch > 0:
      avg_epoch_loss = epoch_loss / step_in_epoch
      avg_epoch_epe = epoch_epe / step_in_epoch
    else:
      avg_epoch_loss = 0.0
      avg_epoch_epe = 0.0

    epoch_time = time.time() - start_time

    print(f"\nEpoch {epoch+1}/{EPOCHS} Summary | Time: {epoch_time:.2f}s | Avg Loss: {avg_epoch_loss:.4f} | Avg Final EPE: {avg_epoch_epe:.4f}")

    # Log epoch metrics to TensorBoard (use epoch number + 1 as step for epoch summaries)
    with summary_writer.as_default(step=epoch + 1):
        tf.summary.scalar('epoch_avg_loss', avg_epoch_loss)
        tf.summary.scalar('epoch_avg_final_epe', avg_epoch_epe)

    # --- Save Checkpoint at End of Epoch ---
    # Update the epoch variable *before* saving
    ckpt_epoch.assign(epoch + 1)
    # Save using the global step number for potential identification
    save_path_epoch = ckpt_manager.save(checkpoint_number=global_step.numpy())
    print(f"Saved epoch checkpoint for epoch {epoch+1} (Global Step {global_step.numpy()}): {save_path_epoch}")


print("\nTraining finished.")
summary_writer.close() # Close the summary writer

# --- 5. Model Saving ---
# ... (keep your final model saving code) ...
print("Saving final model artifacts...")
# Save weights (useful for quick loading)
weights_path = os.path.join(CHECKPOINT_DIR, 'raft_final_weights.weights.h5') # Keras 3 convention
raft_model.save_weights(weights_path)
print(f"Final weights saved to {weights_path}")

# Save as SavedModel format (for deployment/serving)
try:
    # Call the model on dummy data again right before saving if needed
    _ = raft_model([dummy_img1, dummy_img2], training=False)
    # Specify signatures if needed for TF Serving, but often works without for simple cases
    raft_model.save(SAVED_MODEL_DIR) # , signatures=...)
    print(f"Model saved in SavedModel format to {SAVED_MODEL_DIR}")
except Exception as e:
    print(f"Error saving SavedModel (might require specific input signatures or tracing): {e}")
    print("Saving weights was successful.")


# --- 4. Inference and Visualization ---
# ... (keep your inference and visualization code: make_color_wheel, flow_to_color, etc.) ...
# Helper function to visualize flow (Middlebury color code)
def make_color_wheel():
    """Generates a color wheel for flow visualization."""
    RY = 15; YG = 6; GC = 4; CB = 11; BM = 13; MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255; colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY)/RY)
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0, YG)/YG); colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255; colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0, GC)/GC)
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(0, CB)/CB); colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255; colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0, BM)/BM)
    col += BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(0, MR)/MR); colorwheel[col:col+MR, 0] = 255
    return colorwheel.astype(np.uint8) # Return as uint8

def flow_to_color(flow, convert_to_bgr=False):
    """Converts optical flow (u, v) to a color image using Middlebury color scheme."""
    # Input flow: (H, W, 2), numpy array, float32
    if flow is None:
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8) # Return black image if flow is None

    UNKNOWN_FLOW_THRESH = 1e7
    SMALL_FLOW = 1e-9 # Avoid division by zero

    # Check if flow has expected dimensions, handle case where it might be invalid shape
    if not isinstance(flow, np.ndarray) or flow.ndim != 3 or flow.shape[2] != 2:
        print(f"Warning: Invalid flow shape received in flow_to_color: {flow.shape if isinstance(flow, np.ndarray) else type(flow)}. Returning black image.")
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8) # Default size

    height, width, _ = flow.shape
    img = np.zeros((height, width, 3), dtype=np.uint8)
    colorwheel = make_color_wheel()
    ncols = colorwheel.shape[0]

    # Separate u and v components
    u, v = flow[..., 0], flow[..., 1]

    # Handle NaNs or Infs if any
    u = np.nan_to_num(u)
    v = np.nan_to_num(v)

    # Compute magnitude and angle
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(-v, -u) / np.pi # Range -1 to 1

    # Normalize angle to 0-1 range
    ang = (ang + 1.0) / 2.0

    # Normalize magnitude - compute max based on finite values, excluding unknowns
    valid_mag = mag[np.abs(mag) < UNKNOWN_FLOW_THRESH]
    mag_max = np.max(valid_mag) if valid_mag.size > 0 else 0.0

    if mag_max > SMALL_FLOW:
        mag_norm = np.clip(mag / mag_max, 0, 1) # Normalize 0-1
    else:
        mag_norm = np.zeros_like(mag)

    # Map angle to color wheel index
    fk = (ang * (ncols - 1)) # Map 0-1 angle to 0-(ncols-1) index
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0 # Interpolation factor 0-1

    # Interpolate colors
    for i in range(colorwheel.shape[1]): # R, G, B
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        # Increase saturation with magnitude (desaturate towards white)
        col = 1 - mag_norm * (1 - col)
        img[:, :, i] = np.floor(255.0 * col)

    # Handle unknown flow (mark as black)
    idx_unknown = (np.abs(u) > UNKNOWN_FLOW_THRESH) | \
                  (np.abs(v) > UNKNOWN_FLOW_THRESH) | \
                  (mag > mag_max * 1.1 if mag_max > SMALL_FLOW else False) # Also consider magnitude outliers if max_mag is valid
    img[idx_unknown] = 0

    if convert_to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def visualize_flow(image1_np, flow_pred_np, filename_prefix="flow_vis"):
    """Generates and saves flow visualizations."""
    if image1_np is None or flow_pred_np is None:
        print("Warning: Cannot visualize flow, input image or flow is None.")
        return

    # Ensure inputs are numpy
    image1_np = np.asarray(image1_np)
    flow_pred_np = np.asarray(flow_pred_np)

    # Clamp image display range to [0, 1] if it's float
    if image1_np.dtype == np.float32 or image1_np.dtype == np.float64:
        image1_np = np.clip(image1_np, 0, 1)

    # Ensure flow_pred_np has correct shape
    if flow_pred_np.ndim != 3 or flow_pred_np.shape[-1] != 2:
        print(f"Warning: Invalid flow shape for visualization: {flow_pred_np.shape}. Skipping visualization.")
        return
    h, w, _ = flow_pred_np.shape # Use flow shape for consistency
    img_h, img_w, _ = image1_np.shape

    # Resize image to match flow if necessary (shouldn't happen with current code but good practice)
    if img_h != h or img_w != w:
        print(f"Warning: Image shape ({img_h}x{img_w}) doesn't match flow shape ({h}x{w}). Resizing image for display.")
        image1_np = cv2.resize(image1_np, (w, h)) # cv2 uses (width, height)

    # 1. Flow Magnitude Heatmap
    try:
        plt.figure(figsize=(10, 8))
        magnitude = np.sqrt(np.sum(flow_pred_np**2, axis=-1))
        im = plt.imshow(magnitude, cmap='viridis') # Use viridis or jet
        plt.colorbar(im, label='Flow Magnitude (pixels)')
        plt.title(f'{os.path.basename(filename_prefix)} - Flow Magnitude')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_magnitude.png", bbox_inches='tight')
        plt.close()
        print(f"Saved flow magnitude heatmap to {filename_prefix}_magnitude.png")
    except Exception as e:
        print(f"Error generating magnitude plot: {e}")
        plt.close()

    # 2. Vector Field Overlay (Quiver Plot)
    try:
        step = max(1, min(h, w) // 32) # Adjust density based on image size
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow_pred_np[y, x].T

        plt.figure(figsize=(12, 9))
        plt.imshow(image1_np) # Show image 1 as background
        # Adjust quiver scale: smaller values -> longer arrows
        plt.quiver(x, y, fx, fy, color='red', scale=None, scale_units='xy', angles='xy', headwidth=5, headlength=6, width=0.0015)
        plt.title(f'{os.path.basename(filename_prefix)} - Flow Vectors (Overlay)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_vectors.png", bbox_inches='tight')
        plt.close()
        print(f"Saved flow vector field to {filename_prefix}_vectors.png")
    except Exception as e:
        print(f"Error generating vector plot: {e}")
        plt.close()

    # 3. Middlebury Color Visualization
    try:
        flow_color_img = flow_to_color(flow_pred_np)
        plt.figure(figsize=(10, 8))
        plt.imshow(flow_color_img)
        plt.title(f'{os.path.basename(filename_prefix)} - Flow (Color)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_color.png", bbox_inches='tight')
        plt.close()
        print(f"Saved color flow visualization to {filename_prefix}_color.png")
    except Exception as e:
        print(f"Error generating color flow plot: {e}")
        plt.close()

# --- Run Inference and Visualization ---
print("\nRunning inference and visualization...")

# Find frames in the specified visualization scene
scene_clean_dir = os.path.join(DATA_DIR, 'clean', VISUALIZATION_SCENE)
scene_frames = sorted(glob(os.path.join(scene_clean_dir, 'frame_*.png')), key=parse_frame_num)

if len(scene_frames) < 2:
    print(f"Error: Need at least two frames in scene '{VISUALIZATION_SCENE}' for visualization.")
else:
    # Select first two consecutive frames
    frame1_path = scene_frames[0]
    frame2_path = scene_frames[1]
    frame1_num = parse_frame_num(frame1_path)

    print(f"Visualizing flow between: {os.path.basename(frame1_path)} and {os.path.basename(frame2_path)}")

    # Load and preprocess the images for inference (use same functions as training)
    img1_inf = load_and_preprocess_image(frame1_path)
    img2_inf = load_and_preprocess_image(frame2_path)

    # Add batch dimension
    img1_inf_batch = tf.expand_dims(img1_inf, 0)
    img2_inf_batch = tf.expand_dims(img2_inf, 0)

    # Run inference using the trained model
    start_inf = time.time()
    # Ensure model is called in inference mode, get list of predictions
    predicted_flow_list = raft_model([img1_inf_batch, img2_inf_batch], training=False)
    inf_time = time.time() - start_inf
    print(f"Inference time: {inf_time:.3f}s")

    # --- Use the LAST prediction from the list ---
    if predicted_flow_list: # Check if the list is not empty
        predicted_flow_final = predicted_flow_list[-1]

        # Remove batch dimension and convert final prediction to numpy (float32)
        predicted_flow_np = predicted_flow_final[0].numpy()
        img1_np = img1_inf.numpy() # Get numpy version of first image for overlay

        # Visualize the PREDICTED flow
        vis_filename_prefix = f"vis_{VISUALIZATION_SCENE}_frame{frame1_num:04d}_PRED"
        visualize_flow(img1_np, predicted_flow_np, filename_prefix=vis_filename_prefix)
    else:
        print("Warning: Model did not return any flow predictions during inference.")


    # --- Optional: Visualize GROUND TRUTH flow with correct scaling ---
    flow_gt_path = os.path.join(DATA_DIR, 'flow', VISUALIZATION_SCENE, f"frame_{frame1_num:04d}.flo")
    if os.path.exists(flow_gt_path):
        print("Visualizing ground truth flow...")
        try:
            # Read GT flow and its original dimensions
            flow_gt_data, h_gt_orig, w_gt_orig = read_flo_file(tf.constant(flow_gt_path))

            # Resize the GT flow map using TF (bilinear)
            flow_gt_resized = tf.image.resize(flow_gt_data.astype(np.float32), [IMG_HEIGHT, IMG_WIDTH], method='bilinear')

            # Scale the GT flow vectors correctly
            scale_h_gt = tf.cast(IMG_HEIGHT, tf.float32) / tf.cast(h_gt_orig, tf.float32)
            scale_w_gt = tf.cast(IMG_WIDTH, tf.float32) / tf.cast(w_gt_orig, tf.float32)
            u_gt_scaled = flow_gt_resized[..., 0] * scale_w_gt
            v_gt_scaled = flow_gt_resized[..., 1] * scale_h_gt
            flow_gt_scaled_np = tf.stack([u_gt_scaled, v_gt_scaled], axis=-1).numpy() # Convert to numpy

            # Visualize the properly scaled GT flow
            vis_filename_prefix_gt = f"vis_{VISUALIZATION_SCENE}_frame{frame1_num:04d}_GT"
            # Ensure img1_np exists from the prediction visualization part
            if 'img1_np' in locals():
                visualize_flow(img1_np, flow_gt_scaled_np, filename_prefix=vis_filename_prefix_gt)
            else:
                print("Warning: Cannot visualize GT flow as img1_np is not available.")

        except Exception as e:
            print(f"Could not load or process ground truth flow {flow_gt_path}: {e}")
    else:
        print(f"Ground truth flow file not found: {flow_gt_path}")


print("\nRAFT Implementation Script Complete.")