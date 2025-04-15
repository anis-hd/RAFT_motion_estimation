# convert_weights.py
# PURPOSE: Convert Keras H5 weights to TensorFlow Checkpoint format.
# RUN THIS SCRIPT IN THE ORIGINAL TF 2.10.1 ENVIRONMENT (Windows)

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os

print(f"Using TensorFlow version: {tf.__version__}")
if not tf.__version__.startswith('2.10.'):
    print("WARNING: This script is intended for TensorFlow 2.10.x.")
    print("Ensure you are running this in the environment where the H5 weights were created.")

# --- Configuration (MUST MATCH train.py used to save the H5 weights) ---
IMG_HEIGHT = 384       # <-- VERIFY this matches your train.py!
IMG_WIDTH = 512        # <-- VERIFY this matches your train.py!
NUM_ITERATIONS = 8     # <-- VERIFY this matches your train.py!
CHECKPOINT_DIR = './raft_checkpoints' # Directory containing the H5 file
H5_WEIGHTS_FILENAME = 'raft_final_weights.weights.h5' # Input H5 file
TF_CHECKPOINT_PREFIX = 'raft_tf_checkpoint' # Output prefix for TF format

# --- 1. Re-define Model Architecture (Copied directly from your train.py) ---
# Ensure this is identical to the model used when saving H5 weights.

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

# Helper for Correlation
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

# RAFT Model
class RAFT(tf.keras.Model):
    def __init__(self, img_height, img_width, num_iterations=NUM_ITERATIONS, hidden_dim=128, context_dim=128, corr_levels=1, corr_radius=4, name='raft', **kwargs):
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

# --- End of Model Definitions ---


# --- 2. Conversion Logic ---
if __name__ == "__main__":
    print("--- Starting Weight Conversion ---")

    # Construct paths
    h5_weights_path = os.path.join(CHECKPOINT_DIR, H5_WEIGHTS_FILENAME)
    tf_checkpoint_path_prefix = os.path.join(CHECKPOINT_DIR, TF_CHECKPOINT_PREFIX)

    # Check if input H5 file exists
    if not os.path.exists(h5_weights_path):
        print(f"ERROR: Input H5 weights file not found at: {h5_weights_path}")
        exit()

    print(f"Input H5 weights: {h5_weights_path}")
    print(f"Output TF Checkpoint prefix: {tf_checkpoint_path_prefix}")
    print(f"Using Model Config: H={IMG_HEIGHT}, W={IMG_WIDTH}, Iterations={NUM_ITERATIONS}")

    # Instantiate the model
    print("Instantiating RAFT model...")
    try:
        model = RAFT(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_iterations=NUM_ITERATIONS)
    except Exception as e:
        print(f"ERROR: Failed to instantiate RAFT model: {e}")
        exit()
    print("Model instantiated.")

    # Build the model (essential before loading weights)
    print("Building model with dummy data...")
    try:
        # Use float32 for dummy build, should be safe regardless of mixed precision policy
        dummy_img1 = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)
        dummy_img2 = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)
        _ = model([dummy_img1, dummy_img2], training=False)
        print("Model built successfully.")
    except Exception as e:
        print(f"ERROR: Failed during model build: {e}")
        # If build fails, loading weights will definitely fail.
        exit()

    # Load the weights from the H5 file
    print(f"Loading weights from H5 file: {h5_weights_path}...")
    try:
        status = model.load_weights(h5_weights_path)
        # Optional: More strict check if weights loaded correctly
        # status.assert_consumed() # Can uncomment for stricter check, might error if optimizer state isn't saved/loaded
        print("H5 weights loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load weights from {h5_weights_path}: {e}")
        print("Check if IMG_HEIGHT, IMG_WIDTH, NUM_ITERATIONS match the training script.")
        exit()

    # Save the weights in TensorFlow Checkpoint format
    print(f"Saving weights in TensorFlow Checkpoint format to prefix: {tf_checkpoint_path_prefix}...")
    try:
        model.save_weights(tf_checkpoint_path_prefix) # Pass the prefix, NOT with .h5
        print("TensorFlow Checkpoint weights saved successfully.")
        print(f"Look for files starting with '{TF_CHECKPOINT_PREFIX}' in '{CHECKPOINT_DIR}'.")
        print("Files to transfer to WSL2: checkpoint, .index, .data-xxxxx-of-xxxxx")
    except Exception as e:
        print(f"ERROR: Failed to save weights in TensorFlow Checkpoint format: {e}")
        exit()

    print("--- Weight Conversion Complete ---")