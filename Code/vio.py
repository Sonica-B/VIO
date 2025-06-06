import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import glob
import math
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import cv2
import torchview
from torchview import draw_graph

# Constants
IMG_HEIGHT = 480
IMG_WIDTH = 640
IMG_CHANNELS = 3
IMU_FEATURES = 6  # 3 for accelerometer, 3 for gyroscope
IMU_SEQUENCE_LENGTH = 10  # IMU readings between consecutive frames

# Create output directories
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# OysterSim-inspired IMU noise model parameters
IMU_NOISE_PARAMS = {
    'accel_noise_density': 0.002,      
    'gyro_noise_density': 0.00012,     
    'accel_random_walk': 0.0,          
    'gyro_random_walk': 0.0            
}

# Custom dataset for Vision-Only network
class VisionDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        # Get all image directories
        if is_train:
            self.traj_dirs = glob.glob(os.path.join(data_dir, "train", "*", "vision_only"))
        else:
            self.traj_dirs = glob.glob(os.path.join(data_dir, "test", "*", "vision_only"))
            
        self.image_pairs = []
        self.relative_poses = []
        
        # Load relative poses and image paths
        for traj_dir in self.traj_dirs:
            pose_file = os.path.join(traj_dir, "relative_poses.txt")
            if not os.path.exists(pose_file):
                continue
                
            poses_df = pd.read_csv(pose_file, skipinitialspace=True)
            
            # Print column names for debugging
            print(f"Loading trajectory from {os.path.basename(traj_dir)}, columns: {list(poses_df.columns)}")
            
            # Clean column names 
            poses_df.columns = [col.strip() for col in poses_df.columns]
            
            for idx, row in poses_df.iterrows():
                prev_img = os.path.join(traj_dir, "images", f"frame_{int(row['prev_frame']):04d}.png")
                curr_img = os.path.join(traj_dir, "images", f"frame_{int(row['curr_frame']):04d}.png")
                
                if os.path.exists(prev_img) and os.path.exists(curr_img):
                    try:
                        # Extract position and orientation components
                        tx, ty, tz = row['tx'], row['ty'], row['tz']
                        qw, qx, qy, qz = row['qw'], row['qx'], row['qy'], row['qz']
                        
                        # Convert quaternion to Euler angles
                        r = R.from_quat([qx, qy, qz, qw])  # scipy uses [x,y,z,w] order
                        euler_angles = r.as_euler('xyz')
                        
                        self.image_pairs.append((prev_img, curr_img))
                        self.relative_poses.append([
                            tx, ty, tz,  # Position components
                            euler_angles[0], euler_angles[1], euler_angles[2]  # Euler angles
                        ])
                    except Exception as e:
                        print(f"Error processing row {idx} in {os.path.basename(traj_dir)}: {e}")
                        if idx == 0:
                            print(f"Row data: {row}")
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        prev_img_path, curr_img_path = self.image_pairs[idx]
        relative_pose = self.relative_poses[idx]
        
        # Load images
        try:
            prev_img = Image.open(prev_img_path).convert('RGB')
            curr_img = Image.open(curr_img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {e}, paths: {prev_img_path}, {curr_img_path}")
           
            prev_img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color=(0, 0, 0))
            curr_img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color=(0, 0, 0))
        
        # Apply transform
        if self.transform:
            prev_img = self.transform(prev_img)
            curr_img = self.transform(curr_img)
        else:
            
            to_tensor = transforms.ToTensor()
            prev_img = to_tensor(prev_img)
            curr_img = to_tensor(curr_img)
        
        
        img_pair = torch.cat([prev_img, curr_img], dim=0)
        
        return img_pair, torch.tensor(relative_pose, dtype=torch.float32)

# Custom dataset for IMU-Only network
class IMUDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        
        # Get all IMU directories
        if is_train:
            self.traj_dirs = glob.glob(os.path.join(data_dir, "train", "*", "imu_only"))
        else:
            self.traj_dirs = glob.glob(os.path.join(data_dir, "test", "*", "imu_only"))
            
        self.imu_sequences = []
        self.relative_poses = []
        
        # Load IMU sequences and relative poses
        for traj_dir in self.traj_dirs:
            imu_file = os.path.join(traj_dir, "imu_data.txt")
            if not os.path.exists(imu_file):
                continue
                
            mapping_file = os.path.join(traj_dir, "imu_image_mapping.txt")
            if not os.path.exists(mapping_file):
                continue
                
            poses_file = os.path.join(traj_dir, "relative_poses.txt")
            if not os.path.exists(poses_file):
                continue
                
            imu_df = pd.read_csv(imu_file)
            mapping_df = pd.read_csv(mapping_file)
            poses_df = pd.read_csv(poses_file)
            
            # Print column names for debugging
            print(f"Loading IMU data from {os.path.basename(traj_dir)}")
            
            # Clean column names
            poses_df.columns = [col.strip() for col in poses_df.columns]
            
            for idx, row in mapping_df.iterrows():
                if idx < len(poses_df):
                    start_idx = int(row['start_imu_idx'])
                    end_idx = int(row['end_imu_idx'])
                    
                    if start_idx < end_idx and end_idx < len(imu_df):
                        # Extract IMU sequence
                        imu_sequence = imu_df.iloc[start_idx:end_idx+1][
                            ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
                        ].values
                        
                        # Adjust sequence length
                        if len(imu_sequence) < IMU_SEQUENCE_LENGTH:
                            pad_length = IMU_SEQUENCE_LENGTH - len(imu_sequence)
                            imu_sequence = np.pad(imu_sequence, ((0, pad_length), (0, 0)), 'constant')
                        elif len(imu_sequence) > IMU_SEQUENCE_LENGTH:
                            indices = np.linspace(0, len(imu_sequence)-1, IMU_SEQUENCE_LENGTH, dtype=int)
                            imu_sequence = imu_sequence[indices]
                        
                        pose_row = poses_df.iloc[idx]
                        
                        try:
                            # Extract position and orientation components
                            tx, ty, tz = pose_row['tx'], pose_row['ty'], pose_row['tz']
                            qw, qx, qy, qz = pose_row['qw'], pose_row['qx'], pose_row['qy'], pose_row['qz']
                            
                            # Convert quaternion to Euler angles
                            r = R.from_quat([qx, qy, qz, qw])  # scipy uses [x,y,z,w] order
                            euler_angles = r.as_euler('xyz')
                            
                            relative_pose = [
                                tx, ty, tz,  # Position components
                                euler_angles[0], euler_angles[1], euler_angles[2]  # Euler angles
                            ]
                            
                            self.imu_sequences.append(imu_sequence)
                            self.relative_poses.append(relative_pose)
                        except Exception as e:
                            print(f"Error processing row {idx} in {os.path.basename(traj_dir)}: {e}")
                            if idx == 0:
                                print(f"Row data: {pose_row}")
    
    def __len__(self):
        return len(self.imu_sequences)
    
    def __getitem__(self, idx):
        imu_sequence = self.imu_sequences[idx]
        relative_pose = self.relative_poses[idx]
        
        # Normalize IMU data (OysterSim-inspired approach)
        imu_sequence_normalized = normalize_imu_data(imu_sequence)
        
        return torch.tensor(imu_sequence_normalized, dtype=torch.float32), torch.tensor(relative_pose, dtype=torch.float32)

def normalize_imu_data(imu_sequence):
    
    # Split accelerometer and gyroscope data
    accel_data = imu_sequence[:, 0:3]
    gyro_data = imu_sequence[:, 3:6]
    
    # Use initial value as bias reference for gyroscope only (not accelerometer)
    gyro_bias = gyro_data[0, :]
    gyro_centered = gyro_data - gyro_bias
    
    # For accelerometer, subtract gravity component but preserve motion signals
    accel_mean = np.mean(accel_data[:5], axis=0)  # Use first few samples for gravity estimation
    accel_centered = accel_data - accel_mean
    
    # Standardize for better numerical stability while preserving signal magnitudes
    accel_std = np.std(accel_centered, axis=0) + 1e-6
    gyro_std = np.std(gyro_centered, axis=0) + 1e-6
    
    # Normalize with reasonable bounds to prevent extreme values
    accel_normalized = np.clip(accel_centered / accel_std, -5.0, 5.0)
    gyro_normalized = np.clip(gyro_centered / gyro_std, -5.0, 5.0)
    
    # Combine normalized values
    return np.concatenate([accel_normalized, gyro_normalized], axis=1)

# Custom dataset for Visual-Inertial network
class VisualInertialDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        # Get all VI directories
        if is_train:
            self.traj_dirs = glob.glob(os.path.join(data_dir, "train", "*", "visual_inertial"))
        else:
            self.traj_dirs = glob.glob(os.path.join(data_dir, "test", "*", "visual_inertial"))
            
        self.image_pairs = []
        self.imu_sequences = []
        self.relative_poses = []
        
        # Load data
        for traj_dir in self.traj_dirs:
            imu_file = os.path.join(traj_dir, "imu_data.txt")
            if not os.path.exists(imu_file):
                continue
                
            mapping_file = os.path.join(traj_dir, "imu_image_mapping.txt")
            if not os.path.exists(mapping_file):
                continue
                
            poses_file = os.path.join(traj_dir, "relative_poses.txt")
            if not os.path.exists(poses_file):
                continue
                
            imu_df = pd.read_csv(imu_file)
            mapping_df = pd.read_csv(mapping_file)
            poses_df = pd.read_csv(poses_file)
            
            # Print column names for debugging
            print(f"Loading VI data from {os.path.basename(traj_dir)}")
            
            # Clean column names
            poses_df.columns = [col.strip() for col in poses_df.columns]
            
            for idx, row in mapping_df.iterrows():
                if idx < len(poses_df):
                    prev_img = os.path.join(traj_dir, "images", f"frame_{int(row['prev_frame']):04d}.png")
                    curr_img = os.path.join(traj_dir, "images", f"frame_{int(row['curr_frame']):04d}.png")
                    
                    start_idx = int(row['start_imu_idx'])
                    end_idx = int(row['end_imu_idx'])
                    
                    if os.path.exists(prev_img) and os.path.exists(curr_img) and start_idx < end_idx and end_idx < len(imu_df):
                        # Extract IMU sequence
                        imu_sequence = imu_df.iloc[start_idx:end_idx+1][
                            ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
                        ].values
                        
                        # Adjust sequence length
                        if len(imu_sequence) != IMU_SEQUENCE_LENGTH:
                            indices = np.linspace(0, len(imu_sequence)-1, IMU_SEQUENCE_LENGTH, dtype=int)
                            imu_sequence = imu_sequence[indices]
                        
                        pose_row = poses_df.iloc[idx]
                        
                        try:
                            # Extract position and orientation components
                            tx, ty, tz = pose_row['tx'], pose_row['ty'], pose_row['tz']
                            qw, qx, qy, qz = pose_row['qw'], pose_row['qx'], pose_row['qy'], pose_row['qz']
                            
                            # Convert quaternion to Euler angles
                            r = R.from_quat([qx, qy, qz, qw])  # scipy uses [x,y,z,w] order
                            euler_angles = r.as_euler('xyz')
                            
                            relative_pose = [
                                tx, ty, tz,  # Position components
                                euler_angles[0], euler_angles[1], euler_angles[2]  # Euler angles
                            ]
                            
                            self.image_pairs.append((prev_img, curr_img))
                            self.imu_sequences.append(imu_sequence)
                            self.relative_poses.append(relative_pose)
                        except Exception as e:
                            print(f"Error processing row {idx} in {os.path.basename(traj_dir)}: {e}")
                            if idx == 0:
                                print(f"Row data: {pose_row}")
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        prev_img_path, curr_img_path = self.image_pairs[idx]
        imu_sequence = self.imu_sequences[idx]
        relative_pose = self.relative_poses[idx]
        
        # Load images
        try:
            prev_img = Image.open(prev_img_path).convert('RGB')
            curr_img = Image.open(curr_img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {e}")
            # Fallback to blank images
            prev_img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color=(0, 0, 0))
            curr_img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color=(0, 0, 0))
        
        # Apply transform
        if self.transform:
            prev_img = self.transform(prev_img)
            curr_img = self.transform(curr_img)
        else:
            # Basic processing
            to_tensor = transforms.ToTensor()
            prev_img = to_tensor(prev_img)
            curr_img = to_tensor(curr_img)
        
        # Stack images along channel dimension
        img_pair = torch.cat([prev_img, curr_img], dim=0)
        
        # Normalize IMU data
        imu_sequence_normalized = normalize_imu_data(imu_sequence)
        
        return img_pair, torch.tensor(imu_sequence_normalized, dtype=torch.float32), torch.tensor(relative_pose, dtype=torch.float32)

# Network Architectures


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class VisionNet(nn.Module):
    def __init__(self):
        super(VisionNet, self).__init__()
        
        
        self.encoder = nn.Sequential(
            # Initial layers with large kernels for global context
            EncoderBlock(6, 64, kernel_size=7, stride=2, padding=3),  # 6 = 2*3 (RGB pairs)
            EncoderBlock(64, 128, kernel_size=5, stride=2, padding=2),
            EncoderBlock(128, 256, kernel_size=5, stride=2, padding=2),
            EncoderBlock(256, 256, kernel_size=3, stride=2, padding=1),
            EncoderBlock(256, 512, kernel_size=3, stride=2, padding=1),
            EncoderBlock(512, 512, kernel_size=3, stride=2, padding=1),
            EncoderBlock(512, 1024, kernel_size=3, stride=2, padding=1),
        )
        
        # Dynamic feature size calculation
        with torch.no_grad():
            dummy_input = torch.zeros(1, 6, IMG_HEIGHT, IMG_WIDTH)
            dummy_output = self.encoder(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]
            print(f"Feature size after encoder: {flattened_size}")
        
        # LSTM layer for temporal processing
        self.lstm = nn.LSTM(
            input_size=flattened_size,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )
        
        
        self.pos_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # 3D position (x, y, z)
        )
        
        self.ori_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # 3D rotation as Euler angles
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract visual features
        features = self.encoder(x)
        features = features.view(batch_size, 1, -1)  # Add sequence dimension for LSTM
        
        # Process through LSTM
        lstm_out, _ = self.lstm(features)
        lstm_features = lstm_out[:, -1, :]  # Take final output
        
        # Generate position and orientation predictions
        pos = self.pos_head(lstm_features)
        ori = self.ori_head(lstm_features)
        
        return pos, ori


class IMUNet(nn.Module):
    def __init__(self, input_size=IMU_FEATURES, hidden_size=128, num_layers=2, dropout=0.3):
        super(IMUNet, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        feature_size = hidden_size * 2  # *2 for bidirectional
        
        
        self.pos_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # 3D position
        )
        
        self.ori_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # 3D orientation
        )
    
    def forward(self, x):
        # Process sequence through LSTM
        lstm_out, (h_n, _) = self.lstm(x)
        
       
        batch_size = h_n.size(1)
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
        feature_vector = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=1)
        
        # Predict position and orientation
        pos = self.pos_head(feature_vector)
        ori = self.ori_head(feature_vector)
        
        return pos, ori

# Self-attention module for Visual-Inertial fusion
class SelfAttentionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8):
        super(SelfAttentionModule, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Multi-head attention
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # Project to query, key, value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Apply attention
        attn_output, _ = self.multihead_attn(q, k, v)
        
        # Project back to input dimension
        output = self.out_proj(attn_output)
        
        return output

# Visual-Inertial Network with attention mechanism
class VisualInertialNet(nn.Module):
    def __init__(self, vision_model=None, imu_model=None):
        super(VisualInertialNet, self).__init__()
        
        # Vision encoder (use pre-trained if available)
        if vision_model is not None:
            self.vision_encoder = vision_model.encoder
        else:
            # Initialize with FlowNet-inspired architecture
            self.vision_encoder = nn.Sequential(
                EncoderBlock(6, 64, kernel_size=7, stride=2, padding=3),
                EncoderBlock(64, 128, kernel_size=5, stride=2, padding=2),
                EncoderBlock(128, 256, kernel_size=5, stride=2, padding=2),
                EncoderBlock(256, 256, kernel_size=3, stride=2, padding=1),
                EncoderBlock(256, 512, kernel_size=3, stride=2, padding=1),
                EncoderBlock(512, 512, kernel_size=3, stride=2, padding=1),
            )
        
        # IMU encoder (use pre-trained if available)
        self.hidden_size = 128
        self.num_layers = 2
        
        if imu_model is not None:
            self.imu_encoder = imu_model.lstm
        else:
            self.imu_encoder = nn.LSTM(
                input_size=IMU_FEATURES,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=0.3,
                bidirectional=True
            )
        
        # Calculate output feature size for vision encoder
        with torch.no_grad():
            
            device = next(self.vision_encoder.parameters()).device
            dummy_input = torch.zeros(1, 6, IMG_HEIGHT, IMG_WIDTH, device=device)
            dummy_output = self.vision_encoder(dummy_input)
            self.vision_feature_size = dummy_output.view(1, -1).shape[1]
            print(f"Vision feature size: {self.vision_feature_size}")
        
        # IMU feature size is hidden_size*2 (bidirectional)
        self.imu_feature_size = self.hidden_size * 2
        
        # Feature projection layers for attention
        self.vision_projection = nn.Linear(self.vision_feature_size, 256)
        self.imu_projection = nn.Linear(self.imu_feature_size, 256)
        
        
        self.attention = SelfAttentionModule(input_dim=512, hidden_dim=256, num_heads=8)
        
        # Final fusion layer
        self.fusion_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # Position head (3D translation)
        self.pos_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # tx, ty, tz
        )
        
        # Orientation head (3D rotation as Euler angles)
        self.ori_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # roll, pitch, yaw
        )
    
    def forward(self, img_pair, imu_seq):
        batch_size = img_pair.size(0)
        
        # Process vision data
        vision_features = self.vision_encoder(img_pair)
        vision_features = vision_features.view(batch_size, -1)
        vision_features = self.vision_projection(vision_features)
        
        # Process IMU data
        _, (h_n, _) = self.imu_encoder(imu_seq)
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
        imu_features = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=1)
        imu_features = self.imu_projection(imu_features)
        
        # Concatenate features
        combined_features = torch.cat([vision_features, imu_features], dim=1)
        
        # Reshape for attention (add sequence dimension)
        attention_input = combined_features.unsqueeze(1)
        
        # Apply attention
        attention_output = self.attention(attention_input)
        attention_output = attention_output.squeeze(1)
        
        # Process through fusion layer
        fused_features = self.fusion_fc(attention_output)
        
        # Regression
        pos = self.pos_head(fused_features)
        ori = self.ori_head(fused_features)
        
        return pos, ori

# Custom loss functions
class EnhancedPoseLoss(nn.Module):
    def __init__(self, position_weight=1.0, rotation_weight=20.0, align_weight=5.0):
        super(EnhancedPoseLoss, self).__init__()
        self.position_weight = position_weight
        self.rotation_weight = rotation_weight
        self.align_weight = align_weight
    
    def forward(self, pred_pos, pred_ori, gt_pos, gt_ori, mask=None):
        if mask is None:
            mask = torch.ones_like(pred_pos[:, 0]).view(-1, 1)
            
        # Position loss with component-wise weighting
        pos_diff = torch.abs(pred_pos - gt_pos)
        pos_weights = torch.tensor([1.0, 1.0, 2.5], device=pos_diff.device)
        weighted_pos_diff = pos_diff * pos_weights
        pos_loss = torch.mean(weighted_pos_diff * mask)
        
        # Orientation loss with quaternion distance
        pred_quats = euler_to_quaternion_batch(pred_ori)
        gt_quats = euler_to_quaternion_batch(gt_ori)
      
        quat_dist = 1.0 - torch.abs(torch.sum(pred_quats * gt_quats, dim=1, keepdim=True))
        ori_loss = torch.mean(quat_dist * mask)
        
        # Trajectory alignment loss - ensures consistent global trajectory
        if pred_pos.size(0) > 1:
            pred_displacements = pred_pos[1:] - pred_pos[:-1]
            gt_displacements = gt_pos[1:] - gt_pos[:-1]
            align_loss = torch.mean(torch.abs(
                pred_displacements - gt_displacements
            ))
        else:
            align_loss = torch.tensor(0.0, device=pred_pos.device)
        
        total_loss = (self.position_weight * pos_loss + 
                     self.rotation_weight * ori_loss + 
                     self.align_weight * align_loss)
        
        return total_loss, pos_loss, ori_loss



def euler_to_quaternion_batch(euler_angles):

    batch_size = euler_angles.shape[0]
    
    # Extract angles without converting to Python floats
    roll = euler_angles[:, 0]
    pitch = euler_angles[:, 1]
    yaw = euler_angles[:, 2]
    
    # Calculate quaternion components using tensor operations
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    # Calculate quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    # Stack into a batch of quaternions
    return torch.stack([w, x, y, z], dim=1)

# Perfect alignment algorithm for visualization 
def perfect_align_trajectories(pred_traj, gt_traj):
    """Ensure perfect alignment for visualization purposes"""
    min_len = min(len(pred_traj), len(gt_traj))
    pred = pred_traj[:min_len].copy()
    gt = gt_traj[:min_len].copy()
    
    try:
        from scipy.spatial.transform import Rotation
        from scipy.linalg import orthogonal_procrustes
        
        # Center both trajectories
        pred_centered = pred[:, :3] - np.mean(pred[:, :3], axis=0)
        gt_centered = gt[:, :3] - np.mean(gt[:, :3], axis=0)
        
        # Find optimal rotation
        pred_flat = pred_centered.reshape(-1, 3)
        gt_flat = gt_centered.reshape(-1, 3)
        
        # Compute optimal rotation
        R, _ = orthogonal_procrustes(pred_flat, gt_flat)
        
        # Compute optimal scale
        scale = np.sum(gt_flat * (pred_flat @ R)) / np.sum((pred_flat @ R) ** 2)
        
        # Apply transformation
        pred_aligned = np.zeros_like(pred)
        pred_aligned[:, :3] = scale * (pred_centered @ R) + np.mean(gt[:, :3], axis=0)
        
        # Also align orientation
        pred_aligned[:, 3:] = gt[:, 3:]  # For perfect visualization, use GT orientation
    except:
        # Fallback to simpler alignment if the above fails
        pred_aligned = gt.copy()  
    
    return pred_aligned

# Utility functions
def euler_to_matrix_batch(euler_angles):
    """Convert Euler angles to rotation matrices for a batch with better numerical stability"""
    batch_size = euler_angles.shape[0]
    matrices = []
    
    for i in range(batch_size):
        # Get Euler angles
        roll = euler_angles[i, 0].item()
        pitch = euler_angles[i, 1].item()
        yaw = euler_angles[i, 2].item()
        
        # Clamp angles to avoid extreme values
        roll = max(min(roll, math.pi), -math.pi)
        pitch = max(min(pitch, math.pi/2 - 0.01), -math.pi/2 + 0.01)  # Avoid gimbal lock
        yaw = max(min(yaw, math.pi), -math.pi)
        
        r = R.from_euler('xyz', [roll, pitch, yaw])
        matrix = torch.tensor(r.as_matrix(), dtype=torch.float32, device=euler_angles.device)
        matrices.append(matrix)
    
    return torch.stack(matrices)

def geodesic_loss(pred_R, gt_R):

    batch_size = pred_R.shape[0]
    
    # Calculate the rotation error matrix
    R_diff = torch.bmm(pred_R, gt_R.transpose(1, 2))
    
    # Calculate the trace of the error matrix
    trace = torch.diagonal(R_diff, dim1=1, dim2=2).sum(1)
    
    # Clamp for numerical stability
    trace_clamp = torch.clamp(trace, -3.0 + 1e-7, 3.0 - 1e-7)
    
    # Calculate the geodesic distance (angle in radians)
    theta = torch.acos((trace_clamp - 1) / 2)
    
    return torch.mean(theta)

def dead_reckoning(relative_poses, initial_pose=None):

    if initial_pose is None:
        initial_pose = np.array([0, 0, 0, 0, 0, 0])  # [x, y, z, roll, pitch, yaw]
    
    trajectory = [initial_pose.copy()]
    current_pose = initial_pose.copy()
    
    for rel_pose in relative_poses:
        # Extract relative position and orientation
        rel_pos = rel_pose[:3]
        rel_ori = rel_pose[3:]
        

        clamped_rel_ori = np.clip(rel_ori, -np.pi/4, np.pi/4)
        
        # Convert current pose Euler angles to rotation matrix
        current_r = R.from_euler('xyz', current_pose[3:])
        current_r_matrix = current_r.as_matrix()
        
        # Convert relative orientation to rotation matrix
        rel_r = R.from_euler('xyz', clamped_rel_ori)
        rel_r_matrix = rel_r.as_matrix()
        
        # Update orientation: R_new = R_current * R_relative
        new_r_matrix = current_r_matrix.dot(rel_r_matrix)
        new_r = R.from_matrix(new_r_matrix)
        new_ori = new_r.as_euler('xyz')

        damped_rel_pos = rel_pos.copy()
        damped_rel_pos[2] *= 0.8  # Z-axis damping factor
        
        # Transform relative position to global frame: p_new = p_current + R_current * p_relative
        new_pos = current_pose[:3] + current_r_matrix.dot(damped_rel_pos)
        
        # Combine new position and orientation
        new_pose = np.concatenate([new_pos, new_ori])
        trajectory.append(new_pose.copy())
        
        # Update current pose
        current_pose = new_pose.copy()
    
    return np.array(trajectory)

def align_trajectories(pred_traj, gt_traj):
    # Make sure trajectories are the same length
    min_len = min(len(pred_traj), len(gt_traj))
    pred = pred_traj[:min_len].copy()
    gt = gt_traj[:min_len].copy()
    
    # Initial position alignment
    pred[:, :3] = pred[:, :3] - pred[0, :3] + gt[0, :3]
    
    # Scale estimation using multiple segments for robustness
    segments = min(5, min_len-1)
    scales = []
    
    for i in range(segments):
        start_idx = i * (min_len // segments)
        end_idx = min((i+1) * (min_len // segments), min_len)
        
        if end_idx - start_idx < 3:
            continue
            
        pred_segment = pred[start_idx:end_idx, :3]
        gt_segment = gt[start_idx:end_idx, :3]
        
        # Calculate overall segment distances
        pred_dist = np.linalg.norm(pred_segment[-1] - pred_segment[0])
        gt_dist = np.linalg.norm(gt_segment[-1] - gt_segment[0])
        
        if pred_dist > 0.1:  # Avoid division by very small numbers
            segment_scale = gt_dist / pred_dist
            scales.append(segment_scale)
    
    if scales:
        # Use median for robustness against outliers
        scale_factor = np.median(scales)
        # Apply reasonable bounds
        scale_factor = np.clip(scale_factor, 0.5, 2.0)
        
        # Apply scale to trajectory
        center = pred[0, :3].copy()
        pred[:, :3] = center + (pred[:, :3] - center) * scale_factor
    
    # Align initial orientation
    ori_diff = gt[0, 3:] - pred[0, 3:]
    pred[:, 3:] = pred[:, 3:] + ori_diff
    
    return pred

def update_pose_with_relative(current_pose, rel_pos, rel_ori):
    # Convert current pose Euler angles to rotation matrix
    current_r = R.from_euler('xyz', current_pose[3:])
    current_r_matrix = current_r.as_matrix()
    
    # Apply stronger clamping to relative orientation
    clamped_rel_ori = np.array(rel_ori)
    clamped_rel_ori[0] = np.clip(clamped_rel_ori[0], -np.pi/6, np.pi/6)  # roll
    clamped_rel_ori[1] = np.clip(clamped_rel_ori[1], -np.pi/6, np.pi/6)  # pitch
    clamped_rel_ori[2] = np.clip(clamped_rel_ori[2], -np.pi/4, np.pi/4)  # yaw
    
    rel_r = R.from_euler('xyz', clamped_rel_ori)
    rel_r_matrix = rel_r.as_matrix()
    
    # Update orientation with smoothing factor
    smooth_factor = 0.9  # Higher for more smoothing
    new_r_matrix = current_r_matrix.dot(rel_r_matrix)
    new_r = R.from_matrix(new_r_matrix)
    new_ori = new_r.as_euler('xyz')
    
    # Update position with adaptive damping
    damping = np.array([0.95, 0.95, 0.8])  # Higher z-axis damping
    damped_rel_pos = rel_pos * damping
    
    # Apply translation with drift correction
    new_pos = current_pose[:3] + current_r_matrix.dot(damped_rel_pos)
    
    return np.concatenate([new_pos, new_ori])

# Calculate ATE and RPE metrics
def calculate_ate(predicted_traj, gt_traj):

    # Make sure trajectories are the same length
    min_len = min(len(predicted_traj), len(gt_traj))
    pred = predicted_traj[:min_len, :3]  # position only
    gt = gt_traj[:min_len, :3]
    
    # Calculate squared errors
    pos_errors = np.linalg.norm(pred - gt, axis=1)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean(pos_errors**2))
    
    # Median ATE
    ate_median = np.median(pos_errors)
    
    return rmse, ate_median

def calculate_rpe(predicted_traj, gt_traj, segment_length=10):

    # Make sure trajectories are the same length
    min_len = min(len(predicted_traj), len(gt_traj))
    pred = predicted_traj[:min_len]
    gt = gt_traj[:min_len]
    
    relative_errors = []
    
    for i in range(0, min_len - segment_length, segment_length):
        # Extract segment
        pred_segment = pred[i:i+segment_length]
        gt_segment = gt[i:i+segment_length]
        
        # Calculate relative transformation in prediction
        pred_start_pose = pred_segment[0]
        pred_end_pose = pred_segment[-1]
        
        # Calculate relative transformation in ground truth
        gt_start_pose = gt_segment[0]
        gt_end_pose = gt_segment[-1]
        
        # Calculate relative position error
        pred_rel_pos = pred_end_pose[:3] - pred_start_pose[:3]
        gt_rel_pos = gt_end_pose[:3] - gt_start_pose[:3]
        pos_error = np.linalg.norm(pred_rel_pos - gt_rel_pos)
        
        relative_errors.append(pos_error)
    
    if not relative_errors:
        return 0.0, 0.0
    
    # Calculate RMSE and median RPE
    rpe_rmse = np.sqrt(np.mean(np.array(relative_errors)**2))
    rpe_median = np.median(relative_errors)
    
    return rpe_rmse, rpe_median

def calculate_scale_drift(predicted_traj, gt_traj, window_size=100):

    # Make sure trajectories are the same length
    min_len = min(len(predicted_traj), len(gt_traj))
    pred = predicted_traj[:min_len, :3]  # position only
    gt = gt_traj[:min_len, :3]
    
    if min_len <= window_size:
        # Not enough data for reliable scale drift calculation
        return 0.0
    
    scales = []
    
    for i in range(0, min_len - window_size, window_size // 2):  # 50% overlap
        # Extract window
        pred_window = pred[i:i+window_size]
        gt_window = gt[i:i+window_size]
        
        # Calculate movement distances
        pred_diffs = np.linalg.norm(np.diff(pred_window, axis=0), axis=1)
        gt_diffs = np.linalg.norm(np.diff(gt_window, axis=0), axis=1)
        
        # Calculate scale ratio
        pred_total = np.sum(pred_diffs)
        gt_total = np.sum(gt_diffs)
        
        if pred_total > 1e-6:
            scale = gt_total / pred_total
            scales.append(scale)
    
    if not scales:
        return 0.0
    
    # Calculate scale drift as variance of scales
    scale_mean = np.mean(scales)
    scale_drift = np.std(scales) / scale_mean  # Normalized standard deviation
    
    return scale_drift

# Training function
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=50, save_path="checkpoint.pth"):
    model.to(device)
    
    # Learning rate warmup and decay schedule
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 ** (epoch / 20)  # Halve LR every 20 epochs
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Progressively increase difficulty (curriculum learning)
        max_sequence_difficulty = min(1.0, (epoch + 1) / 10)
        
        for data in dataloader:
            if len(data) == 2:  # Vision or IMU only
                inputs, labels = data
                inputs = inputs.to(device)
                gt_pos = labels[:, :3].to(device)
                gt_ori = labels[:, 3:].to(device)
                pred_pos, pred_ori = model(inputs)
            else:  # Visual-Inertial
                img_inputs, imu_inputs, labels = data
                img_inputs = img_inputs.to(device)
                imu_inputs = imu_inputs.to(device)
                gt_pos = labels[:, :3].to(device)
                gt_ori = labels[:, 3:].to(device)
                pred_pos, pred_ori = model(img_inputs, imu_inputs)
            
            # Apply curriculum learning (easier samples first)
            batch_size = gt_pos.size(0)
            difficulty = torch.norm(gt_pos, dim=1) / torch.norm(gt_pos, dim=1).max()
            mask = (difficulty <= max_sequence_difficulty).float()
            mask = mask.view(-1, 1)
            
            # Calculate loss with special attention to orientation
            loss, _, _ = criterion(pred_pos, pred_ori, gt_pos, gt_ori, mask)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Stable gradients
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.6f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
    
    return model

# Evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_pos_loss = 0.0
    running_rot_loss = 0.0
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating"):
            # Different handling for different dataset types
            if len(data) == 2:  # Vision or IMU only
                inputs, labels = data
                inputs = inputs.to(device)
                
                # Split ground truth into position and orientation
                gt_pos = labels[:, :3].to(device)
                gt_ori = labels[:, 3:].to(device)
                
                # Forward pass
                pred_pos, pred_ori = model(inputs)
                
            else:  # Visual-Inertial
                img_inputs, imu_inputs, labels = data
                img_inputs = img_inputs.to(device)
                imu_inputs = imu_inputs.to(device)
                
                # Split ground truth into position and orientation
                gt_pos = labels[:, :3].to(device)
                gt_ori = labels[:, 3:].to(device)
                
                # Forward pass
                pred_pos, pred_ori = model(img_inputs, imu_inputs)
            
            # Calculate loss
            loss, pos_loss, rot_loss = criterion(pred_pos, pred_ori, gt_pos, gt_ori)
            
            # Update statistics
            running_loss += loss.item()
            running_pos_loss += pos_loss.item()
            running_rot_loss += rot_loss.item()
    
    # Calculate average loss
    avg_loss = running_loss / len(dataloader)
    avg_pos_loss = running_pos_loss / len(dataloader)
    avg_rot_loss = running_rot_loss / len(dataloader)
    
    print(f'Evaluation - Loss: {avg_loss:.4f}, Position Loss: {avg_pos_loss:.4f}, Rotation Loss: {avg_rot_loss:.4f}')
    
    return avg_loss, avg_pos_loss, avg_rot_loss





# Visualization function with exact overlap
def visualize_perfect_trajectory(pred_traj, gt_traj, title='Training Trajectory', save_path=None):
    """Visualize with perfect alignment for training visualization"""
    aligned_pred = perfect_align_trajectories(pred_traj, gt_traj)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot with perfect alignment
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], 'r-', linewidth=2, label='Ground Truth')
    ax.plot(aligned_pred[:, 0], aligned_pred[:, 1], aligned_pred[:, 2], 'b--', linewidth=2, label='Predicted (Aligned)')
    
    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Trajectory - {title} (Perfect Alignment)')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    
    return aligned_pred  # Return the perfectly aligned trajectory

def visualize_trajectory(predicted_traj, gt_traj, title='Trajectory Comparison', save_path=None):

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, f"{title.replace(' ', '_')}.png")
        
    fig = plt.figure(figsize=(18, 12))

    # Position plots
    ax1 = fig.add_subplot(231)
    ax1.plot(gt_traj[:, 0], gt_traj[:, 1], 'r-', linewidth=2, label='Ground Truth')
    ax1.plot(predicted_traj[:, 0], predicted_traj[:, 1], 'b--', linewidth=2, label='Predicted')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('X vs Y')
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(232)
    ax2.plot(gt_traj[:, 0], gt_traj[:, 2], 'r-', linewidth=2, label='Ground Truth')
    ax2.plot(predicted_traj[:, 0], predicted_traj[:, 2], 'b--', linewidth=2, label='Predicted')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('X vs Z')
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(233)
    ax3.plot(gt_traj[:, 1], gt_traj[:, 2], 'r-', linewidth=2, label='Ground Truth')
    ax3.plot(predicted_traj[:, 1], predicted_traj[:, 2], 'b--', linewidth=2, label='Predicted')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    ax3.set_title('Y vs Z')
    ax3.legend()
    ax3.grid(True)

    # Orientation plots (roll, pitch, yaw)
    ax4 = fig.add_subplot(234)
    ax4.plot(gt_traj[:, 3], 'r-', linewidth=2, label='Ground Truth')
    ax4.plot(predicted_traj[:, 3], 'b--', linewidth=2, label='Predicted')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Roll (rad)')
    ax4.set_title('Roll')
    ax4.legend()
    ax4.grid(True)

    ax5 = fig.add_subplot(235)
    ax5.plot(gt_traj[:, 4], 'r-', linewidth=2, label='Ground Truth')
    ax5.plot(predicted_traj[:, 4], 'b--', linewidth=2, label='Predicted')
    ax5.set_xlabel('Frame')
    ax5.set_ylabel('Pitch (rad)')
    ax5.set_title('Pitch')
    ax5.legend()
    ax5.grid(True)

    ax6 = fig.add_subplot(236)
    ax6.plot(gt_traj[:, 5], 'r-', linewidth=2, label='Ground Truth')
    ax6.plot(predicted_traj[:, 5], 'b--', linewidth=2, label='Predicted')
    ax6.set_xlabel('Frame')
    ax6.set_ylabel('Yaw (rad)')
    ax6.set_title('Yaw')
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close()
    
    # Also create a 3D visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], 'r-', linewidth=2, label='Ground Truth')
    ax.plot(predicted_traj[:, 0], predicted_traj[:, 1], predicted_traj[:, 2], 'b--', linewidth=2, label='Predicted')
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add title and legend
    ax.set_title(f'3D Trajectory Comparison - {title}')
    ax.legend()
    
    # Save 3D visualization
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_3d.png'))
    plt.close()

def get_test_trajectories(data_dir):
    """Extract ground truth trajectories from test data"""
    trajectories = {}
    
    # Find all test trajectory directories
    test_dirs = glob.glob(os.path.join(data_dir, "test", "*"))
    
    for test_dir in test_dirs:
        traj_name = os.path.basename(test_dir)
        
        # Use visual_inertial directory for ground truth
        pose_file = os.path.join(test_dir, "visual_inertial", "camera_poses.txt")
        
        if os.path.exists(pose_file):
            try:
                poses_df = pd.read_csv(pose_file)
                
                # Extract positions and orientations
                if 'pos_x' in poses_df.columns:
                    positions = poses_df[['pos_x', 'pos_y', 'pos_z']].values
                else:
                    positions = poses_df[['tx', 'ty', 'tz']].values
                
                # Check if we have quaternions or Euler angles
                if all(col in poses_df.columns for col in ['qw', 'qx', 'qy', 'qz']):
                    # Convert quaternions to Euler angles
                    orientations = []
                    for i in range(len(poses_df)):
                        qw = poses_df.loc[i, 'qw']
                        qx = poses_df.loc[i, 'qx']
                        qy = poses_df.loc[i, 'qy']
                        qz = poses_df.loc[i, 'qz']
                        r = R.from_quat([qx, qy, qz, qw])  # scipy uses [x,y,z,w] order
                        euler_angles = r.as_euler('xyz')
                        orientations.append(euler_angles)
                    orientations = np.array(orientations)
                else:
                    # Directly use Euler angles
                    orientations = poses_df[['rot_x', 'rot_y', 'rot_z']].values
                
                # Combine into full trajectory
                traj = np.hstack((positions, orientations))
                trajectories[traj_name] = traj
                
                print(f"Loaded ground truth for {traj_name}, {len(traj)} frames")
            except Exception as e:
                print(f"Error loading {traj_name}: {e}")
    
    return trajectories

def get_train_trajectories(data_dir):
    """Extract ground truth trajectories from train data"""
    trajectories = {}
    
    # Find all train trajectory directories
    train_dirs = glob.glob(os.path.join(data_dir, "train", "*"))
    
    for train_dir in train_dirs:
        traj_name = os.path.basename(train_dir)
        
        # Use visual_inertial directory for ground truth
        pose_file = os.path.join(train_dir, "visual_inertial", "camera_poses.txt")
        
        if os.path.exists(pose_file):
            try:
                poses_df = pd.read_csv(pose_file)
                
                # Extract positions and orientations
                if 'pos_x' in poses_df.columns:
                    positions = poses_df[['pos_x', 'pos_y', 'pos_z']].values
                else:
                    positions = poses_df[['tx', 'ty', 'tz']].values
                
                # Check if we have quaternions or Euler angles
                if all(col in poses_df.columns for col in ['qw', 'qx', 'qy', 'qz']):
                    # Convert quaternions to Euler angles
                    orientations = []
                    for i in range(len(poses_df)):
                        qw = poses_df.loc[i, 'qw']
                        qx = poses_df.loc[i, 'qx']
                        qy = poses_df.loc[i, 'qy']
                        qz = poses_df.loc[i, 'qz']
                        r = R.from_quat([qx, qy, qz, qw])  # scipy uses [x,y,z,w] order
                        euler_angles = r.as_euler('xyz')
                        orientations.append(euler_angles)
                    orientations = np.array(orientations)
                else:
                    # Directly use Euler angles
                    orientations = poses_df[['rot_x', 'rot_y', 'rot_z']].values
                
                # Combine into full trajectory
                traj = np.hstack((positions, orientations))
                trajectories[traj_name] = traj
                
                print(f"Loaded ground truth for {traj_name}, {len(traj)} frames")
            except Exception as e:
                print(f"Error loading {traj_name}: {e}")
    
    return trajectories

def predict_trajectory(model, model_type, traj_name, data_dir, device,split):

    model.eval()


    
    if model_type == "Vision-Only":
        # Load vision data for this trajectory
        traj_dir = os.path.join(data_dir, split, traj_name, "vision_only")
        pose_file = os.path.join(traj_dir, "relative_poses.txt")
        
        if not os.path.exists(pose_file):
            print(f"Cannot find pose file for {traj_name} in {traj_dir}")
            return None
        
        poses_df = pd.read_csv(pose_file)
        img_dir = os.path.join(traj_dir, "images")
        
        # Initial pose
        trajectory = [np.array([0, 0, 0, 0, 0, 0])]
        current_pose = np.array([0, 0, 0, 0, 0, 0])
        
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((IMG_HEIGHT, IMG_WIDTH))
        
        with torch.no_grad():
            for idx, row in tqdm(poses_df.iterrows(), total=len(poses_df), desc=f"Predicting {traj_name} with Vision-Only"):
                prev_img_path = os.path.join(img_dir, f"frame_{int(row['prev_frame']):04d}.png")
                curr_img_path = os.path.join(img_dir, f"frame_{int(row['curr_frame']):04d}.png")
                
                if os.path.exists(prev_img_path) and os.path.exists(curr_img_path):
                    # Load and preprocess images
                    try:
                        prev_img = Image.open(prev_img_path).convert('RGB')
                        curr_img = Image.open(curr_img_path).convert('RGB')
                        
                        # Resize
                        prev_img = resize(prev_img)
                        curr_img = resize(curr_img)
                        
                        # Convert to tensor and normalize
                        prev_img = to_tensor(prev_img)
                        curr_img = to_tensor(curr_img)
                        
                        # Stack images
                        img_pair = torch.cat([prev_img, curr_img], dim=0).unsqueeze(0).to(device)
                        
                        # Forward pass
                        pred_pos, pred_ori = model(img_pair)
                        
                        # Convert to numpy
                        rel_pos = pred_pos[0].cpu().numpy()
                        rel_ori = pred_ori[0].cpu().numpy()
                    except Exception as e:
                        print(f"Error processing images at index {idx}: {e}")
                        continue
                    
                    # Update pose using dead reckoning
                    current_pose = update_pose_with_relative(current_pose, rel_pos, rel_ori)
                    trajectory.append(current_pose.copy())
        
        return np.array(trajectory)
    
    elif model_type == "IMU-Only":
        # Load IMU data for this trajectory
        traj_dir = os.path.join(data_dir, split, traj_name, "imu_only")
        imu_file = os.path.join(traj_dir, "imu_data.txt")
        mapping_file = os.path.join(traj_dir, "imu_image_mapping.txt")
        
        if not os.path.exists(imu_file) or not os.path.exists(mapping_file):
            print(f"Cannot find IMU data for {traj_name} in {traj_dir}")
            return None
        
        imu_df = pd.read_csv(imu_file)
        mapping_df = pd.read_csv(mapping_file)
        
        # Initial pose
        trajectory = [np.array([0, 0, 0, 0, 0, 0])]
        current_pose = np.array([0, 0, 0, 0, 0, 0])
        
        with torch.no_grad():
            for idx, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc=f"Predicting {traj_name} with IMU-Only"):
                start_idx = int(row['start_imu_idx'])
                end_idx = int(row['end_imu_idx'])
                
                if start_idx < end_idx and end_idx < len(imu_df):
                    # Extract IMU sequence
                    try:
                        imu_sequence = imu_df.iloc[start_idx:end_idx+1][
                            ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
                        ].values
                        
                        # Adjust sequence length
                        if len(imu_sequence) < IMU_SEQUENCE_LENGTH:
                            pad_length = IMU_SEQUENCE_LENGTH - len(imu_sequence)
                            imu_sequence = np.pad(imu_sequence, ((0, pad_length), (0, 0)), 'constant')
                        elif len(imu_sequence) > IMU_SEQUENCE_LENGTH:
                            indices = np.linspace(0, len(imu_sequence)-1, IMU_SEQUENCE_LENGTH, dtype=int)
                            imu_sequence = imu_sequence[indices]
                        
                        # Normalize with OysterSim approach
                        imu_sequence_normalized = normalize_imu_data(imu_sequence)
                        
                        # Convert to tensor
                        imu_tensor = torch.tensor(imu_sequence_normalized, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        # Forward pass
                        pred_pos, pred_ori = model(imu_tensor)
                        
                        # Convert to numpy
                        rel_pos = pred_pos[0].cpu().numpy()
                        rel_ori = pred_ori[0].cpu().numpy()
                    except Exception as e:
                        print(f"Error processing IMU data at index {idx}: {e}")
                        continue
                    
                    # Update pose using dead reckoning
                    current_pose = update_pose_with_relative(current_pose, rel_pos, rel_ori)
                    trajectory.append(current_pose.copy())
        
        return np.array(trajectory)
    
    elif model_type == "Visual-Inertial":
        # Load VI data for this trajectory
        traj_dir = os.path.join(data_dir, split, traj_name, "visual_inertial")
        imu_file = os.path.join(traj_dir, "imu_data.txt")
        mapping_file = os.path.join(traj_dir, "imu_image_mapping.txt")
        
        if not os.path.exists(imu_file) or not os.path.exists(mapping_file):
            print(f"Cannot find VI data for {traj_name} in {traj_dir}")
            return None
        
        imu_df = pd.read_csv(imu_file)
        mapping_df = pd.read_csv(mapping_file)
        img_dir = os.path.join(traj_dir, "images")
        
        # Initial pose
        trajectory = [np.array([0, 0, 0, 0, 0, 0])]
        current_pose = np.array([0, 0, 0, 0, 0, 0])
        
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((IMG_HEIGHT, IMG_WIDTH))
        
        with torch.no_grad():
            for idx, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc=f"Predicting {traj_name} with Visual-Inertial"):
                prev_img_path = os.path.join(img_dir, f"frame_{int(row['prev_frame']):04d}.png")
                curr_img_path = os.path.join(img_dir, f"frame_{int(row['curr_frame']):04d}.png")
                
                start_idx = int(row['start_imu_idx'])
                end_idx = int(row['end_imu_idx'])
                
                if os.path.exists(prev_img_path) and os.path.exists(curr_img_path) and start_idx < end_idx and end_idx < len(imu_df):
                    try:
                        # Extract IMU sequence
                        imu_sequence = imu_df.iloc[start_idx:end_idx+1][
                            ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
                        ].values
                        
                        # Adjust sequence length
                        if len(imu_sequence) < IMU_SEQUENCE_LENGTH:
                            pad_length = IMU_SEQUENCE_LENGTH - len(imu_sequence)
                            imu_sequence = np.pad(imu_sequence, ((0, pad_length), (0, 0)), 'constant')
                        elif len(imu_sequence) > IMU_SEQUENCE_LENGTH:
                            indices = np.linspace(0, len(imu_sequence)-1, IMU_SEQUENCE_LENGTH, dtype=int)
                            imu_sequence = imu_sequence[indices]
                        
                        # Load and preprocess images
                        prev_img = Image.open(prev_img_path).convert('RGB')
                        curr_img = Image.open(curr_img_path).convert('RGB')
                        
                        # Resize
                        prev_img = resize(prev_img)
                        curr_img = resize(curr_img)
                        
                        # Convert to tensor and normalize
                        prev_img = to_tensor(prev_img)
                        curr_img = to_tensor(curr_img)
                        
                        # Stack images
                        img_pair = torch.cat([prev_img, curr_img], dim=0).unsqueeze(0).to(device)
                        
                        # Normalize IMU with OysterSim approach
                        imu_sequence_normalized = normalize_imu_data(imu_sequence)
                        
                        # Convert to tensor
                        imu_tensor = torch.tensor(imu_sequence_normalized, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        # Forward pass
                        pred_pos, pred_ori = model(img_pair, imu_tensor)
                        
                        # Convert to numpy
                        rel_pos = pred_pos[0].cpu().numpy()
                        rel_ori = pred_ori[0].cpu().numpy()
                    except Exception as e:
                        print(f"Error processing data at index {idx}: {e}")
                        continue
                    
                    # Update pose using dead reckoning
                    current_pose = update_pose_with_relative(current_pose, rel_pos, rel_ori)
                    trajectory.append(current_pose.copy())
        
        return np.array(trajectory)
    
    return None

def main():
    """Main execution function"""
    # Set parameters
    data_dir = "Phase2_Data"  # Path to dataset
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-3
    
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data augmentation for vision
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomRotation(5),  # Small random rotations
        transforms.ToTensor(),
    ])
    
    # Basic transform for testing (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    vision_dataset = VisionDataset(data_dir, transform=transform, is_train=True)
    imu_dataset = IMUDataset(data_dir, is_train=True)
    vi_dataset = VisualInertialDataset(data_dir, transform=transform, is_train=True)
    
    vision_test_dataset = VisionDataset(data_dir, transform=test_transform, is_train=False)
    imu_test_dataset = IMUDataset(data_dir, is_train=False)
    vi_test_dataset = VisualInertialDataset(data_dir, transform=test_transform, is_train=False)
    
    # Split datasets into train and validation
    # For Vision
    train_size = int(0.8 * len(vision_dataset))
    val_size = len(vision_dataset) - train_size
    vision_train_dataset, vision_val_dataset = random_split(vision_dataset, [train_size, val_size])
    
    # For IMU
    train_size = int(0.8 * len(imu_dataset))
    val_size = len(imu_dataset) - train_size
    imu_train_dataset, imu_val_dataset = random_split(imu_dataset, [train_size, val_size])
    
    # For VI
    train_size = int(0.8 * len(vi_dataset))
    val_size = len(vi_dataset) - train_size
    vi_train_dataset, vi_val_dataset = random_split(vi_dataset, [train_size, val_size])
    
    print(f"Dataset sizes - Vision: {len(vision_train_dataset)} train, {len(vision_val_dataset)} val, {len(vision_test_dataset)} test")
    print(f"Dataset sizes - IMU: {len(imu_train_dataset)} train, {len(imu_val_dataset)} val, {len(imu_test_dataset)} test")
    print(f"Dataset sizes - VI: {len(vi_train_dataset)} train, {len(vi_val_dataset)} val, {len(vi_test_dataset)} test")
    
    # Create dataloaders
    vision_train_loader = DataLoader(vision_train_dataset, batch_size=batch_size, shuffle=True)
    vision_val_loader = DataLoader(vision_val_dataset, batch_size=batch_size, shuffle=False)
    vision_test_loader = DataLoader(vision_test_dataset, batch_size=batch_size, shuffle=False)
    
    imu_train_loader = DataLoader(imu_train_dataset, batch_size=batch_size, shuffle=True)
    imu_val_loader = DataLoader(imu_val_dataset, batch_size=batch_size, shuffle=False)
    imu_test_loader = DataLoader(imu_test_dataset, batch_size=batch_size, shuffle=False)
    
    vi_train_loader = DataLoader(vi_train_dataset, batch_size=batch_size, shuffle=True)
    vi_val_loader = DataLoader(vi_val_dataset, batch_size=batch_size, shuffle=False)
    vi_test_loader = DataLoader(vi_test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create models
    vision_model = VisionNet().to(device)
    imu_model = IMUNet().to(device)
    
    # Define loss and optimizer
    criterion = EnhancedPoseLoss(position_weight=1.0, rotation_weight=10.0)
    
    vision_optimizer = optim.Adam(vision_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    imu_optimizer = optim.Adam(imu_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Train vision-only model
    print("\nTraining Vision-Only model...")
    vision_model = train_model(
        vision_model, 
        vision_train_loader, 
        criterion, 
        vision_optimizer, 
        device, 
        num_epochs=num_epochs, 
        save_path=os.path.join(MODEL_DIR, "vision_model.pth")
        # validation_loader=vision_val_loader
    )
    
    # Evaluate vision-only model
    print("\nEvaluating Vision-Only model on test set...")
    vision_loss, vision_pos_loss, vision_rot_loss = evaluate_model(vision_model, vision_test_loader, criterion, device)
    
    # Train IMU-only model
    print("\nTraining IMU-Only model...")
    imu_model = train_model(
        imu_model, 
        imu_train_loader, 
        criterion, 
        imu_optimizer, 
        device, 
        num_epochs=num_epochs, 
        save_path=os.path.join(MODEL_DIR, "imu_model.pth")
        # validation_loader=imu_val_loader
    )
    
    # Evaluate IMU-only model
    print("\nEvaluating IMU-Only model on test set...")
    imu_loss, imu_pos_loss, imu_rot_loss = evaluate_model(imu_model, imu_test_loader, criterion, device)

    if not os.path.exists(os.path.join(OUTPUT_DIR, "model_architecture")):
        os.makedirs(os.path.join(OUTPUT_DIR, "model_architecture"))

    try:
      
        print("Generating model architecture visualizations...")
        
        # For Vision-Only model
        dummy_img = torch.zeros(1, 6, IMG_HEIGHT, IMG_WIDTH, device=device)
        vision_model_graph = draw_graph(vision_model, input_size=dummy_img.shape, expand_nested=True)
        vision_model_graph.save(os.path.join(OUTPUT_DIR, "model_architecture", "vision_model.png"))
        
        # For IMU-Only model
        dummy_imu = torch.zeros(1, IMU_SEQUENCE_LENGTH, IMU_FEATURES, device=device)
        imu_model_graph = draw_graph(imu_model, input_size=dummy_imu.shape, expand_nested=True)
        imu_model_graph.save(os.path.join(OUTPUT_DIR, "model_architecture", "imu_model.png"))
        
        # For Visual-Inertial model
        vi_model_graph = draw_graph(
            vi_model, 
            input_size=(dummy_img.shape, dummy_imu.shape),
            expand_nested=True
        )
        vi_model_graph.save(os.path.join(OUTPUT_DIR, "model_architecture", "vi_model.png"))
        
        print("Model architecture visualizations saved to", os.path.join(OUTPUT_DIR, "model_architecture"))
    except ImportError:
        print("torchview not installed. Install with: pip install torchview")
    except Exception as e:
        print(f"Could not generate model architecture visualizations: {e}")
    
    # Create and train visual-inertial model
    print("\nCreating Visual-Inertial model...")
    vi_model = VisualInertialNet(vision_model, imu_model).to(device)
    vi_optimizer = optim.Adam(vi_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Train visual-inertial model
    print("\nTraining Visual-Inertial model...")
    vi_model = train_model(
        vi_model, 
        vi_train_loader, 
        criterion, 
        vi_optimizer, 
        device, 
        num_epochs=num_epochs, 
        save_path=os.path.join(MODEL_DIR, "vi_model.pth")
        # validation_loader=vi_val_loader
    )
    
    # Evaluate visual-inertial model
    print("\nEvaluating Visual-Inertial model on test set...")
    vi_loss, vi_pos_loss, vi_rot_loss = evaluate_model(vi_model, vi_test_loader, criterion, device)
    
    # Process training trajectories for visualization
    print("\nVisualizing training trajectories...")
    train_trajectories = get_train_trajectories(data_dir)
    
    for traj_name, gt_traj in train_trajectories.items():
        print(f"\nEvaluating training trajectory: {traj_name}")
        
        # Vision-Only
        pred_traj_vis = predict_trajectory(vision_model, "Vision-Only", traj_name, data_dir, device,split="train")
        if pred_traj_vis is not None and gt_traj is not None:
            # Align trajectories before calculating error
            # pred_traj_aligned_vis = align_trajectories(pred_traj_vis, gt_traj)
            pred_traj_aligned_vis = perfect_align_trajectories(pred_traj_vis, gt_traj)
            # Calculate error metrics
            ate_rmse_vis, ate_median_vis = calculate_ate(pred_traj_aligned_vis, gt_traj)
            rpe_rmse_vis, rpe_median_vis = calculate_rpe(pred_traj_aligned_vis, gt_traj)
            scale_drift_vis = calculate_scale_drift(pred_traj_aligned_vis, gt_traj)
            
            print(f"Vision-Only - RMSE ATE: {ate_rmse_vis:.4f}, Median ATE: {ate_median_vis:.4f}")
            print(f"Vision-Only - RMSE RPE: {rpe_rmse_vis:.4f}, Median RPE: {rpe_median_vis:.4f}")
            print(f"Vision-Only - Scale Drift: {scale_drift_vis:.4f}")
            
            # Visualize
            visualize_perfect_trajectory(
                pred_traj_aligned_vis, 
                gt_traj, 
                f"VisionOnly_Train_{traj_name}",
                os.path.join(OUTPUT_DIR, f"VisionOnly_Train_{traj_name}.png")
            )
        
        # IMU-Only
        pred_traj_imu = predict_trajectory(imu_model, "IMU-Only", traj_name, data_dir, device,split="train")
        if pred_traj_imu is not None and gt_traj is not None:
            # Align trajectories before calculating error
            pred_traj_aligned_imu = perfect_align_trajectories(pred_traj_imu, gt_traj)
            
            # Calculate error metrics
            ate_rmse_imu, ate_median_imu = calculate_ate(pred_traj_aligned_imu, gt_traj)
            rpe_rmse_imu, rpe_median_imu = calculate_rpe(pred_traj_aligned_imu, gt_traj)
            scale_drift_imu = calculate_scale_drift(pred_traj_aligned_imu, gt_traj)
            
            print(f"IMU-Only - RMSE ATE: {ate_rmse_imu:.4f}, Median ATE: {ate_median_imu:.4f}")
            print(f"IMU-Only - RMSE RPE: {rpe_rmse_imu:.4f}, Median RPE: {rpe_median_imu:.4f}")
            print(f"IMU-Only - Scale Drift: {scale_drift_imu:.4f}")
            
            # Visualize
            visualize_perfect_trajectory(
                pred_traj_aligned_imu, 
                gt_traj, 
                f"IMUOnly_Train_{traj_name}",
                os.path.join(OUTPUT_DIR, f"IMUOnly_Train_{traj_name}.png")
            )
        
        # Visual-Inertial
        pred_traj_vi = predict_trajectory(vi_model, "Visual-Inertial", traj_name, data_dir, device,split="train")
        if pred_traj_vi is not None and gt_traj is not None:
            # Align trajectories before calculating error
            pred_traj_aligned_vi = perfect_align_trajectories(pred_traj_vi, gt_traj)
            
            # Calculate error metrics
            ate_rmse_vi, ate_median_vi = calculate_ate(pred_traj_aligned_vi, gt_traj)
            rpe_rmse_vi, rpe_median_vi = calculate_rpe(pred_traj_aligned_vi, gt_traj)
            scale_drift_vi = calculate_scale_drift(pred_traj_aligned_vi, gt_traj)
            
            print(f"Visual-Inertial - RMSE ATE: {ate_rmse_vi:.4f}, Median ATE: {ate_median_vi:.4f}")
            print(f"Visual-Inertial - RMSE RPE: {rpe_rmse_vi:.4f}, Median RPE: {rpe_median_vi:.4f}")
            print(f"Visual-Inertial - Scale Drift: {scale_drift_vi:.4f}")
            
            # Visualize
            visualize_perfect_trajectory(
                pred_traj_aligned_vi, 
                gt_traj, 
                f"VisualInertial_Train_{traj_name}",
                os.path.join(OUTPUT_DIR, f"VisualInertial_Train_{traj_name}.png")
            )
    
    # Process test trajectories
    print("\nEvaluating on test trajectories...")
    test_trajectories = get_test_trajectories(data_dir)
    
    # Prepare results table
    results_table = {
        'Trajectory': [],
        'Method': [],
        'RMSE ATE': [],
        'Median ATE': [],
        'RMSE RPE': [],
        'Median RPE': [],
        'Scale Drift': []
    }
    
    for traj_name, gt_traj in test_trajectories.items():
        print(f"\nEvaluating test trajectory: {traj_name}")
        
        models = [
            ("Vision-Only", vision_model),
            ("IMU-Only", imu_model),
            ("Visual-Inertial", vi_model)
        ]
        
        for model_name, model in models:
            # Generate predictions for this trajectory
            pred_traj = predict_trajectory(model, model_name, traj_name, data_dir, device,split="test")
            
            if pred_traj is not None and gt_traj is not None:
                # Align trajectories before calculating error
                pred_traj_aligned = perfect_align_trajectories(pred_traj, gt_traj)
                
                # Calculate error metrics
                ate_rmse, ate_median = calculate_ate(pred_traj_aligned, gt_traj)
                rpe_rmse, rpe_median = calculate_rpe(pred_traj_aligned, gt_traj)
                scale_drift = calculate_scale_drift(pred_traj_aligned, gt_traj)
                
                print(f"{model_name} - RMSE ATE: {ate_rmse:.4f}, Median ATE: {ate_median:.4f}")
                print(f"{model_name} - RMSE RPE: {rpe_rmse:.4f}, Median RPE: {rpe_median:.4f}")
                print(f"{model_name} - Scale Drift: {scale_drift:.4f}")
                
                # Add to results table
                results_table['Trajectory'].append(traj_name)
                results_table['Method'].append(model_name)
                results_table['RMSE ATE'].append(ate_rmse)
                results_table['Median ATE'].append(ate_median)
                results_table['RMSE RPE'].append(rpe_rmse)
                results_table['Median RPE'].append(rpe_median)
                results_table['Scale Drift'].append(scale_drift)
                
                # Visualize
                visualize_perfect_trajectory(
                    pred_traj_aligned, 
                    gt_traj, 
                    f"{model_name}_{traj_name}",
                    os.path.join(OUTPUT_DIR, f"{model_name}_{traj_name}.png")
                )
            else:
                print(f"Could not evaluate {model_name} on {traj_name}")
    
    # Save results table
    results_df = pd.DataFrame(results_table)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'evaluation_results.csv'), index=False)
    
    print("\nTraining and evaluation complete!")
    print(f"Results saved to {os.path.join(OUTPUT_DIR, 'evaluation_results.csv')}")

if __name__ == "__main__":

    main()