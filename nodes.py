import os
import sys
import json
import subprocess
import datetime
import base64
import requests
import numpy as np
import re
import torch
import torch.nn.functional as F
import wave
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
from pathlib import Path
from string import Template
import itertools
import math
import time
import threading

# Import yt-dlp for YouTube downloads
try:
    import yt_dlp
    HAS_YTDLP = True
except ImportError:
    print(f"[OshifiedNode] yt-dlp not found. YouTube import will not be available. Install with: pip install yt-dlp")
    HAS_YTDLP = False

# Load environment variables manually
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

OSHIFIED_API_ENDPOINT = os.getenv("OSHIFIED_API_ENDPOINT")

# Required dependencies for S3 upload
try:
    import boto3
    from botocore.client import Config
    HAS_BOTO3 = True
except ImportError:
    print(f"[OshifiedNode] boto3 not found. S3 upload will not be available. Please install it via ComfyUI-Manager or with: pip install -r requirements.txt")
    HAS_BOTO3 = False

# S3/R2 Configuration for Cloudflare
# IMPORTANT: REPLACE THESE VALUES WITH YOUR ACTUAL CREDENTIALS
S3_CONFIG = {
    'endpoint_url': os.getenv("S3_ENDPOINT_URL"),
    'aws_access_key_id': os.getenv("S3_ACCESS_KEY_ID"),
    'aws_secret_access_key': os.getenv("S3_SECRET_ACCESS_KEY"),
    'bucket_name': os.getenv("S3_BUCKET_NAME"),
    'region_name': os.getenv("S3_REGION_NAME"),
}

import folder_paths
from .utils import ffmpeg_path, tensor_to_bytes, tensor_to_shorts, logger

# Define our formats path
folder_paths.folder_names_and_paths["OSHIFIED_video_formats"] = (
    [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_formats"),
    ],
    [".json"]
)

class VideoPathManager:
    """Manages all file paths for video processing to prevent overwrites."""
    def __init__(self, filename_prefix, output_dir, save_output=True):
        # Store the original filename_prefix (generation ID) for later use
        self.filename_prefix = filename_prefix
        
        # Ensure we have a valid prefix - if empty, use a default
        if not filename_prefix or filename_prefix.strip() == "":
            self.filename_prefix = f"oshified_{int(time.time())}"
            print(f"[VideoPathManager] WARNING: Empty filename_prefix provided, using: {self.filename_prefix}")
        
        self.output_dir = output_dir if save_output else folder_paths.get_temp_directory()
        
        # Get paths from ComfyUI's system (but don't rely on the filename it returns)
        self.full_output_folder, comfy_filename, _, self.subfolder, _ = folder_paths.get_save_image_path(self.filename_prefix, self.output_dir)
        
        # Log the paths for debugging
        print(f"[VideoPathManager] Initialized with prefix: {self.filename_prefix}")
        print(f"[VideoPathManager] Output folder: {self.full_output_folder}")
        
        self.output_files = []
        
    def get_first_frame_path(self):
        """Get path for the first frame PNG."""
        path = os.path.join(self.full_output_folder, f"{self.filename_prefix}_00001.png")
        return path
        
    def get_animation_path(self, format_ext):
        """Get path for animated image formats (GIF/WebP)."""
        path = os.path.join(self.full_output_folder, f"{self.filename_prefix}.{format_ext}")
        return path
        
    def get_video_path(self, extension):
        """Get path for the video file."""
        # Always use the original filename_prefix (generation ID) for the video filename
        path = os.path.join(self.full_output_folder, f"{self.filename_prefix}.{extension}")
        print(f"[VideoPathManager] Video path: {path}")
        return path
        
    def get_temp_video_path(self, extension):
        """Get temporary path for video processing."""
        path = os.path.join(folder_paths.get_temp_directory(), f"temp_{self.filename_prefix}.{extension}")
        return path
    
    def add_output_file(self, path):
        """Add a file to the list of output files."""
        self.output_files.append(path)
        return path
    
    def get_output_files(self):
        """Get all output files."""
        return self.output_files


class VideoProcessingLogger:
    """Enhanced logging for video processing."""
    def __init__(self, node_name):
        self.node_name = node_name
        self.logs = []
        self.send_logs_to_server = True
    
    def debug(self, msg):
        """Log debug message."""
        log_msg = f"[{self.node_name}] DEBUG: {msg}"
        print(log_msg)
        if self.send_logs_to_server:
            self.send_log_to_server(log_msg, "DEBUG")
        self.logs.append(log_msg)
    
    def info(self, msg):
        """Log info message."""
        log_msg = f"[{self.node_name}] INFO: {msg}"
        print(log_msg)
        if self.send_logs_to_server:
            self.send_log_to_server(log_msg, "INFO")
        self.logs.append(log_msg)
        
    def warn(self, msg):
        """Log warning message."""
        log_msg = f"[{self.node_name}] WARNING: {msg}"
        print(log_msg)
        if self.send_logs_to_server:
            self.send_log_to_server(log_msg, "WARNING")
        self.logs.append(log_msg)
        
    def error(self, msg):
        """Log error message."""
        log_msg = f"[{self.node_name}] ERROR: {msg}"
        print(log_msg, file=sys.stderr)
        if self.send_logs_to_server:
            self.send_log_to_server(log_msg, "ERROR")
        self.logs.append(log_msg)
    
    def get_logs(self):
        """Get all logs as a string."""
        return "\n".join(self.logs)
    
    def send_log_to_server(self, msg, type="INFO"):
        """Send log message to endpoint."""
        if OSHIFIED_API_ENDPOINT is not None:
            try:
                requests.post(f"{OSHIFIED_API_ENDPOINT.rstrip('/')}/logging/log", json={"log": {"message": msg, "context": "OSHIFIED_NODE_PROCESSING", "type": type}})
            except Exception as e:
                print(f"[{self.node_name}] WARNING: Failed to send log to server: {str(e)}")
        


class VideoProcessingState:
    """Tracks the state of video processing."""
    def __init__(self):
        self.total_frames = 0
        self.processed_frames = 0
        self.has_corner_watermark = False
        self.has_append_watermark = False
        self.has_audio = False
        self.audio_processed = False
        self.saved_first_frame = False
        self.video_created = False
        self.upload_attempted = False
        self.upload_successful = False
        self.errors = []
        
    def add_error(self, error_msg):
        """Add an error message."""
        self.errors.append(error_msg)
    
    def has_errors(self):
        """Check if there are any errors."""
        return len(self.errors) > 0


def gen_format_widgets(video_format):
    """Generate format widgets from video format."""
    for k in video_format:
        if k.endswith("_pass"):
            for i in range(len(video_format[k])):
                if isinstance(video_format[k][i], list):
                    item = [video_format[k][i]]
                    yield item
                    video_format[k][i] = item[0]
        else:
            if isinstance(video_format[k], list):
                item = [video_format[k]]
                yield item
                video_format[k] = item[0]


def get_video_formats():
    """Get available video formats."""
    formats = []
    for format_name in folder_paths.get_filename_list("OSHIFIED_video_formats"):
        format_name = format_name[:-5]
        video_format_path = folder_paths.get_full_path("OSHIFIED_video_formats", format_name + ".json")
        with open(video_format_path, 'r') as stream:
            video_format = json.load(stream)
        widgets = [w[0] for w in gen_format_widgets(video_format)]
        if (len(widgets) > 0):
            formats.append(["video/" + format_name, widgets])
        else:
            formats.append("video/" + format_name)
    return formats


def apply_format_widgets(format_name, kwargs):
    """Apply format widgets to video format."""
    video_format_path = folder_paths.get_full_path("OSHIFIED_video_formats", format_name + ".json")
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    for w in gen_format_widgets(video_format):
        if w[0][0] not in kwargs:
            if len(w[0]) > 2 and 'default' in w[0][2]:
                default = w[0][2]['default']
            else:
                if type(w[0][1]) is list:
                    default = w[0][1][0]
                else:
                    default = {"BOOLEAN": False, "INT": 0, "FLOAT": 0, "STRING": ""}[w[0][1]]
            kwargs[w[0][0]] = default
            logger.warn(f"Missing input for {w[0][0]} has been set to {default}")
        if len(w[0]) > 3:
            w[0] = Template(w[0][3]).substitute(val=kwargs[w[0][0]])
        else:
            w[0] = str(kwargs[w[0][0]])
    return video_format


def ffmpeg_process(args, video_format, video_metadata, file_path, env, custom_logger=None):
    """Run ffmpeg process with generator pattern."""
    log = custom_logger or logger
    res = None
    frame_data = yield
    total_frames_output = 0
    if video_format.get('save_metadata', 'False') != 'False':
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        metadata = json.dumps(video_metadata)
        metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")
        metadata = metadata.replace("\\","\\\\")
        metadata = metadata.replace(";","\\;")
        metadata = metadata.replace("#","\\#")
        metadata = metadata.replace("=","\\=")
        metadata = metadata.replace("\n","\\\n")
        metadata = "comment=" + metadata
        with open(metadata_path, "w") as f:
            f.write(";FFMETADATA1\n")
            f.write(metadata)
        m_args = args[:1] + ["-i", metadata_path] + args[1:] + ["-metadata", "creation_time=now"]
        with subprocess.Popen(m_args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                err = proc.stderr.read()
                log.error(f"FFMPEG BROKEN PIPE (during metadata pass). STDERR: {err.decode('utf-8', errors='ignore')}")
                if os.path.exists(file_path):
                    raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                            + err.decode('utf-8'))
                print(err.decode('utf-8'), end="", file=sys.stderr)
                log.warn("An error occurred when saving with metadata")
    if res != b'':
        with subprocess.Popen(args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                res = proc.stderr.read()
                log.error(f"FFMPEG BROKEN PIPE (during main pass). STDERR: {res.decode('utf-8', errors='ignore')}")
                raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                        + res.decode('utf-8'))
    
    # The 'log' variable (e.g., log = custom_logger or logger) should be defined at the start of ffmpeg_process
    if res is not None and len(res) > 0:
        log.warn(f"FFMPEG FINAL STDERR CONTENT (after processing all frames): {res.decode('utf-8', errors='ignore')}")

    yield total_frames_output


def resample_frames_ffmpeg(frames, source_fps, target_fps, logger):
    """
    Resamples frames using ffmpeg for higher quality conversion.
    """
    if not frames or source_fps <= 0 or target_fps <= 0 or source_fps == target_fps:
        return frames

    logger.info(f"Resampling frames from {source_fps} to {target_fps} using ffmpeg.")
    
    temp_dir = folder_paths.get_temp_directory()
    temp_input_path = os.path.join(temp_dir, f"temp_resample_in_{int(time.time())}.mkv")
    
    # 1. Encode the source frames into a temporary lossless video
    h, w, c = frames[0].shape
    pix_fmt_in = 'rgba' if c == 4 else 'rgb24'
    
    encode_cmd = [
        ffmpeg_path, '-y', '-f', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', pix_fmt_in, '-r', str(source_fps),
        '-i', '-', '-an', '-vcodec', 'ffv1', temp_input_path
    ]
    
    logger.info(f"Running ffmpeg encode: {' '.join(encode_cmd)}")
    
    # Prepare all frame data in one go
    input_bytes = b''.join([tensor_to_bytes(frame).tobytes() for frame in frames])
    
    # Use communicate to handle stdin, stdout, and stderr safely
    proc_encode = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc_encode.communicate(input=input_bytes)

    if proc_encode.returncode != 0:
        logger.error(f"ffmpeg encode failed with code {proc_encode.returncode}")
        logger.error(f"ffmpeg stderr: {stderr.decode('utf-8', errors='ignore')}")
        # Clean up and raise exception
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        raise Exception("ffmpeg frame resampling (encode step) failed.")

    # 2. Decode the video, letting ffmpeg handle the frame rate conversion, and read raw frames back
    pix_fmt_out = 'rgb24' # We want a consistent 3-channel output for the main video stream
    decode_cmd = [
        ffmpeg_path, '-i', temp_input_path, '-vf', f'fps={target_fps}',
        '-f', 'rawvideo', '-pix_fmt', pix_fmt_out, '-'
    ]

    logger.info(f"Running ffmpeg decode: {' '.join(decode_cmd)}")
    proc_decode = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    output_frames = []
    while True:
        frame_bytes = proc_decode.stdout.read(w * h * 3) # 3 bytes per pixel for rgb24
        if not frame_bytes:
            break
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w, 3))
        output_frames.append(torch.from_numpy(frame).float() / 255.0)
        
    stdout, stderr = proc_decode.communicate()
    if proc_decode.returncode != 0:
        logger.error(f"ffmpeg decode failed with code {proc_decode.returncode}")
        logger.error(f"ffmpeg stderr: {stderr.decode('utf-8', errors='ignore')}")
        # Clean up and raise exception
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        raise Exception("ffmpeg frame resampling (decode step) failed.")
    
    # Clean up temporary file
    if os.path.exists(temp_input_path):
        os.unlink(temp_input_path)
        
    logger.info(f"Resampling complete. Original: {len(frames)} frames, Resampled: {len(output_frames)} frames.")
    return output_frames

def process_watermark_frames(watermark_frames, main_shape):
    """Process watermark frames to match main video dimensions."""
    # Ensure watermark_frames is a list of 3D tensors for torch.stack
    if isinstance(watermark_frames, torch.Tensor):
        if watermark_frames.ndim == 4:  # Batch of frames [N, H, W, C]
            watermark_frames = [watermark_frames[i] for i in range(watermark_frames.size(0))]
        elif watermark_frames.ndim == 3:  # Single frame [H, W, C]
            watermark_frames = [watermark_frames]  # Wrap in a list
        else:
            # Handle unexpected tensor dimensions
            print(f"[process_watermark_frames] ERROR: Unexpected watermark tensor dimension: {watermark_frames.ndim}")
            return []  # Return empty list to skip append watermark

    if not watermark_frames:  # Empty list check
        print("[process_watermark_frames] WARNING: Empty watermark_frames list")
        return []
            
    h, w = main_shape[:2]
    
    # Get watermark dimensions - handle non-square watermarks correctly
    wm_h, wm_w = watermark_frames[0].shape[0], watermark_frames[0].shape[1]
    
    # Always create a black background matching main video dimensions
    black_bg = torch.zeros((h, w, main_shape[2]), dtype=torch.float32)
    
    if h >= wm_h and w >= wm_w:
        # Watermark fits within main dimensions - center it on black background
        x_offset = (w - wm_w) // 2
        y_offset = (h - wm_h) // 2
        # Place watermark frames on background
        for i, frame in enumerate(watermark_frames):
            new_bg = black_bg.clone()  # Create a new tensor to avoid in-place modification issues
            new_bg[y_offset:y_offset+wm_h, x_offset:x_offset+wm_w] = frame
            watermark_frames[i] = new_bg
    else:
        # Watermark is too large - scale it down preserving aspect ratio
        scale_h = h / wm_h
        scale_w = w / wm_w
        scale = min(scale_h, scale_w)  # Use smallest scale to ensure it fits
        
        # Calculate new dimensions preserving aspect ratio
        new_h = int(wm_h * scale)
        new_w = int(wm_w * scale)
        new_size = (new_h, new_w)
        
        # Stack frames into a batch, then resize
        watermark_tensor = torch.stack(watermark_frames)
        resized = F.interpolate(watermark_tensor.permute(0,3,1,2), size=new_size)
        resized_frames = list(resized.permute(0,2,3,1))
        
        # Center the resized watermark on black background
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2
        
        # Place each resized watermark on its own black background
        for i, frame in enumerate(resized_frames):
            new_bg = black_bg.clone()  # Create a new tensor to avoid in-place modification issues
            new_bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame
            watermark_frames[i] = new_bg
    
    return watermark_frames


def process_corner_watermark(watermark_frames, mask_frames, main_shape, opacity=0.4, logger=None):
    """Process corner watermark with mask for transparency."""
    log = logger or logging.getLogger("VideoHelperSuite")
    
    # Ensure watermark_frames is a list of 3D tensors
    if isinstance(watermark_frames, torch.Tensor) and watermark_frames.ndim == 4:
        watermark_frames = [frame for frame in watermark_frames]  # Convert 4D tensor to list of 3D tensors
    elif isinstance(watermark_frames, torch.Tensor) and watermark_frames.ndim == 3:
        # If it's a single 3D tensor, wrap it in a list
        watermark_frames = [watermark_frames]

    # Ensure mask_frames is also a list
    if isinstance(mask_frames, torch.Tensor) and mask_frames.ndim == 3:  # Batch of masks (BxHxW)
        mask_frames = [mask for mask in mask_frames]
    elif isinstance(mask_frames, torch.Tensor) and mask_frames.ndim == 2:
        mask_frames = [mask_frames]

    # Check if watermark_frames or mask_frames are empty or invalid
    if not watermark_frames or watermark_frames[0].numel() == 0 or watermark_frames[0].shape[0] == 0 or watermark_frames[0].shape[1] == 0:
        log.warn("Corner watermark input is empty or has zero dimensions. Skipping corner watermark.")
        # Return empty lists and zero offset so calling code can handle gracefully
        return [], [], (0, 0) 
        
    if not mask_frames or mask_frames[0].numel() == 0 or mask_frames[0].shape[0] == 0 or mask_frames[0].shape[1] == 0:
        log.warn("Corner watermark mask input is empty or has zero dimensions. Skipping corner watermark.")
        return [], [], (0, 0)

    h, w = main_shape[:2]
    watermark_h, watermark_w = watermark_frames[0].shape[:2]
    
    # Additional check for watermark dimensions
    if watermark_h == 0 or watermark_w == 0:
        log.warn(f"Corner watermark has zero height or width ({watermark_h}x{watermark_w}). Skipping corner watermark.")
        return [], [], (0, 0)

    # Scale to % of video width
    target_width = int(w * 0.2)  # 20% of video width
    if target_width == 0:  # Should not happen if w > 0, but as a safeguard
        log.warn(f"Main video width is too small ({w}) to calculate target_width for corner watermark. Skipping.")
        return [], [], (0, 0)
        
    scale = target_width / watermark_w
    new_h = int(watermark_h * scale)
    new_w = target_width  # target_width is already int

    if new_h == 0 or new_w == 0:
        log.warn(f"Calculated new_size for corner watermark is invalid ({new_h}x{new_w}). Skipping corner watermark.")
        return [], [], (0, 0)
    new_size = (new_h, new_w)
    
    # Position in bottom right
    x_offset = w - new_size[1] - 20  # 20px padding
    y_offset = h - new_size[0] - 20
    
    try:
        # Process frames
        watermark_tensor = torch.stack(watermark_frames)
        mask_tensor = torch.stack(mask_frames)
        
        # Resize both watermark and mask
        resized_watermark = F.interpolate(watermark_tensor.permute(0,3,1,2), size=new_size)
        resized_mask = F.interpolate(mask_tensor.unsqueeze(1), size=new_size).squeeze(1)
        
        # Convert back to HWC
        processed_watermark = list(resized_watermark.permute(0,2,3,1))
        processed_mask = list(resized_mask)
        
        return processed_watermark, processed_mask, (x_offset, y_offset)
    except Exception as e:
        log.error(f"Error processing corner watermark: {e}")
        return [], [], (0, 0)


def to_pingpong(inp):
    """Convert input to ping-pong pattern."""
    if not hasattr(inp, "__getitem__"):
        inp = list(inp)
    yield from inp
    for i in range(len(inp)-2,0,-1):
        yield inp[i]


class OshifiedVideoSaver:
    """
    Comprehensive video processing and saving node for ComfyUI - Oshified Edition.
    
    This node takes generated frames from ComfyUI workflows and processes them into a complete,
    properly formatted video file with audio. It handles multiple critical aspects of video 
    creation:
    
    1. Frame Processing - Handles dimension alignment, watermark overlays, and appending
    2. Audio Processing - Extracts audio from inputs, creates silent audio if needed, and 
                         concatenates user and watermark audio segments
    3. Video Creation - Uses ffmpeg to encode frames with proper settings
    4. Cloud Upload - Uploads completed videos to cloud storage
    
    The node preserves the original video timing and speed throughout all processing steps,
    never altering playback speed even when handling audio synchronization. This ensures
    the final output plays exactly as intended, with proper timing for both video and audio.
    
    All processing steps include extensive logging and error handling to aid in diagnosing
    any issues that might arise during the video creation process.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats = get_video_formats()
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "FLOAT",
                    {"default": 8, "min": 1, "step": 1},
                ),
                "filename_prefix": ("STRING", {"default": "Oshified"}),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats,),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO", {"lazy": True}),
                "vae": ("VAE",),
                "append_watermark": ("IMAGE",),
                "watermark_audio": ("AUDIO", {"lazy": True}),
                "corner_watermark": ("IMAGE",),
                "corner_watermark_mask": ("MASK",),
                "corner_opacity": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
                "append_watermark_source_fps": ("FLOAT", {"default": 0, "min": 0, "max": 120, "step": 1}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("OSHIFIED_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    CATEGORY = "Oshified 🌸"
    FUNCTION = "save_video"

    def __init__(self):
        self.logger = VideoProcessingLogger("OshifiedVideoSaver")
    
    def _send_progress_update(self, current_step, total_steps):
        """Send progress update to ComfyUI server."""
        if self.server and hasattr(self.server, 'send_sync'):
            try:
                progress_data = {
                    "node": self.node_id,
                    "value": current_step,
                    "max": total_steps
                }
                self.server.send_sync("progress", progress_data)
                self.logger.info(f"Progress update sent: {current_step}/{total_steps}")
            except Exception as e:
                self.logger.warn(f"Failed to send progress update: {str(e)}")

    def check_lazy_status(self, images, audio=None, watermark_audio=None, **kwargs):
        # This method is called before the main save_video function to check if lazy inputs are needed.
        # If an audio input is connected but not yet available (is None), we tell the executor to wait for it.
        
        # Since we can't know from here if the user *intended* to connect audio,
        # we have to assume that if the lazy input is None, it's because an upstream node
        # hasn't finished executing yet. We will always wait for them.
        
        needed = []
        if audio is None:
            needed.append('audio')
        if watermark_audio is None:
            needed.append('watermark_audio')
            
        # If the list is not empty, the execution will be paused until these inputs are available.
        if needed:
            self.logger.info(f"Waiting for lazy inputs: {', '.join(needed)}")
        
        return needed

    def save_video(
        self,
        images,
        frame_rate: int,
        filename_prefix="Oshified",
        format="video/h264-mp4",
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        audio=None,
        vae=None,
        append_watermark=None,
        watermark_audio=None,
        corner_watermark=None,
        corner_watermark_mask=None,
        corner_opacity=0.4,
        append_watermark_source_fps=0,
        server=None,
        **kwargs
    ):
        """
        Main entry point for video saving process.
        
        This method orchestrates the entire video creation pipeline, maintaining original
        video timing throughout. The process involves several key steps:
        
        1. Preprocessing input frames and validation
        2. Generating metadata from prompt information
        3. Saving the first frame with metadata for reference
        4. Processing frames including overlay/append watermarks while preserving original speed
        5. Creating the video file with constant frame rate
        6. Processing and concatenating audio to match video segments
        7. Uploading to cloud storage if configured
        
        Unlike some approaches that alter frame counts to match audio, this implementation
        prioritizes consistent video playback speed. Audio segments are prepared to match
        the natural timing of video segments at the specified frame rate.
        
        Args:
            images: Input video frames (Tensor or list of Tensors)
            frame_rate: Desired output frame rate
            filename_prefix: Prefix for output files (default: "Oshified")
            format: Output format (e.g., "video/h264-mp4")
            save_output: Whether to save to output directory or temp directory
            prompt: Optional prompt information for metadata
            extra_pnginfo: Additional metadata for the video
            audio: Optional audio data to include with the video
            vae: Optional VAE model for preprocessing
            append_watermark: Optional frames to append after the main video
            watermark_audio: Optional audio to append after main audio
            corner_watermark: Optional watermark to overlay in corner
            corner_watermark_mask: Mask for corner watermark
            corner_opacity: Opacity for corner watermark (0.0-1.0)
            **kwargs: Additional format-specific arguments
            
        Returns:
            Dictionary with UI information and result tuple containing success status and output files
        """
        # Initialize helper classes
        self.path_manager = VideoPathManager(
            filename_prefix, 
            folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory(),
            save_output
        )
        self.state = VideoProcessingState()
        
        # Initialize progress reporting
        self.server = server
        self.node_id = "487"  # VidiaVideoSaver node ID
        self.total_progress_steps = 10
        self.current_progress = 0
        
        # Send initial progress update
        self._send_progress_update(1, self.total_progress_steps)
        
        # Log start of processing
        self.logger.info(f"Starting video processing with {len(images) if isinstance(images, list) else images.size(0)} frames")
        
        # Input validation and preprocessing
        images = self._preprocess_inputs(images, vae)
        if isinstance(images, tuple) and len(images) == 0:
            return ((save_output, []),)
        
        # Create metadata
        video_metadata = self._prepare_metadata(prompt, extra_pnginfo)
        
        # Skip first frame PNG save for speed (not needed for final output)
        # self._save_first_frame(images[0], video_metadata)
        self._send_progress_update(3, self.total_progress_steps)
        
        # Process based on format type
        format_type, format_ext = format.split("/")
        self.logger.info(f"Using format: {format_type}/{format_ext}")
        
        if format_type == "image":
            result = self._process_image_format(images, format_ext, frame_rate)
        else:
            # Use ffmpeg for video processing
            if ffmpeg_path is None:
                self.logger.error("ffmpeg not found")
                raise ProcessLookupError("ffmpeg is required for video outputs and could not be found.")
            
            # Get format settings and prepare
            video_format = apply_format_widgets(format_ext, kwargs)
            
            # Keep track of the number of frames before append watermark
            user_frames_count = len(images)
            self.state.processed_frames = user_frames_count
            
            # Process frames (watermarks, etc.) and adjust frame rates for proper timing
            processed_frames = self._prepare_frames(
                images, video_format, corner_watermark,
                corner_watermark_mask, corner_opacity, append_watermark,
                audio, watermark_audio, frame_rate, append_watermark_source_fps
            )
            
            # Calculate append watermark frames (if any)
            watermark_frames_count = len(processed_frames) - user_frames_count if self.state.has_append_watermark else 0
            self.logger.info(f"Frame counts - User: {user_frames_count}, Append Watermark: {watermark_frames_count}, Total: {len(processed_frames)}")
            
            # Create video file
            success = self._create_video_file(processed_frames, video_format, video_metadata, frame_rate)
            self._send_progress_update(6, self.total_progress_steps)
            
            if success:
                # Process audio if present or if we have frames that might need silent audio
                self._process_audio(
                    audio, watermark_audio, video_format, frame_rate,
                    user_frames_count=user_frames_count,
                    watermark_frames_count=watermark_frames_count
                )
                self._send_progress_update(7, self.total_progress_steps)
                
                # Upload to Cloudflare
                try:
                    # Try S3 upload first, fall back to base64 if needed
                    if HAS_BOTO3:
                        upload_result = self._upload_to_cloudflare_s3(format, self.path_manager.get_output_files()[-1], filename_prefix)
                    else:
                        # Base64 upload is deprecated and removed.
                        self.logger.error("boto3 is not installed, cannot upload to S3.")
                        self._send_progress_update(self.total_progress_steps, self.total_progress_steps)
                        return {"ui": {"error": "boto3 is not installed, cannot upload to S3. Video saved locally."}, 
                                "result": ((save_output, self.path_manager.get_output_files()),)}

                    # Process upload result - ensure error is always a string
                    if upload_result.get("success", False):
                        # Send final completion signal
                        self._send_progress_update(self.total_progress_steps, self.total_progress_steps)
                        return {"ui": {"response": upload_result}, 
                                "result": ((save_output, self.path_manager.get_output_files()),)}
                    else:
                        # Ensure error is a string (fixes issue with error arrays)
                        if isinstance(upload_result.get('error'), list):
                            error_msg = ''.join(upload_result.get('error'))
                        else:
                            error_msg = str(upload_result.get('error', 'Unknown error'))
                        
                        self._send_progress_update(self.total_progress_steps, self.total_progress_steps)    
                        return {"ui": {"error": f"Upload failed but video saved locally: {error_msg}"}, 
                                "result": ((save_output, self.path_manager.get_output_files()),)}
                except Exception as e:
                    error_msg = str(e)
                    self.logger.error(f"Upload failed: {error_msg}")
                    self._send_progress_update(self.total_progress_steps, self.total_progress_steps)
                    return {"ui": {"error": f"Upload failed but video saved locally: {error_msg}"}, 
                            "result": ((save_output, self.path_manager.get_output_files()),)}
        
        # Send final completion signal for any other code path
        self._send_progress_update(self.total_progress_steps, self.total_progress_steps)
        return {"ui": {"success": "Video processing completed"}, 
                "result": ((save_output, self.path_manager.get_output_files()),)}

    def _preprocess_inputs(self, images, vae=None):
        """Validate and preprocess input images."""
        if isinstance(images, dict):
            images = images['samples']
        
        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return tuple()

        if vae is not None:
            if isinstance(images, dict):
                images = images['samples']
            else:
                vae = None
                
        return images

    def _prepare_metadata(self, prompt, extra_pnginfo):
        """Prepare metadata for the video file."""
        video_metadata = {}
        
        if prompt is not None:
            video_metadata["prompt"] = json.dumps(prompt)
            
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                video_metadata[x] = extra_pnginfo[x]
                
        video_metadata["CreationTime"] = datetime.datetime.now().isoformat(" ")[:19]
        
        return video_metadata

    def _save_first_frame(self, first_frame, video_metadata):
        """Save the first frame as a PNG with metadata."""
        # Create PNG metadata
        metadata = PngInfo()
        for key, value in video_metadata.items():
            if isinstance(value, str):
                metadata.add_text(key, value)
            else:
                metadata.add_text(key, json.dumps(value))
        
        # Save first frame
        first_image_path = self.path_manager.get_first_frame_path()
        self.logger.info(f"Saving first frame to {first_image_path}")
        
        try:
            Image.fromarray(tensor_to_bytes(first_frame)).save(
                first_image_path,
                pnginfo=metadata,
                compress_level=4,
            )
            self.path_manager.add_output_file(first_image_path)
            self.state.saved_first_frame = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to save first frame: {str(e)}")
            self.state.add_error(f"First frame save failed: {str(e)}")
            return False

    def _process_image_format(self, images, format_ext, frame_rate):
        """Process and save as an animated image format (GIF/WebP)."""
        # Prepare image kwargs based on format
        image_kwargs = {}
        if format_ext == "gif":
            image_kwargs['disposal'] = 2
        if format_ext == "webp":
            # Save timestamp information
            exif = Image.Exif()
            exif[ExifTags.IFD.Exif] = {36867: datetime.datetime.now().isoformat(" ")[:19]}
            image_kwargs['exif'] = exif
        
        # Generate output path
        file_path = self.path_manager.get_animation_path(format_ext)
        self.logger.info(f"Saving animation to {file_path}")
        
        try:
            # Convert frames to PIL images
            frames = map(lambda x: Image.fromarray(tensor_to_bytes(x)), images)
            
            # Save animated image
            next(frames).save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames,
                duration=round(1000 / frame_rate),
                loop=0,
                compress_level=4,
                **image_kwargs
            )
            
            # Update state and output files
            self.path_manager.add_output_file(file_path)
            self.state.video_created = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to create {format_ext} animation: {str(e)}")
            self.state.add_error(f"Animation creation failed: {str(e)}")
            return False

    def _prepare_frames(self, images, video_format, corner_watermark=None, 
                       corner_watermark_mask=None, corner_opacity=0.4, append_watermark=None, 
                       audio=None, watermark_audio=None, frame_rate=None, append_watermark_source_fps=0):
        """
        Prepare frames including any watermarks, with frame rate adjustment for proper timing.
        
        Args:
            images: User video frames
            video_format: Video format settings
            corner_watermark: Overlay watermark frames
            corner_watermark_mask: Mask for overlay watermark
            corner_opacity: Opacity for overlay watermark
            append_watermark: Frames to append after main video
            audio: User audio data (dict with 'samples' and 'sample_rate')
            watermark_audio: Watermark audio data (dict with 'samples' and 'sample_rate')
            frame_rate: Output frame rate
        """
        # Convert to list for dimension checks
        if isinstance(images, torch.Tensor):
            images = list(images)
        first_frame = images[0]
        has_alpha = first_frame.shape[-1] == 4
        
        # Store original user frames count (before any modifications)
        original_user_frames_count = len(images)
        
        # Skip dimension alignment - Oshified always outputs 640x640
        dimensions = f"{first_frame.shape[1]}x{first_frame.shape[0]}"
        
        # Process corner watermark if provided (OPTIMIZED: Batch processing)
        # Only apply to the user video frames (not append watermark)
        if corner_watermark is not None and corner_watermark_mask is not None:
            self.logger.info("Applying corner watermark (batch optimized)")
            
            # Convert to list if tensor
            if isinstance(corner_watermark, torch.Tensor) and corner_watermark.ndim == 3:
                corner_watermark = [corner_watermark] * len(images)
            if isinstance(corner_watermark_mask, torch.Tensor) and corner_watermark_mask.ndim == 2:
                corner_watermark_mask = [corner_watermark_mask] * len(images)
                
            # Process corner watermark
            processed_watermark, processed_mask, (x_offset, y_offset) = process_corner_watermark(
                corner_watermark, corner_watermark_mask, first_frame.shape, corner_opacity, logger=self.logger
            )
            
            # Check if processed_watermark or processed_mask are empty
            if not processed_watermark or not processed_mask:
                self.logger.info("Skipping corner watermark application due to invalid input or processing error.")
            else:
                try:
                    # OPTIMIZED: Batch process all frames at once
                    # Stack frames and watermarks into batches
                    images_batch = torch.stack(images)
                    
                    # Repeat watermarks to match frame count if needed
                    num_frames = len(images)
                    num_watermarks = len(processed_watermark)
                    if num_watermarks < num_frames:
                        watermark_indices = [i % num_watermarks for i in range(num_frames)]
                        watermarks_batch = torch.stack([processed_watermark[i] for i in watermark_indices])
                        masks_batch = torch.stack([processed_mask[i] for i in watermark_indices])
                    else:
                        watermarks_batch = torch.stack(processed_watermark[:num_frames])
                        masks_batch = torch.stack(processed_mask[:num_frames])
                    
                    # Expand mask to 3 channels (batch operation)
                    masks_3d = masks_batch.unsqueeze(-1).repeat(1, 1, 1, 3)
                    
                    # Get watermark dimensions
                    h, w = watermarks_batch.shape[1:3]
                    
                    # Calculate slice bounds
                    y_start = max(0, y_offset)
                    x_start = max(0, x_offset)
                    y_end = min(images_batch.shape[1], y_offset + h)
                    x_end = min(images_batch.shape[2], x_offset + w)
                    
                    h_slice = y_end - y_start
                    w_slice = x_end - x_start
                    
                    if h_slice > 0 and w_slice > 0:
                        # Calculate watermark slices
                        wm_y_start = max(0, -y_offset)
                        wm_x_start = max(0, -x_offset)
                        wm_y_end = wm_y_start + h_slice
                        wm_x_end = wm_x_start + w_slice
                        
                        # Extract regions (batch operation)
                        frame_regions = images_batch[:, y_start:y_end, x_start:x_end]
                        watermark_regions = watermarks_batch[:, wm_y_start:wm_y_end, wm_x_start:wm_x_end] * corner_opacity
                        mask_regions = masks_3d[:, wm_y_start:wm_y_end, wm_x_start:wm_x_end]
                        
                        # Batch blend: frame * (1 - mask) + watermark * mask
                        blended_regions = frame_regions * (1.0 - mask_regions) + watermark_regions * mask_regions
                        
                        # Update all frames at once
                        images_batch[:, y_start:y_end, x_start:x_end] = blended_regions
                        
                        # Convert back to list
                        images = list(images_batch)
                        
                        self.state.has_corner_watermark = True
                    else:
                        self.logger.warn("Watermark slice dimensions invalid, skipping corner watermark")
                        
                except Exception as e:
                    self.logger.error(f"Error applying corner watermark (batch mode): {e}")
                    # Fall back to original images without watermark
        
        # Process append watermark if provided
        watermark_frames = []
        if append_watermark is not None:
            self.logger.info("Processing append watermark")
            
            # Convert to list if tensor
            if isinstance(append_watermark, torch.Tensor) and append_watermark.ndim == 3:
                append_watermark = [append_watermark] * len(images)
                
            # Process watermark frames - adjust dimensions but don't append yet
            watermark_frames = process_watermark_frames(append_watermark, first_frame.shape)
            
            # Skip FPS resampling - Oshified always uses 16fps, watermarks should match
            # If FPS mismatch detected, warn but don't resample (too slow)
            if append_watermark_source_fps > 0 and frame_rate > 0 and abs(append_watermark_source_fps - frame_rate) > 0.01:
                self.logger.warn(f"Watermark FPS ({append_watermark_source_fps}) doesn't match output FPS ({frame_rate}). Export watermark at {frame_rate} fps for best results.")
            
            self.state.has_append_watermark = True
            
        # Calculate video durations to provide to audio processing
        # Note: We preserve original frame counts to maintain video speed
        if frame_rate is not None and frame_rate > 0:
            # Calculate user video duration
            user_video_duration_sec = len(images) / frame_rate
            
            # Store for audio processing
            self.user_video_duration = user_video_duration_sec
            
            # Check if we have watermark frames
            if watermark_frames:
                watermark_video_duration_sec = len(watermark_frames) / frame_rate
                self.watermark_video_duration = watermark_video_duration_sec
            else:
                self.watermark_video_duration = 0
        
        # Now append watermark frames after processing
        if watermark_frames:
            images.extend(watermark_frames)
        
        # Return processed frames and dimensions
        return images
        
    def _create_video_file(self, processed_frames, video_format, video_metadata, frame_rate):
        """Create video file with ffmpeg."""
        # Determine video format extension and other settings
        format_ext = video_format.get("extension", "mp4")
        pix_fmt = video_format.get("pix_fmt", "yuv420p")
        
        # Determine encoder and other ffmpeg settings
        encoder = video_format.get("encoder", "libx264")
        parameters = []
        if "parameters" in video_format:
            parameters = video_format.get("parameters").split(" ")
            
        # Get additional args for first and second pass
        first_pass = video_format.get("first_pass", [])
        second_pass = video_format.get("second_pass", [])
        
        # Generate output path
        file_path = self.path_manager.get_video_path(format_ext)
        self.logger.info(f"Creating video at {file_path}")
        
        try:
            # Convert frames to the appropriate format
            if isinstance(processed_frames, torch.Tensor):
                processed_frames = list(processed_frames)
                
            # Prepare the ffmpeg process
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
            
            # Configure ffmpeg args
            args = [
                ffmpeg_path,
                "-y",  # Add overwrite flag to prevent "file already exists" errors
                "-v", "error", "-f", "rawvideo",
                "-pix_fmt", "rgb24" if processed_frames[0].shape[2] == 3 else "rgba",
                "-s", f"{processed_frames[0].shape[1]}x{processed_frames[0].shape[0]}",
                "-r", str(frame_rate),
                "-i", "-"
            ]
            
            # Add format-specific args
            for arg in parameters:
                args.append(arg)
                
            for arg in second_pass:
                args.append(arg)
                
            # Add encoder and pixel format
            args.extend(["-c:v", encoder, "-pix_fmt", pix_fmt])
            
            # Start ffmpeg process
            # Reduced logging for speed
            proc = ffmpeg_process(args, video_format, video_metadata, file_path, env, self.logger)
            next(proc)  # Advance generator to first yield
            
            # Send frame data to ffmpeg
            for frame in processed_frames:
                proc.send(tensor_to_bytes(frame).tobytes())
            
            # Close the process and get total frames
            total_frames_output = proc.send(None)
            self.logger.info(f"Finished encoding video with {total_frames_output} frames")
            
            # Check if output file exists and has valid size
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size < 1000:  # Less than 1KB is suspicious
                    self.logger.error(f"Video file is too small, likely corrupt: {file_path} ({file_size} bytes)")
                    self.state.add_error(f"Video file is too small: {file_size} bytes")
                    return False
                else:
                    self.logger.info(f"Created video file: {file_path} ({file_size} bytes)")
                    self.path_manager.add_output_file(file_path)
                    self.state.video_created = True
                    return True
            else:
                self.logger.error(f"Video file not created: {file_path}")
                self.state.add_error("Video file not created")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to create video: {str(e)}")
            self.state.add_error(f"Video creation failed: {str(e)}")
            return False
            
    def _process_audio(self, audio, watermark_audio, video_format, frame_rate, user_frames_count=None, watermark_frames_count=None):
        if audio is None and watermark_audio is None and not user_frames_count:
            self.logger.info("No audio provided and cannot determine user video duration")
            return False

        format_ext = video_format.get("extension", "mp4")
        video_path = self.path_manager.get_output_files()[-1]
        temp_path = self.path_manager.get_temp_video_path(format_ext)
        temp_dir = folder_paths.get_temp_directory()
        temp_files = []
        default_sample_rate = 44100

        def get_audio_details(audio_data, name):
            if audio_data is None:
                return None, 0
            try:
                waveform = audio_data["waveform"]
                if len(waveform.shape) == 3:
                    waveform = waveform.squeeze(0)
                channels = 1 if len(waveform.shape) == 1 else waveform.shape[0]
                return waveform, channels
            except Exception as e:
                self.logger.error(f"Invalid {name} audio format: {e}")
                return None, 0

        def save_wav(waveform, path, sample_rate, channels):
            with wave.open(path, "w") as f:
                f.setnchannels(channels)
                f.setsampwidth(2)
                f.setframerate(sample_rate)
                if isinstance(waveform, torch.Tensor):
                    waveform = waveform.cpu().numpy()
                if waveform.dtype != np.int16:
                    waveform = (waveform * 32767).astype(np.int16)
                if channels > 1:
                    waveform = waveform.T # Transpose for interleaving
                f.writeframes(waveform.tobytes())
            return path

        try:
            user_waveform, user_channels = get_audio_details(audio, "user")
            wm_waveform, wm_channels = get_audio_details(watermark_audio, "watermark")

            # Standardize to stereo if there's a mismatch
            if user_channels > 0 and wm_channels > 0 and user_channels != wm_channels:
                self.logger.info(f"Channel mismatch: User ({user_channels}) vs Watermark ({wm_channels}). Upmixing to stereo.")
                if user_channels == 1:
                    # A mono waveform of shape [1, samples] becomes stereo [2, samples] by repeating the first channel.
                    user_waveform = user_waveform.repeat(2, 1)
                    user_channels = 2
                if wm_channels == 1:
                    wm_waveform = wm_waveform.repeat(2, 1)
                    wm_channels = 2
            
            final_audio_files = []

            # Process User Audio
            user_video_duration = (user_frames_count or 0) / frame_rate
            if user_waveform is not None:
                raw_user_audio_path = os.path.join(temp_dir, f"user_audio_raw_{int(time.time())}.wav")
                temp_files.append(raw_user_audio_path)
                save_wav(user_waveform, raw_user_audio_path, audio["sample_rate"], user_channels)

                # Adjust duration to match video segment
                adjusted_user_audio_path = os.path.join(temp_dir, f"user_audio_adjusted_{int(time.time())}.wav")
                temp_files.append(adjusted_user_audio_path)
                
                # Use ffmpeg to trim or pad the audio to the exact video duration
                adjust_cmd = [ffmpeg_path, "-y", "-i", raw_user_audio_path, "-af", f"apad,atrim=0:{user_video_duration}", "-t", str(user_video_duration), adjusted_user_audio_path]
                subprocess.run(adjust_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                final_audio_files.append(adjusted_user_audio_path)

            elif user_video_duration > 0:
                user_audio_path = os.path.join(temp_dir, f"silent_user_{int(time.time())}.wav")
                temp_files.append(user_audio_path)
                subprocess.run([ffmpeg_path, "-y", "-f", "lavfi", "-i", f"anullsrc=cl=stereo:r={default_sample_rate}", "-t", str(user_video_duration), user_audio_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                final_audio_files.append(user_audio_path)

            # Process Watermark Audio
            wm_video_duration = (watermark_frames_count or 0) / frame_rate
            if wm_waveform is not None:
                raw_wm_audio_path = os.path.join(temp_dir, f"wm_audio_raw_{int(time.time())}.wav")
                temp_files.append(raw_wm_audio_path)
                save_wav(wm_waveform, raw_wm_audio_path, watermark_audio["sample_rate"], wm_channels)
                
                adjusted_wm_audio_path = os.path.join(temp_dir, f"wm_audio_adjusted_{int(time.time())}.wav")
                temp_files.append(adjusted_wm_audio_path)
                
                adjust_cmd = [ffmpeg_path, "-y", "-i", raw_wm_audio_path, "-af", f"apad,atrim=0:{wm_video_duration}", "-t", str(wm_video_duration), adjusted_wm_audio_path]
                subprocess.run(adjust_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                final_audio_files.append(adjusted_wm_audio_path)

            elif wm_video_duration > 0:
                wm_audio_path = os.path.join(temp_dir, f"silent_wm_{int(time.time())}.wav")
                temp_files.append(wm_audio_path)
                subprocess.run([ffmpeg_path, "-y", "-f", "lavfi", "-i", f"anullsrc=cl=stereo:r={default_sample_rate}", "-t", str(wm_video_duration), wm_audio_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                final_audio_files.append(wm_audio_path)

            if not final_audio_files:
                self.logger.info("No audio to process.")
                return True

            # Concatenate all processed audio files
            concat_list_path = os.path.join(temp_dir, f"concat_list_{int(time.time())}.txt")
            temp_files.append(concat_list_path)
            with open(concat_list_path, 'w') as f:
                for audio_file in final_audio_files:
                    f.write(f"file '{os.path.realpath(audio_file)}'\n")

            combined_audio_path = os.path.join(temp_dir, f"combined_audio_{int(time.time())}.aac")
            temp_files.append(combined_audio_path)
            
            concat_cmd = [ffmpeg_path, "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c:a", "aac", "-b:a", "192k", combined_audio_path]
            self.logger.info(f"Running ffmpeg for audio concatenation: {' '.join(concat_cmd)}")
            result = subprocess.run(concat_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"FFmpeg audio concatenation failed: {result.stderr}")
                return False

            # Mux video and combined audio
            mux_cmd = [ffmpeg_path, "-y", "-i", video_path, "-i", combined_audio_path, "-c:v", "copy", "-c:a", "copy", temp_path]
            self.logger.info(f"Running ffmpeg to mux video and audio: {' '.join(mux_cmd)}")
            result = subprocess.run(mux_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"FFmpeg muxing failed: {result.stderr}")
                return False

            os.replace(temp_path, video_path)
            self.logger.info("Successfully added concatenated audio to video")
            self.state.audio_processed = True
            return True

        except Exception as e:
            self.logger.error(f"Audio processing failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                
    def _upload_to_cloudflare_s3(self, format, file_path, filename_prefix):
        """Upload video to Cloudflare R2 storage using boto3 after stripping metadata."""
        # Check if boto3 is available
        if not HAS_BOTO3:
            self.logger.error("boto3 not available, cannot upload to S3.")
            return {"success": False, "error": "boto3 is not installed."}

        # Check if file exists
        if not os.path.exists(file_path):
            error = f"Video file not found: {file_path}"
            self.logger.error(error)
            return {"success": False, "error": error}

        # Check if endpoint configuration is available
        if not all([S3_CONFIG.get(key) for key in ['endpoint_url', 'aws_access_key_id', 'aws_secret_access_key', 'bucket_name']]):
            error = "S3 configuration incomplete. Check your .env file."
            self.logger.error(error)
            return {"success": False, "error": error}

        # Get file extension
        _, file_extension = os.path.splitext(file_path)
        
        # Define path for the metadata-stripped video
        temp_dir = folder_paths.get_temp_directory()
        cleaned_video_path = os.path.join(temp_dir, f"{filename_prefix}_cleaned{file_extension}")

        try:
            # --- Step 1: Strip metadata using ffmpeg ---
            self.logger.info(f"Stripping metadata from {file_path}")
            strip_cmd = [
                ffmpeg_path,
                "-y",
                "-i", file_path,
                "-c", "copy",          # Copy all streams (video, audio) without re-encoding
                "-map_metadata", "-1", # Remove all metadata
                cleaned_video_path
            ]
            
            result = subprocess.run(strip_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                error_msg = f"Failed to strip metadata from video. FFmpeg stderr: {result.stderr}"
                self.logger.error(error_msg)
                # Critical failure: do not upload original file
                return {"success": False, "error": "Metadata stripping failed. Upload aborted."}
            
            self.logger.info(f"Metadata stripped successfully. Cleaned video at: {cleaned_video_path}")
            
            # --- Step 2: Upload the cleaned video ---
            upload_file_path = cleaned_video_path

            # Create S3 client
            s3_client = boto3.client(
                's3',
                endpoint_url=S3_CONFIG['endpoint_url'],
                aws_access_key_id=S3_CONFIG['aws_access_key_id'],
                aws_secret_access_key=S3_CONFIG['aws_secret_access_key'],
                region_name=S3_CONFIG.get('region_name', 'auto'),
                config=Config(signature_version='s3v4')
            )
            
            # Generate unique key for the video using the filename_prefix (generationID)
            s3_key = f"{filename_prefix}{file_extension}"
            
            # Get file size for progress tracking
            file_size = os.path.getsize(upload_file_path)
            
            # Create progress tracker
            class SimpleProgressTracker:
                def __init__(self, total_size, logger):
                    self.total_size = total_size
                    self.bytes_uploaded = 0
                    self.last_update = 0
                    self.logger = logger
                
                def __call__(self, bytes_amount):
                    self.bytes_uploaded += bytes_amount
                    progress = int((self.bytes_uploaded / self.total_size) * 100)
                    
                    if progress >= self.last_update + 10 or progress == 100:
                        self.last_update = progress
                        message = f"Uploaded {self.bytes_uploaded} of {self.total_size} bytes ({progress}%)"
                        self.logger.info(message)
            
            progress_callback = SimpleProgressTracker(file_size, self.logger)
            
            self.logger.info(f"Starting upload of cleaned video to {S3_CONFIG['endpoint_url']} bucket {S3_CONFIG['bucket_name']}")
            
            # Upload the cleaned file to S3
            s3_client.upload_file(
                upload_file_path, 
                S3_CONFIG['bucket_name'], 
                s3_key,
                Callback=progress_callback,
                ExtraArgs={
                    'ContentType': 'video/mp4' if format.startswith('video/') else 'image/' + format.split('/')[1]
                }
            )
            
            # Generate public URL
            public_url = f"https://exports.oshified.com/{s3_key}"
            
            self.state.upload_attempted = True
            self.state.upload_successful = True
            
            self.logger.info(f"Upload successful: {public_url}")
            
            # --- Step 3: Notify backend ---
            try:
                backend_url = os.getenv("OSHIFIED_BACKEND_URL", "https://oshified-backend.your-subdomain.workers.dev")
                video_ready_url = f"{backend_url}/api/runpod/videoReady"
                
                self.logger.info(f"Notifying backend at: {video_ready_url}")
                
                response = requests.post(
                    video_ready_url,
                    json={'generationID': filename_prefix, 'videoUrl': public_url},
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                
                if response.status_code == 200:
                    self.logger.info("Successfully notified backend about video completion")
                else:
                    self.logger.error(f"Backend notification failed: {response.status_code} {response.text}")
                    
            except Exception as e:
                self.logger.error(f"Failed to notify backend: {str(e)}")
            
            return {"success": True, "url": public_url, "key": s3_key}
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Upload process failed: {error_msg}")
            self.state.upload_attempted = True
            return {"success": False, "error": error_msg}
        
        finally:
            # --- Step 4: Clean up the temporary cleaned file ---
            if os.path.exists(cleaned_video_path):
                try:
                    os.remove(cleaned_video_path)
                    self.logger.info(f"Successfully removed temporary cleaned file: {cleaned_video_path}")
                except OSError as e:
                    self.logger.error(f"Error removing temporary file {cleaned_video_path}: {e}")

class OshifiedAudioImporter:
    """
    Downloads YouTube video and extracts audio for use in Oshified workflows.
    Uses yt-dlp first, falls back to RapidAPI if blocked by YouTube bot detection.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {
                    "multiline": False,
                    "default": "https://www.youtube.com/watch?v="
                }),
                "max_duration": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 60,
                    "step": 1
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "import_audio"
    CATEGORY = "Oshified 🌸"
    
    @staticmethod
    def _extract_video_id(url):
        """Extract YouTube video ID from URL."""
        # Pattern matches various YouTube URL formats
        pattern = r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return None
    
    def _download_via_api(self, video_id, max_duration):
        """
        Download audio via RapidAPI with key rotation.
        Falls back through multiple keys if rate limited.
        """
        # Get API keys from environment
        api_keys = [
            os.getenv("RAPIDAPI_KEY_1"),
            os.getenv("RAPIDAPI_KEY_2"),
            os.getenv("RAPIDAPI_KEY_3"),
            os.getenv("RAPIDAPI_KEY_4"),
        ]
        
        # Filter out empty keys
        valid_keys = [k for k in api_keys if k and k.strip()]
        
        if not valid_keys:
            raise Exception("No RapidAPI keys configured. Please add RAPIDAPI_KEY_1 to .env")
        
        print(f"[OshifiedAudioImporter] Attempting API download with {len(valid_keys)} available key(s)")
        
        for i, api_key in enumerate(valid_keys, 1):
            try:
                print(f"[OshifiedAudioImporter] Trying API key #{i}...")
                
                # Make API request
                response = requests.get(
                    f"https://youtube-mp36.p.rapidapi.com/dl?id={video_id}",
                    headers={
                        "X-RapidAPI-Host": "youtube-mp36.p.rapidapi.com",
                        "X-RapidAPI-Key": api_key
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    print(f"[OshifiedAudioImporter] API key #{i} returned status {response.status_code}")
                    continue
                
                data = response.json()
                status = data.get('status', 'fail')
                
                # Handle processing status with polling
                max_polls = 30  # Max 30 seconds of polling
                poll_count = 0
                
                while status == 'processing' and poll_count < max_polls:
                    print(f"[OshifiedAudioImporter] Conversion in progress... {poll_count+1}s")
                    time.sleep(1)
                    poll_count += 1
                    
                    # Poll again
                    response = requests.get(
                        f"https://youtube-mp36.p.rapidapi.com/dl?id={video_id}",
                        headers={
                            "X-RapidAPI-Host": "youtube-mp36.p.rapidapi.com",
                            "X-RapidAPI-Key": api_key
                        },
                        timeout=30
                    )
                    data = response.json()
                    status = data.get('status', 'fail')
                
                if status == 'ok':
                    download_url = data.get('link')
                    if not download_url:
                        raise Exception("API returned 'ok' but no download link")
                    
                    print(f"[OshifiedAudioImporter] API download successful with key #{i}")
                    
                    # Download MP3 file
                    temp_dir = folder_paths.get_temp_directory()
                    mp3_path = os.path.join(temp_dir, f'oshified_api_{video_id}.mp3')
                    
                    mp3_response = requests.get(download_url, timeout=60)
                    with open(mp3_path, 'wb') as f:
                        f.write(mp3_response.content)
                    
                    print(f"[OshifiedAudioImporter] Downloaded MP3 to: {mp3_path}")
                    
                    # Convert MP3 to WAV using ffmpeg
                    wav_path = os.path.join(temp_dir, f'oshified_api_{video_id}.wav')
                    convert_cmd = [
                        ffmpeg_path, '-y', '-i', mp3_path,
                        '-t', str(max_duration),  # Trim to max duration
                        '-ar', '48000',  # Resample to 48kHz
                        '-ac', '1',  # Force mono
                        wav_path
                    ]
                    
                    result = subprocess.run(convert_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise Exception(f"FFmpeg conversion failed: {result.stderr}")
                    
                    # Clean up MP3
                    if os.path.exists(mp3_path):
                        os.remove(mp3_path)
                    
                    return wav_path  # Return path to WAV file
                    
                elif status == 'fail':
                    print(f"[OshifiedAudioImporter] API key #{i} conversion failed")
                    continue
                else:
                    print(f"[OshifiedAudioImporter] API key #{i} timed out (status: {status})")
                    continue
                    
            except Exception as e:
                print(f"[OshifiedAudioImporter] API key #{i} error: {str(e)}")
                continue
        
        # All keys exhausted
        raise Exception(
            "YouTube download blocked and all API keys exhausted or failed.\n"
            "Please upload your audio/video file directly."
        )
    
    def import_audio(self, video_url, max_duration=20):
        """
        Download YouTube video and extract audio.
        Uses yt-dlp first, falls back to RapidAPI if bot detected.
        
        Args:
            video_url: YouTube URL
            max_duration: Maximum duration in seconds (default: 20 for Oshified)
        
        Returns:
            Tuple containing AUDIO dict: {'waveform': torch.Tensor, 'sample_rate': int}
        """
        if not ffmpeg_path:
            raise Exception("ffmpeg is required for audio extraction")
        
        temp_dir = folder_paths.get_temp_directory()
        wav_path = None
        
        # Extract video ID for fallback
        video_id = self._extract_video_id(video_url)
        if not video_id:
            raise Exception(f"Could not extract video ID from URL: {video_url}")
        
        print(f"[OshifiedAudioImporter] Video ID: {video_id}")
        print(f"[OshifiedAudioImporter] Max duration: {max_duration}s")
        
        # Try yt-dlp first
        try:
            if not HAS_YTDLP:
                raise Exception("yt-dlp not available, trying API...")
            
            print(f"[OshifiedAudioImporter] Attempting direct download with yt-dlp...")
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(temp_dir, f'oshified_yt_{video_id}.%(ext)s'),
                'noplaylist': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': 'best',
                }],
                'quiet': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                duration = info.get('duration', 0)
                print(f"[OshifiedAudioImporter] Title: {info.get('title', 'Unknown')}")
                print(f"[OshifiedAudioImporter] Duration: {duration}s")
                
                # Download
                ydl.download([video_url])
                
                wav_path = os.path.join(temp_dir, f'oshified_yt_{video_id}.wav')
                
                if not os.path.exists(wav_path):
                    raise Exception("WAV file not created")
                
                print(f"[OshifiedAudioImporter] yt-dlp download successful")
                
                # Trim if needed
                if duration > max_duration:
                    trimmed_path = os.path.join(temp_dir, f'oshified_yt_{video_id}_trimmed.wav')
                    trim_cmd = [ffmpeg_path, '-y', '-i', wav_path, '-t', str(max_duration), '-c', 'copy', trimmed_path]
                    result = subprocess.run(trim_cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        os.remove(wav_path)
                        wav_path = trimmed_path
                        print(f"[OshifiedAudioImporter] Trimmed to {max_duration}s")
                
        except Exception as ytdlp_error:
            error_str = str(ytdlp_error)
            
            # Check if it's a bot detection error
            if "not a bot" in error_str or "Sign in to confirm" in error_str:
                print(f"[OshifiedAudioImporter] YouTube bot detection triggered, falling back to API...")
                
                # Try API fallback
                try:
                    wav_path = self._download_via_api(video_id, max_duration)
                    print(f"[OshifiedAudioImporter] API fallback successful")
                except Exception as api_error:
                    # API also failed
                    raise Exception(
                        f"YouTube download failed (bot detection) and API fallback also failed.\n"
                        f"Please upload your audio/video file directly.\n"
                        f"API error: {str(api_error)}"
                    )
            else:
                # Other yt-dlp error, no fallback
                raise Exception(f"YouTube download failed: {error_str}")
        
        # At this point, wav_path should be set (either from yt-dlp or API)
        if not wav_path or not os.path.exists(wav_path):
            raise Exception("No audio file was created")
        
        # Load WAV file into ComfyUI AUDIO format
        try:
            with wave.open(wav_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                n_frames = wav_file.getnframes()
                audio_data = wav_file.readframes(n_frames)
            
            # Convert to numpy array
            if wav_file.getsampwidth() == 2:  # 16-bit
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
            else:
                raise Exception(f"Unsupported sample width: {wav_file.getsampwidth()}")
            
            # Convert to float32 normalized to [-1, 1]
            audio_np = audio_np.astype(np.float32) / 32767.0
            
            # De-interleave channels correctly
            if n_channels > 1:
                audio_np = audio_np.reshape((-1, n_channels)).T
                audio_np = np.mean(audio_np, axis=0, keepdims=True)
                print(f"[OshifiedAudioImporter] Converted {n_channels} channels to mono")
            else:
                audio_np = audio_np.reshape((1, -1))
            
            # Convert to torch tensor with batch dimension
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
            
            print(f"[OshifiedAudioImporter] Loaded audio: mono, {sample_rate}Hz, {n_frames} frames")
            print(f"[OshifiedAudioImporter] Tensor shape: {audio_tensor.shape}")
            
            # Clean up
            if os.path.exists(wav_path):
                os.remove(wav_path)
                print(f"[OshifiedAudioImporter] Cleaned up temp file")
            
            return ({'waveform': audio_tensor, 'sample_rate': sample_rate},)
            
        except Exception as e:
            error_msg = f"Audio loading failed: {str(e)}"
            print(f"[OshifiedAudioImporter] ERROR: {error_msg}")
            raise Exception(error_msg)


class OshifiedImageSaver:
    """
    Image saver node for Oshified - saves still image to R2 and notifies backend.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Oshified"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "Oshified 🌸"
    FUNCTION = "save_image"

    def __init__(self):
        self.logger = VideoProcessingLogger("OshifiedImageSaver")
    
    def save_image(self, images, filename_prefix="Oshified", quality=95):
        """
        Save first image as JPEG to R2 and notify backend.
        
        Args:
            images: Input images tensor
            filename_prefix: Filename prefix (generationID)
            quality: JPEG quality (1-100)
        
        Returns:
            Dictionary with UI information
        """
        # Get first image
        if isinstance(images, torch.Tensor):
            if images.size(0) == 0:
                return {"ui": {"error": "No images provided"}}
            first_image = images[0]
        else:
            if len(images) == 0:
                return {"ui": {"error": "No images provided"}}
            first_image = images[0]
        
        # Convert to PIL Image and save as JPEG
        try:
            # Convert tensor to bytes (0-255 range)
            image_np = tensor_to_bytes(first_image)
            pil_image = Image.fromarray(image_np)
            
            # Save to temp file first
            temp_dir = folder_paths.get_temp_directory()
            temp_path = os.path.join(temp_dir, f"{filename_prefix}_temp.jpg")
            
            pil_image.save(temp_path, format='JPEG', quality=quality, optimize=True)
            
            self.logger.info(f"Saved temporary JPEG to {temp_path}")
            
            # Upload to R2
            if not HAS_BOTO3:
                self.logger.error("boto3 not available, cannot upload to S3.")
                return {"ui": {"error": "boto3 is not installed, cannot upload image."}}
            
            # Check S3 config
            if not all([S3_CONFIG.get(key) for key in ['endpoint_url', 'aws_access_key_id', 'aws_secret_access_key', 'bucket_name']]):
                error = "S3 configuration incomplete. Check your .env file."
                self.logger.error(error)
                return {"ui": {"error": error}}
            
            # Create S3 client
            s3_client = boto3.client(
                's3',
                endpoint_url=S3_CONFIG['endpoint_url'],
                aws_access_key_id=S3_CONFIG['aws_access_key_id'],
                aws_secret_access_key=S3_CONFIG['aws_secret_access_key'],
                region_name=S3_CONFIG.get('region_name', 'auto'),
                config=Config(signature_version='s3v4')
            )
            
            # Upload with filename {generationID}-still.jpg
            s3_key = f"{filename_prefix}-still.jpg"
            
            s3_client.upload_file(
                temp_path,
                S3_CONFIG['bucket_name'],
                s3_key,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            
            # Generate public URL
            public_url = f"https://exports.oshified.com/{s3_key}"
            
            self.logger.info(f"Image uploaded successfully: {public_url}")
            
            # Notify backend
            try:
                backend_url = os.getenv("OSHIFIED_BACKEND_URL", "https://oshified-backend.your-subdomain.workers.dev")
                image_ready_url = f"{backend_url}/api/runpod/imageReady"
                
                self.logger.info(f"Notifying backend at: {image_ready_url}")
                
                response = requests.post(
                    image_ready_url,
                    json={'generationID': filename_prefix, 'imageUrl': public_url},
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                
                if response.status_code == 200:
                    self.logger.info("Successfully notified backend about image completion")
                else:
                    self.logger.error(f"Backend notification failed: {response.status_code} {response.text}")
                    
            except Exception as e:
                self.logger.error(f"Failed to notify backend: {str(e)}")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return {"ui": {"response": {"success": True, "url": public_url, "key": s3_key}}}
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Image save/upload failed: {error_msg}")
            return {"ui": {"error": f"Image processing failed: {error_msg}"}}


# Export the node classes for ComfyUI to find them
NODE_CLASS_MAPPINGS = {
    "OshifiedVideoSaver": OshifiedVideoSaver,
    "OshifiedImageSaver": OshifiedImageSaver,
    "OshifiedAudioImporter": OshifiedAudioImporter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OshifiedVideoSaver": "Oshified Video Saver 🌸",
    "OshifiedImageSaver": "Oshified Image Saver 🌸",
    "OshifiedAudioImporter": "Oshified Audio Importer 🌸"
}
