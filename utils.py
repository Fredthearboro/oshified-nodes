import os
import sys
import logging
import torch
import re
import subprocess
from collections.abc import Mapping

# Setup path for ffmpeg
ffmpeg_path = "ffmpeg"
paths_to_try = [
    "ffmpeg",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ffmpeg", "ffmpeg"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ffmpeg", "bin", "ffmpeg"),
]

for path in paths_to_try:
    try:
        if sys.platform == "win32":
            path += ".exe"
        if os.path.isfile(path) or os.system(f"which {path} > /dev/null 2>&1") == 0:
            ffmpeg_path = path
            break
    except Exception as e:
        print(f"[VideoHelperSuite] Error checking ffmpeg path {path}: {e}")

if ffmpeg_path == "ffmpeg" and sys.platform == "win32" and not os.system("where ffmpeg > nul 2>&1") == 0:
    ffmpeg_path = None
elif ffmpeg_path == "ffmpeg" and sys.platform != "win32" and os.system("which ffmpeg > /dev/null 2>&1") != 0:
    ffmpeg_path = None

# Configure logging
logger = logging.getLogger("VideoHelperSuite")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

def round_up_to_even(x):
    """Round up to the nearest even number."""
    return (x + 1) & ~1

def tensor_to_bytes(frame, bits=8):
    """Convert tensor frame to bytes."""
    if frame.shape[2] == 4:
        # RGBA
        return (frame[:, :, :3] * 255).round().to(torch.uint8).cpu().numpy()

    if bits == 16:
        # RGB16
        return (frame * 65535).round().to(torch.int16).cpu().numpy()
    else:
        # Default to RGB8
        return (frame * 255).round().to(torch.uint8).cpu().numpy()

def tensor_to_shorts(frame):
    """Convert tensor frame to short integers."""
    return tensor_to_bytes(frame, bits=16)

# Constants for encoding arguments
ENCODE_ARGS = ("utf-8", 'backslashreplace')

def get_audio(file, start_time=0, duration=0):
    """
    Extract audio from a file using ffmpeg.
    
    Args:
        file: Path to the audio/video file
        start_time: Start time in seconds
        duration: Duration in seconds
        
    Returns:
        Dictionary with 'waveform' tensor and 'sample_rate' integer
    """
    args = [ffmpeg_path, "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    try:
        res = subprocess.run(args + ["-f", "f32le", "-"],
                            capture_output=True, check=True)
        audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        match = re.search(', (\\d+) Hz, (\\w+), ', res.stderr.decode(*ENCODE_ARGS))
    except subprocess.CalledProcessError as e:
        raise Exception(f"VHS failed to extract audio from {file}:\n" \
                + e.stderr.decode(*ENCODE_ARGS))
    if match:
        ar = int(match.group(1))
        # Handle channel types
        ac = {"mono": 1, "stereo": 2}[match.group(2)]
    else:
        ar = 44100
        ac = 2
    audio = audio.reshape((-1, ac)).transpose(0, 1).unsqueeze(0)
    return {'waveform': audio, 'sample_rate': ar}
