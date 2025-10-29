# Oshified Custom Nodes for ComfyUI

Custom ComfyUI nodes for the Oshified VTuber transformation service. These nodes handle saving and uploading generated images and videos to Cloudflare R2 storage.

## Nodes Included

### 1. Oshified Video Saver 🌸
Saves animated video with audio, watermarks, and uploads to R2.

**Inputs**:
- `images` (IMAGE) - Generated video frames
- `frame_rate` (FLOAT) - Output frame rate (default: 8)
- `filename_prefix` (STRING) - Filename prefix (generationID)
- `format` (STRING) - Video format (default: video/h264-mp4)
- `save_output` (BOOLEAN) - Whether to save locally (default: True)
- `audio` (AUDIO, optional) - Audio to include with video
- `append_watermark` (IMAGE, optional) - Frames to append after main video
- `watermark_audio` (AUDIO, optional) - Audio for appended watermark
- `corner_watermark` (IMAGE, optional) - Watermark to overlay in corner
- `corner_watermark_mask` (MASK, optional) - Mask for corner watermark
- `corner_opacity` (FLOAT, optional) - Corner watermark opacity (0.0-1.0)

**Outputs**:
- Uploads video to R2 as `{generationID}.mp4`
- Notifies backend at `/api/runpod/videoReady`
- Returns `OSHIFIED_FILENAMES` type

### 2. Oshified Image Saver ��
Saves still image (face guess) as JPEG and uploads to R2.

**Inputs**:
- `images` (IMAGE) - Generated images (uses first image)
- `filename_prefix` (STRING) - Filename prefix (generationID)
- `quality` (INT) - JPEG quality 1-100 (default: 95)

**Outputs**:
- Uploads image to R2 as `{generationID}-still.jpg`
- Notifies backend at `/api/runpod/imageReady`
- No return type (OUTPUT_NODE only)

## Installation

### 1. Add to ComfyUI Custom Nodes

```bash
cd ComfyUI/custom_nodes
git clone <your-repo-url> oshified-node
# OR copy the oshified-node directory into ComfyUI/custom_nodes/
```

### 2. Install Dependencies

```bash
cd oshified-node
pip install -r requirements.txt
```

**Note**: If you're using ComfyUI's portable installation, you may need to install to the correct Python environment.

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your actual credentials
```

Required configuration in `.env`:

```bash
# Your Cloudflare Account ID (found in R2 dashboard)
S3_ENDPOINT_URL=https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com

# R2 API Token (create in Cloudflare dashboard)
S3_ACCESS_KEY_ID=your_access_key_here
S3_SECRET_ACCESS_KEY=your_secret_key_here

# R2 Bucket Name
S3_BUCKET_NAME=oshified-exports

# Cloudflare Workers backend URL (after deploying backend)
OSHIFIED_BACKEND_URL=https://oshified-backend.your-subdomain.workers.dev
```

### 4. Get Cloudflare Credentials

1. **R2 API Token**:
   - Go to Cloudflare Dashboard → R2 → Manage R2 API Tokens
   - Create API Token with "Object Read and Write" permissions
   - Copy Access Key ID and Secret Access Key

2. **Account ID**:
   - Found in R2 dashboard URL or Account settings
   - Format: `YOUR_ACCOUNT_ID.r2.cloudflarestorage.com`

3. **Bucket**:
   - Bucket `oshified-exports` should already exist
   - If not, create it in Cloudflare R2 dashboard

### 5. Restart ComfyUI

Restart ComfyUI for the nodes to appear. They will be in the "Oshified 🌸" category.

## Usage in Workflow

### Video Output Node

Connect to the end of your video generation workflow:
1. Connect generated frames to `images` input
2. Connect audio (from video load node) to `audio` input
3. Set `filename_prefix` to the generationID (use a string input)
4. Connect watermark inputs if you have Oshified watermarks
5. The node will:
   - Process video with watermarks
   - Add audio
   - Upload to R2
   - Notify backend when complete

### Image Output Node

Connect to save the still face image:
1. Connect generated face image to `images` input
2. Set `filename_prefix` to the same generationID
3. The node will:
   - Convert to JPEG
   - Upload to R2 as `{generationID}-still.jpg`
   - Notify backend

## Backend Integration

Both nodes notify the Oshified backend when complete:

**Image Ready**:
```
POST {OSHIFIED_BACKEND_URL}/api/runpod/imageReady
{
  "generationID": "uuid",
  "imageUrl": "https://exports.oshified.com/uuid-still.jpg"
}
```

**Video Ready**:
```
POST {OSHIFIED_BACKEND_URL}/api/runpod/videoReady
{
  "generationID": "uuid", 
  "videoUrl": "https://exports.oshified.com/uuid.mp4"
}
```

The backend tracks both notifications and sends a combined `resultsReady` event to the frontend when both image and video are uploaded.

## Troubleshooting

### boto3 not found
```bash
pip install boto3
```

### S3 Upload Fails
- Check `.env` file has correct credentials
- Verify bucket `oshified-exports` exists
- Check Account ID is correct in endpoint URL
- Verify API token has Read and Write permissions

### Backend Notification Fails
- Check `OSHIFIED_BACKEND_URL` is correct in `.env`
- Verify backend is deployed and running
- Check backend logs for incoming requests

## File Structure

```
oshified-node/
├── __init__.py           # Node registration
├── nodes.py              # OshifiedVideoSaver + OshifiedImageSaver
├── utils.py              # Helper functions
├── requirements.txt      # Python dependencies
├── .env.example          # Configuration template
├── .env                  # Your actual config (create this)
└── video_formats/        # Video format definitions
    └── h264-mp4.json
```

## Development

To modify the nodes:
1. Edit `nodes.py`
2. Restart ComfyUI to reload changes
3. Test in a workflow

## License

Same as Oshified project.
