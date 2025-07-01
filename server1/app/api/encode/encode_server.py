from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
import tempfile
import subprocess
from pathlib import Path
import sys
# needed the error trace too ...
import traceback
import logging

# Had to use logging to debug some issues, not removing it now tho. 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the encodec path from my system to Python path
sys.path.append('/home/arteofejzo/Documents/MadTask/encodec')

try:
    from encodec import EncodecModel
    logger.info("Successfully imported EnCodec modules")
except ImportError as e:
    logger.error(f"Failed to import EnCodec: {e}")
    raise

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load EnCodec model
try:
    logger.info("Loading EnCodec model...")
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    logger.info("EnCodec model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load EnCodec model: {e}")
    raise

def convert_to_24khz_with_ffmpeg(input_path, output_path):
    """Convert audio to 24kHz mono using FFmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-ar', '24000',  # Sample rate 24kHz
            '-ac', '1',      # Mono
            '-y',            # Overwrite output
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")
        logger.info(f"Converted audio to 24kHz: {output_path}")
        return True
    except Exception as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        return False

@app.post("/encode")
async def encode_audio(audio: UploadFile = File(...)):
    tmp_file_path = None
    converted_file_path = None
    try:
        logger.info(f"Received audio file: {audio.filename}, content_type: {audio.content_type}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            logger.info(f"Saved temporary file: {tmp_file_path}")

        # Convert to 24kHz using FFmpeg
        converted_file_path = tmp_file_path.replace('.wav', '_24khz.wav')
        if not convert_to_24khz_with_ffmpeg(tmp_file_path, converted_file_path):
            raise HTTPException(status_code=500, detail="Failed to convert audio to 24kHz")

        # Load the converted audio
        logger.info("Loading converted audio file...")
        wav, sr = torchaudio.load(converted_file_path)
        logger.info(f"Loaded audio: shape={wav.shape}, sample_rate={sr}")
        
        # Ensure correct format for EnCodec
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # Add channel dimension
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # Convert to mono
        
        # Add batch dimension
        wav = wav.unsqueeze(0)
        logger.info(f"Preprocessed audio: shape={wav.shape}")

        # Encode with EnCodec
        logger.info("Encoding with EnCodec...")
        with torch.no_grad():
            encoded_frames = model.encode(wav)
        logger.info(f"Encoded {len(encoded_frames)} frames")
        
        # Convert to serializable format
        logger.info("Converting to serializable format...")
        encoded_data = []
        for i, (codes, scale) in enumerate(encoded_frames):
            logger.info(f"Frame {i}: codes_shape={codes.shape}, scale={scale}")
            frame_data = {
                'codes': codes.cpu().numpy().tolist(),
                'scale': scale.cpu().item() if scale is not None else None
            }
            encoded_data.append(frame_data)

        logger.info("Encoding completed successfully")
        return {
            "status": "success",
            "encoded_data": encoded_data,
            "sample_rate": 24000,
            "channels": 1
        }

    except Exception as e:
        logger.error(f"Error during encoding: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")
    
    finally:
        # Clean up temp files
        for file_path in [tmp_file_path, converted_file_path]:
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()
                logger.info(f"Cleaned up: {file_path}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")