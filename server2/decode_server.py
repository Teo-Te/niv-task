from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torchaudio
import numpy as np
import tempfile
import os
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the encodec path to Python path
sys.path.append('/home/arteofejzo/Documents/MadTask/encodec')

from encodec import EncodecModel
from encodec.utils import save_audio

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
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)

# Create a directory for decoded files
DECODED_FILES_DIR = Path("/home/arteofejzo/Documents/MadTask/decoded_audio")
DECODED_FILES_DIR.mkdir(exist_ok=True)

class EncodedData(BaseModel):
    encoded_data: List[Dict[str, Any]]
    sample_rate: Optional[int] = 24000
    channels: Optional[int] = 1

@app.post("/decode")
async def decode_audio(data: EncodedData):
    try:
        logger.info(f"Received {len(data.encoded_data)} encoded frames")
        
        # Reconstruct encoded frames
        frames = []
        for i, frame_data in enumerate(data.encoded_data):
            codes = torch.tensor(frame_data['codes'], dtype=torch.int)
            scale = torch.tensor([frame_data['scale']], dtype=torch.float32) if frame_data['scale'] is not None else None
            logger.info(f"Frame {i}: codes_shape={codes.shape}, scale={scale}")
            frames.append((codes, scale))

        # Decode with EnCodec
        logger.info("Decoding with EnCodec...")
        with torch.no_grad():
            decoded_wav = model.decode(frames)
        
        logger.info(f"Decoded audio shape: {decoded_wav.shape}")
        
        # Remove batch dimension and ensure mono
        if decoded_wav.dim() == 3:
            decoded_wav = decoded_wav.squeeze(0)  # Remove batch dim
        if decoded_wav.shape[0] > 1:
            decoded_wav = decoded_wav.mean(dim=0, keepdim=True)  # Convert to mono

        # Generate unique filename
        import time
        filename = f"decoded_audio_{int(time.time())}.wav"
        output_path = DECODED_FILES_DIR / filename

        # Convert to 22050 Hz as requested
        target_sr = 22050
        current_sr = model.sample_rate
        
        if current_sr != target_sr:
            logger.info(f"Resampling from {current_sr}Hz to {target_sr}Hz")
            resampler = torchaudio.transforms.Resample(current_sr, target_sr)
            decoded_wav = resampler(decoded_wav)

        # Normalize audio to prevent clipping
        max_val = decoded_wav.abs().max()
        if max_val > 0:
            decoded_wav = decoded_wav / max_val * 0.9  # Scale to 90% to prevent clipping

        logger.info(f"Final audio shape: {decoded_wav.shape}, max_val: {decoded_wav.abs().max()}")

        # Save the decoded audio
        torchaudio.save(str(output_path), decoded_wav, target_sr)
        logger.info(f"Saved decoded audio to: {output_path}")

        return {
            "status": "success",
            "message": f"Audio decoded and saved successfully",
            "filename": filename,
            "download_url": f"http://localhost:8001/download/{filename}",
            "output_path": str(output_path),
            "sample_rate": target_sr
        }

    except Exception as e:
        logger.error(f"Decode error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = DECODED_FILES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='audio/wav'
    )

@app.get("/files")
async def list_files():
    """List all decoded audio files"""
    files = []
    for file_path in DECODED_FILES_DIR.glob("*.wav"):
        files.append({
            "filename": file_path.name,
            "download_url": f"http://localhost:8001/download/{file_path.name}",
            "size": file_path.stat().st_size,
            "created": file_path.stat().st_mtime
        })
    return {"files": files}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)