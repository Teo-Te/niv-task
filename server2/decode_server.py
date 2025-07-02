from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torchaudio
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional

sys.path.append('/home/arteofejzo/Documents/MadTask/encodec')

from encodec import EncodecModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)

DECODED_FILES_DIR = Path("/home/arteofejzo/Documents/MadTask/decoded_audio")
DECODED_FILES_DIR.mkdir(exist_ok=True)

class EncodedData(BaseModel):
    encoded_data: List[Dict[str, Any]]
    sample_rate: Optional[int] = 24000
    channels: Optional[int] = 1

@app.post("/decode")
async def decode_audio(data: EncodedData):
    try:
        # Get the client-encoded chunks
        frame_data = data.encoded_data[0]
        chunks = frame_data['chunks']
        
        # Process each chunk
        frames = []
        for chunk in chunks:
            codes_flat = chunk['codes']
            scale_value = chunk['scale']
            structure = chunk['structure']
            
            n_q = int(structure['n_q'])
            channels = int(structure['channels'])
            time_steps = int(structure['time_steps'])
            
            # Reshape codes
            codes_tensor = torch.tensor(codes_flat, dtype=torch.float32).round().long()
            codes_reshaped = codes_tensor.view(n_q, channels, time_steps)
            codes_reshaped = torch.clamp(codes_reshaped, 0, 1023)
            
            scale_tensor = torch.tensor([scale_value], dtype=torch.float32)
            frames.append((codes_reshaped, scale_tensor))
        
        # Decode each frame and concatenate
        decoded_chunks = []
        with torch.no_grad():
            for frame in frames:
                decoded_chunk = model.decode([frame])
                decoded_chunks.append(decoded_chunk)
        
        # Concatenate all chunks
        decoded_wav = torch.cat(decoded_chunks, dim=-1)
        
        # Remove batch dimension
        if decoded_wav.dim() == 3:
            decoded_wav = decoded_wav.squeeze(0)
        
        # Convert to mono if needed
        if decoded_wav.dim() == 2 and decoded_wav.shape[0] > 1:
            decoded_wav = decoded_wav.mean(dim=0, keepdim=True)
        elif decoded_wav.dim() == 1:
            decoded_wav = decoded_wav.unsqueeze(0)

        # Generate filename
        import time
        filename = f"decoded_audio_{int(time.time())}.wav"
        output_path = DECODED_FILES_DIR / filename

        # Resample to 22050 Hz
        target_sr = 22050
        current_sr = model.sample_rate
        
        if current_sr != target_sr:
            resampler = torchaudio.transforms.Resample(current_sr, target_sr)
            decoded_wav = resampler(decoded_wav)

        # Normalize
        max_val = decoded_wav.abs().max()
        if max_val > 0:
            decoded_wav = decoded_wav / max_val * 0.9

        # Save
        torchaudio.save(str(output_path), decoded_wav, target_sr)

        return {
            "status": "success",
            "message": f"Audio decoded successfully from {len(frames)} chunks",
            "filename": filename,
            "download_url": f"http://localhost:8001/download/{filename}",
            "sample_rate": target_sr,
            "total_chunks": len(frames)
        }

    except Exception as e:
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "decode_server"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)