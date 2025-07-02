from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import torch
from pathlib import Path
import sys

# Appending EnCodec so as to get the model
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

# It had two models, as per the task, I used the 24kHz model
model = EncodecModel.encodec_model_24khz()
# To my understanding, this determines the quality of the audio encoding
# there were these options: 1.5, 3, 6, 12 or 24 kbps
model.set_target_bandwidth(6.0)

@app.post("/convert-to-onnx")
async def convert_model_to_onnx():
    try:
        # Set the path within the App directory of next.js for the model to be stored
        onnx_path = Path("./models/encodec_encoder.onnx")
        onnx_path.parent.mkdir(exist_ok=True)
        
        # If the model has been converted before, Then simply skip it and return success
        if onnx_path.exists():
            return {
                "status": "success", 
                "message": "ONNX model already exists",
            }
        # This class is a wrapper around the EnCodec model to simplify the ONNX export
        # EnCodec has a lot nested objects and this just flattens them out.
        class EnCodecSimpleWrapper(torch.nn.Module):
            def __init__(self, encodec_model):
                super().__init__()
                self.encodec_model = encodec_model
            
            def forward(self, x):
                with torch.no_grad():
                    encoded_frames = self.encodec_model.encode(x)
                    codes, scale = encoded_frames[0]
                    
                    codes_flat = codes.flatten().float()
                    scale_out = scale.float() if scale is not None else torch.ones(1, dtype=torch.float32, device=codes.device)
                    
                    n_q = torch.tensor(codes.shape[0], dtype=torch.float32)
                    channels = torch.tensor(codes.shape[1], dtype=torch.float32)
                    time_steps = torch.tensor(codes.shape[2], dtype=torch.float32)
                    
                    return codes_flat, scale_out, n_q, channels, time_steps
        
        wrapper = EnCodecSimpleWrapper(model)
        wrapper.eval()
        
        chunk_size = 45000
        dummy_input = torch.randn(1, 1, chunk_size)
        
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['audio'],
            output_names=['codes', 'scale', 'n_q', 'channels', 'time_steps'],
            verbose=False
        )
        
        model_size_mb = onnx_path.stat().st_size / 1024 / 1024
        
        return {
            "status": "success",
            "message": "EnCodec encoder converted to ONNX successfully",
            "model_size_mb": model_size_mb,
            "chunk_size": chunk_size
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-onnx-model")
async def get_onnx_model():
    onnx_path = Path("./models/encodec_encoder.onnx")
    if onnx_path.exists():
        return {"status": "exists", "path": str(onnx_path)}
    else:
        raise HTTPException(status_code=404, detail="ONNX model not found")

@app.get("/model/encodec_encoder.onnx")
async def serve_onnx_model():
    onnx_path = Path("./models/encodec_encoder.onnx")
    if not onnx_path.exists():
        raise HTTPException(status_code=404, detail="ONNX model not found")
    
    return FileResponse(
        path=str(onnx_path),
        filename="encodec_encoder.onnx",
        media_type="application/octet-stream"
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)