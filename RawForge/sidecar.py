import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path

from RawForge.main import run_pipeline

app = FastAPI(title="RawForge Sidecar API")

class ProcessRequest(BaseModel):
    model_names: List[str] = Field(..., description="List of models to run in sequence")
    in_file: str
    out_file: str
    conditioning: Optional[List[int]] = None
    dims: Optional[List[int]] = None
    lumi: float = 0.0
    chroma: float = 0.0
    tile_size: int = 256
    tile_overlap: float = 0.25
    clip_highlights: bool = False
    affine: bool = False
    save_as_cfa: bool = False
    use_onnx: bool = False
    device: Optional[str] = None

@app.post("/process")
async def process_image(req: ProcessRequest):
    """
    Wraps the run_pipeline logic into a REST endpoint.
    """
    if not Path(req.in_file).exists():
        raise HTTPException(status_code=404, detail=f"Input file not found: {req.in_file}")

    try:
        model_str = ",".join(req.model_names)
        
        cond_str = ",".join(map(str, req.conditioning)) if req.conditioning else None
        run_pipeline(
            model_names=model_str,
            in_file=req.in_file,
            out_file=req.out_file,
            conditioning_str=cond_str,
            dims=req.dims,
            cfa=req.save_as_cfa,
            tile_size=req.tile_size,
            tile_overlap=req.tile_overlap,
            lumi=req.lumi,
            chroma=req.chroma,
            clip_highlights=req.clip_highlights,
            affine=req.affine,
            use_onnx=req.use_onnx,
            device=req.device,
            verbose=0,  
            disable_tqdm=True
        )

        return {
            "status": "success", 
            "output": req.out_file,
            "runtime_params": {
                "models": req.model_names,
                "device": req.device or "auto"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "online"}

def main():
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)