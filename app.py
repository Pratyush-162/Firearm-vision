import argparse
from contextlib import asynccontextmanager
from pathlib import Path
import threading
import json

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.stream_manager import RTSPStreamManager

# Global stream manager reference
stream_manager: RTSPStreamManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global stream_manager
    model_path = getattr(app.state, "model_path", "runs/detect/runs/universal-weapon/weights/best.pt")
    initial_source = getattr(app.state, "source", "0")
    confidence = getattr(app.state, "confidence", 0.35)
    frames = getattr(app.state, "frames", 5)
    required = getattr(app.state, "required", 3)
    faces = getattr(app.state, "faces", False)
    blur_faces = getattr(app.state, "blur_faces", False)
    cctv_mode = getattr(app.state, "cctv_mode", True)
    clahe_enhance = getattr(app.state, "clahe_enhance", False)
    sharpness_boost = getattr(app.state, "sharpness_boost", True)
    tile_inference = getattr(app.state, "tile_inference", True)
    imgsz = getattr(app.state, "imgsz", 960)
    detect_mode = getattr(app.state, "detect_mode", "weapons_only")
    weapon_filter = getattr(app.state, "weapon_filter", "all_weapons")

    stream_manager = RTSPStreamManager(
        model_path=model_path,
        source=initial_source,
        confidence=confidence,
        frames=frames,
        required=required,
        faces=faces,
        blur_faces=blur_faces,
        cctv_mode=cctv_mode,
        clahe_enhance=clahe_enhance,
        sharpness_boost=sharpness_boost,
        tile_inference=tile_inference,
        imgsz=imgsz,
        detect_mode=detect_mode,
        weapon_filter=weapon_filter,
        output_dir=Path("outputs")
    )
    stream_manager.start()
    print("[INFO] Universal Weapon & Multi-Tile CCTV Stream Manager started in WEAPONS-ONLY Mode.")
    yield
    # Shutdown logic
    if stream_manager:
        stream_manager.stop()
        print("[INFO] RTSP Stream Manager stopped.")


app = FastAPI(title="Firearm Vision — High-Accuracy Weapon Detection Surveillance System", lifespan=lifespan)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount faces directory for frontend autograph/picture panels
faces_dir = Path(__file__).parent / "data" / "enrolled_faces"
faces_dir.mkdir(parents=True, exist_ok=True)
app.mount("/faces", StaticFiles(directory=str(faces_dir)), name="faces")


class ConnectRequest(BaseModel):
    source: str = Field(..., description="RTSP URL (e.g. rtsp://192.168.1.100:554/stream), webcam index '0', or video path")


class SettingsRequest(BaseModel):
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    frames: int | None = Field(None, ge=1, le=30)
    required: int | None = Field(None, ge=1, le=30)
    faces: bool | None = None
    blur_faces: bool | None = None
    cctv_mode: bool | None = None
    clahe_enhance: bool | None = None
    sharpness_boost: bool | None = None
    tile_inference: bool | None = None
    imgsz: int | None = Field(None, ge=320, le=1920)
    detect_mode: str | None = Field(None, description="'weapons_only', 'faces_only', 'persons_only', or 'all'")
    weapon_filter: str | None = Field(None, description="'all_weapons' or 'firearms_only'")


class EnrollRequest(BaseModel):
    name: str = Field(..., description="Full Name of Person to Enroll")


@app.get("/")
async def serve_index():
    index_file = static_dir / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(str(index_file))


@app.get("/api/stream")
async def video_stream():
    """Returns real-time multipart MJPEG video stream with weapon detection annotations."""
    if not stream_manager:
        raise HTTPException(status_code=503, detail="Stream manager not initialized")
    return StreamingResponse(
        stream_manager.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/status")
async def get_status():
    if not stream_manager:
        raise HTTPException(status_code=503, detail="Stream manager not initialized")
    return stream_manager.get_status()


@app.get("/api/alerts")
async def get_alerts():
    if not stream_manager:
        raise HTTPException(status_code=503, detail="Stream manager not initialized")
    return {"alerts": stream_manager.get_alerts()}


@app.post("/api/connect")
async def connect_source(payload: ConnectRequest):
    if not stream_manager:
        raise HTTPException(status_code=503, detail="Stream manager not initialized")
    stream_manager.connect(payload.source)
    return {"status": "ok", "message": f"Connecting to {payload.source}", "source": payload.source}


@app.post("/api/settings")
async def update_settings(payload: SettingsRequest):
    if not stream_manager:
        raise HTTPException(status_code=503, detail="Stream manager not initialized")
    stream_manager.update_settings(
        confidence=payload.confidence,
        frames=payload.frames,
        required=payload.required,
        faces=payload.faces,
        blur_faces=payload.blur_faces,
        cctv_mode=payload.cctv_mode,
        clahe_enhance=payload.clahe_enhance,
        sharpness_boost=payload.sharpness_boost,
        tile_inference=payload.tile_inference,
        imgsz=payload.imgsz,
        detect_mode=payload.detect_mode,
        weapon_filter=payload.weapon_filter
    )
    return {"status": "ok", "settings": stream_manager.get_status()}


@app.post("/api/enroll")
async def enroll_person(payload: EnrollRequest):
    if not stream_manager or stream_manager.latest_raw_frame is None:
        raise HTTPException(status_code=400, detail="No active video stream frame available for enrollment")

    success = stream_manager.enroll_person_burst(payload.name, num_samples=30)
    if success:
        return {"status": "ok", "message": f"Successfully captured & enrolled 30 multi-angle facial photos for '{payload.name}'"}
    raise HTTPException(status_code=500, detail=f"Could not extract facial embeddings for '{payload.name}'")


def main():
    parser = argparse.ArgumentParser(description="Firearm Vision Universal Weapon & CCTV Dashboard Server")
    parser.add_argument("--model", type=str, default="runs/detect/runs/universal-weapon/weights/best.pt", help="Path to custom YOLO weapon model")
    parser.add_argument("--source", default="0", help="RTSP URL (rtsp://...), webcam index (0), or video file")
    parser.add_argument("--confidence", type=float, default=0.35, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--frames", type=int, default=5, help="Temporal window size")
    parser.add_argument("--required", type=int, default=3, help="Required temporal confirmations")
    parser.add_argument("--faces", action="store_true", help="Draw face boxes")
    parser.add_argument("--blur-faces", action="store_true", help="Blur detected faces")
    parser.add_argument("--disable-cctv-mode", action="store_true", help="Disable CCTV high-resolution slice inference")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE low-light enhancement")
    parser.add_argument("--disable-sharpness", action="store_true", help="Disable CCTV edge sharpening filter")
    parser.add_argument("--disable-tiling", action="store_true", help="Disable multi-tile crop inference")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference resolution (640, 960, 1280)")
    parser.add_argument("--mode", default="weapons_only", choices=["weapons_only", "faces_only", "persons_only", "all"], help="Detection target mode")
    parser.add_argument("--weapon-filter", default="all_weapons", choices=["all_weapons", "firearms_only"], help="Filter scope")
    parser.add_argument("--host", default="0.0.0.0", help="Server host IP")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    app.state.model_path = args.model
    app.state.source = args.source
    app.state.confidence = args.confidence
    app.state.frames = args.frames
    app.state.required = args.required
    app.state.faces = args.faces
    app.state.blur_faces = args.blur_faces
    app.state.cctv_mode = not args.disable_cctv_mode
    app.state.clahe_enhance = args.clahe
    app.state.sharpness_boost = not args.disable_sharpness
    app.state.tile_inference = not args.disable_tiling
    app.state.imgsz = args.imgsz
    app.state.detect_mode = args.mode
    app.state.weapon_filter = args.weapon_filter

    print(f"[INFO] Starting Firearm Vision Server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
