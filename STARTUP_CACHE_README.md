# 🚀 Fast Startup Guide

## Initial Setup (One-Time)

After the first git clone, run the face cache setup to eliminate startup delays:

```bash
./.venv/bin/python setup_face_cache.py
```

This pre-computes embeddings for all 209 enrolled faces and saves them to `data/enrolled_faces/arcface_cache.pkl` (~872 KB).

**First run:** ~1-2 minutes (computes embeddings)  
**All subsequent runs:** Instant (loads from cache)

## Running the Server

```bash
./.venv/bin/python app.py --source 0
```

Server should be ready in **<5 seconds** with the cache pre-generated.

## How It Works

- **Without cache:** Every app restart re-processes 209 face images → ~30-60 seconds
- **With cache:** App loads pre-computed embeddings from disk → ~1 second
- Cache auto-invalidates if new faces are enrolled or images modified

## Troubleshooting

If face recognition is still slow after first use:
- Cache rebuilds automatically if enrollment adds new faces
- Run `setup_face_cache.py` again to force refresh

