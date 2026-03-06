# 🧠 Singularity Vision

**Build. Train. Deploy. Locally.** — A no-code computer vision platform for industry-scale spatial analysis.

Singularity Vision is a hybrid desktop system designed to handle complex computer vision tasks—from simple object detection to advanced 3D geospatial reconstruction—without requiring extensive coding knowledge.

## 🚀 Key Features

- **No-Code CV Workflows**: Manage datasets, train models, and run inference through a unified graphical interface.
- **Spatial Vision Lab**: Native support for GeoTIFF, GeoJSON, and R-tree spatial indexing for geospatial computer vision.
- **Model Hub**: Integrated support for ONNX runtimes and Hugging Face model integration.
- **Industry Profiles**: Behavioral configurations tailored for Mining, Defense, and Health sectors.
- **Offline-First**: Privacy-centric design that prioritizes local execution while allowing optional hybrid cloud offloading.

## 🏗️ Architecture

The system follows a strict engine-isolation principle to ensure scalability and UI responsiveness:

- **Frontend**: Electron + React + Vite (Fast, responsive UI)
- **Backend Orchestrator**: Python (FastAPI / gRPC)
- **Execution Engines**: Modular Python services for Datasets, Training, and Inference.
- **Storage**: Local file system with SQLite/DuckDB for metadata and spatial indexing.

## 📂 Project Structure

Each workspace follows a strict schema for portability:

```text
project_root/
 ├─ project.json       # Core metadata & configuration
 ├─ datasets/          # Raw images, videos, and tiled geo-data
 ├─ annotations/       # Versioned label data (Geo-aware)
 ├─ models/            # ONNX, PyTorch, or TensorFlow models
 ├─ experiments/       # Immutable training snapshots & metrics
 ├─ spatial/           # Maps, GeoJSON, and 3D reconstructions
 └─ outputs/           # Inference results and reports
```

## 🛠️ Getting Started

### Prerequisites
- Node.js (v18+)
- Python 3.10+
- (Optional) NVIDIA GPU for accelerated training/inference

### Development Setup

1. **Install Dependencies**:
   ```bash
   npm install
   pip install -r backend/requirements.txt
   ```

2. **Run in Development Mode**:
   ```bash
   npm run dev
   ```

3. **Run with Electron**:
   ```bash
   npm run electron:dev
   ```

## 📝 License

Private - All rights reserved.
