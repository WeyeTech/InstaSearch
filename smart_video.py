import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
import imageio
from pathlib import Path
import uuid
import time
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, OptimizersConfigDiff
from ultralytics import YOLO
import clip
from utils import CroppedImage


class SmartVideoSearch:
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str = "smart_video_embeddings",
        temp_frame_dir: str = "temp_frames",
        clip_model_name: str = "ViT-B/32",
        yolo_model_path: str = "yolov8x.pt"
    ):
        BASE_DIR = Path(__file__).parent.resolve()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(clip_model_name, device=self.device)
        self.yolo_model = YOLO(str(BASE_DIR / yolo_model_path))
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        self.temp_frame_dir = BASE_DIR / temp_frame_dir
        self.temp_frame_dir.mkdir(parents=True, exist_ok=True)
        print(f"Initialized SmartVideoSearch on device: {self.device}")
        self._warmup_gpu()

    def _warmup_gpu(self):
        if self.device == "cuda":
            dummy = torch.zeros(1).to(self.device)
            _ = dummy * 1
            print("GPU warmup complete.")

    def get_device_status(self) -> dict:
        return {
            "device": self.device,
            "gpu_name": torch.cuda.get_device_name(0) if self.device == "cuda" else None,
            "available": torch.cuda.is_available()
        }

    def extract_frames(self, video_path: str, segment_duration: int = 5) -> List[Tuple[Image.Image, float]]:
        frames = []
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data()['fps']
        total_frames = reader.count_frames()
        frame_interval = int(segment_duration * fps)

        for frame_index in range(0, total_frames, frame_interval):
            try:
                frame = reader.get_data(frame_index)
                pil_image = Image.fromarray(frame)
                timestamp = frame_index / fps
                frames.append((pil_image, timestamp))
            except Exception as e:
                print(f"Skipping frame {frame_index}: {e}")
        reader.close()
        return frames

    def detect_objects(self, image: Image.Image, frame_path: str) -> List[CroppedImage]:
        np_image = np.array(image)
        results = self.yolo_model(np_image)
        cropped_images = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                cropped_images.append(CroppedImage(parent_path=frame_path, box=(x1, y1, x2, y2), cls=class_name))
        return cropped_images

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(input_tensor)
            emb /= emb.norm(dim=-1, keepdim=True)
        return emb.cpu()

    def generate_embeddings(self, video_path: str, segment_duration: int = 5) -> List[dict]:
        embeddings = []
        frames = self.extract_frames(video_path, segment_duration)

        for i, (image, timestamp) in enumerate(frames):
            frame_path = self.temp_frame_dir / f"frame_{i:04d}.jpg"
            image.save(frame_path)

            full_frame_emb = self._encode_image(image)
            embeddings.append({
                "embedding": full_frame_emb.numpy(),
                "image_path": str(frame_path),
                "class": None,
                "timestamp": timestamp
            })

            cropped_images = self.detect_objects(image, str(frame_path))
            for cropped in cropped_images:
                obj_img = cropped.get_cropped_image()
                obj_emb = self._encode_image(obj_img)
                embeddings.append({
                    "embedding": obj_emb.numpy(),
                    "image_path": str(frame_path),
                    "class": cropped.cls,
                    "timestamp": timestamp
                })

        return embeddings

    def get_collection_size(self) -> int:
        info = self.qdrant.get_collection(collection_name=self.collection_name)
        return getattr(info, 'points_count', 0)

    def recreate_collection(self):
        self.qdrant.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=1)
        )

    def upsert_embeddings(self, embeddings: List[dict], batch_size: int = 1000):
        points = []
        for emb in embeddings:
            vector = emb["embedding"].flatten().tolist()
            payload = {
                "image_path": emb["image_path"],
                "class": emb["class"],
                "timestamp": emb.get("timestamp")
            }
            point_id = str(uuid.uuid4())
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.qdrant.upsert(collection_name=self.collection_name, points=batch)
            time.sleep(1)

    def ensure_embeddings(self, video_path: str, min_points_threshold: int = 5):
        current_size = self.get_collection_size()
        if current_size <= min_points_threshold:
            self.recreate_collection()
            embeddings = self.generate_embeddings(video_path)
            self.upsert_embeddings(embeddings)

    def search(self, text_query: str, top_k: int = 5) -> List[Tuple[dict, float]]:
        text_tokens = clip.tokenize([text_query]).to(self.device)
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        query_vector = text_embedding.cpu().numpy().flatten().tolist()

        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        return [(hit.payload, hit.score) for hit in results]
