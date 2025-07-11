import torch
import numpy as np
import cv2
from typing import List, Tuple, Optional
import torchvision.transforms as transforms
import groundingdino.datasets.transforms as T
from PIL import Image

import sys
import os
sys.path.append(os.path.dirname("/home/vgpu/GroundingDINO/"))

import groundingdino
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict

import os
import urllib.request

from ultralytics import YOLO
import traceback


class DetectionModel:
    """
    Base class for object detection models that can be integrated with the viewer.
    This class provides a common interface for different detection models.
    """
    
    def __init__(self, model_name: str = "grounding_dino", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = self.transform = None
        self.class_names = []
        self.text_prompt = "person . car . bicycle . motorcycle . bus . truck . chair . table . bottle . cup . book . cell phone . laptop . mouse . keyboard . tv . remote ."
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the detection model. Override this method for different models."""
        if self.model_name.lower() == "yolo":
            self._initialize_yolo()
        elif self.model_name.lower() == "grounding_dino":
            self._initialize_grounding_dino()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _initialize_yolo(self):
        """Initialize YOLO model using ultralytics."""   
        self.model = YOLO('weights/yolo11x.pt')
        self.model.to(self.device)
        # Get class names from the model
        self.class_names = list(self.model.names.values())
        print(f"Loaded Ultralytics YOLO model with {len(self.class_names)} classes")
    
    def _initialize_grounding_dino(self):
        """Initialize Grounding DINO model."""
        try:
            
            
            # Load Grounding DINO model
            config_file = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
            checkpoint_path = "groundingdino_swint_ogc.pth"
            
            # Try to find the model files
            
            if not os.path.exists(config_file):
                # Try alternative paths
                possible_configs = [
                    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
                    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                    "GroundingDINO/config/GroundingDINO_SwinT_OGC.py"
                ]
                for config in possible_configs:
                    if os.path.exists(config):
                        config_file = config
                        break
            
            if not os.path.exists(checkpoint_path):
                # Try alternative paths
                possible_checkpoints = [
                    "groundingdino_swint_ogc.pth",
                    "GroundingDINO/groundingdino_swint_ogc.pth",
                    "weights/groundingdino_swint_ogc.pth"
                ]
                for checkpoint in possible_checkpoints:
                    if os.path.exists(checkpoint):
                        checkpoint_path = checkpoint
                        break
            
            # Load model configuration
            args = SLConfig.fromfile(config_file)
            args.device = self.device
            
            # Build model
            self.model = build_model(args)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            # Set up text prompt classes
            self.class_names = [name.strip() for name in self.text_prompt.split('.') if name.strip()]
            print(f"Loaded Grounding DINO model with {len(self.class_names)} text classes")
            print(f"Text prompt: {self.text_prompt}")
            
        except Exception as e:
            print(f"Error loading Grounding DINO: {e}")
            print("Attempting to download model...")
            self._download_grounding_dino()
    
    def _download_grounding_dino(self):
        """Download Grounding DINO model if not available."""
        try:

            # Create directories
            os.makedirs("weights", exist_ok=True)
            
            # Download model files
            model_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            
            print("Downloading Grounding DINO model...")
            urllib.request.urlretrieve(model_url, "weights/groundingdino_swint_ogc.pth")
            urllib.request.urlretrieve(config_url, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
            
            print("Download complete. Reinitializing...")
            self._initialize_grounding_dino()
            
        except Exception as e:
            print(f"Failed to download Grounding DINO: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for detection model."""
        if self.model_name.lower() == "faster_rcnn":
            return self._preprocess_faster_rcnn(image)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _preprocess_faster_rcnn(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for Faster R-CNN model."""
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform(pil_image).to(self.device)
    
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.25, text_prompt: str = None) -> List[dict]:
        """
        Detect objects in the image.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
            confidence_threshold: Minimum confidence for detections
            text_prompt: Text prompt for Grounding DINO (only used for grounding_dino model)
            
        Returns:
            List of detection dictionaries with keys: 'bbox', 'confidence', 'class_id', 'class_name'
        """
        with torch.no_grad():
            if self.model_name.lower() == "yolo":
                return self._detect_yolo(image, confidence_threshold)
            elif self.model_name.lower() == "faster_rcnn":
                return self._detect_faster_rcnn(image, confidence_threshold)
            elif self.model_name.lower() == "grounding_dino":
                return self._detect_grounding_dino(image, confidence_threshold, text_prompt)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _detect_yolo(self, image: np.ndarray, confidence_threshold: float) -> List[dict]:
        """Detect objects using YOLO model."""
        if hasattr(self.model, 'predict'):  # Ultralytics YOLO
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            # image = Image.fromarray(image)
            print("image.shape", image.shape)
            results = self.model(image, conf=confidence_threshold)
            # print(results)
            # results = self.model(image)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.class_names[class_id]
                        })
            
            return detections
    
    def _detect_faster_rcnn(self, image: np.ndarray, confidence_threshold: float) -> List[dict]:
        """Detect objects using Faster R-CNN model."""
        input_tensor = self._preprocess_faster_rcnn(image)
        predictions = self.model([input_tensor])
        
        detections = []
        for prediction in predictions:
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                if score >= confidence_threshold:
                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': float(score),
                        'class_id': int(label),
                        'class_name': self.class_names[label]
                    })
        
        return detections
    
    def gdino_preprocess_image(self, image_bgr: np.ndarray) -> torch.Tensor:
        if image_bgr.dtype == np.float32:
            image_bgr = (image_bgr * 255).clip(0, 255).astype(np.uint8)
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    def _detect_grounding_dino(self, image: np.ndarray, confidence_threshold: float, text_prompt: str = None) -> List[dict]:
        """Detect objects using Grounding DINO model."""
        try:
            
            
            # Use provided text prompt or default
            if text_prompt is None:
                text_prompt = self.text_prompt
            
            # preprocess image
            transformed_image = self.gdino_preprocess_image(image)
            
            # Run prediction
            boxes, logits, phrases = predict(
                model=self.model,
                image=transformed_image,
                caption=text_prompt,
                box_threshold=confidence_threshold,
                text_threshold=confidence_threshold,
                device = self.device
            )
            
            detections = []
            if boxes is not None and len(boxes) > 0:
                boxes = boxes.cpu().numpy()
                logits = logits.cpu().numpy()
                
                for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': float(logit),
                        'class_id': i,
                        'class_name': phrase.strip()
                    })
            
            return detections
            
        except Exception as e:
            traceback.print_exc()
            # print(f"Grounding DINO detection error: {e}")
            return []
    
    def draw_detections(self, image: np.ndarray, detections: List[dict], 
                       draw_labels: bool = True, draw_confidence: bool = True) -> np.ndarray:
        """
        Draw detection boxes and labels on the image.
        
        Args:
            image: Input image as numpy array
            detections: List of detection dictionaries
            draw_labels: Whether to draw class labels
            draw_confidence: Whether to draw confidence scores
            
        Returns:
            Image with detections drawn
        """
        # Convert to uint8 if needed
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        h, w = image.shape[:2]

        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        for detection in detections:
            bbox = detection['bbox']
            # If bbox is normalized, convert to pixel coordinates
            if all(0 <= v <= 1 for v in bbox):
                x_center = bbox[0] * w
                y_center = bbox[1] * h
                box_w = bbox[2] * w
                box_h = bbox[3] * h
                x1 = int(x_center - box_w / 2)
                y1 = int(y_center - box_h / 2)
                x2 = int(x_center + box_w / 2)
                y2 = int(y_center + box_h / 2)
            else:
                x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label text
            label_parts = []
            if draw_labels:
                label_parts.append(detection['class_name'])
            if draw_confidence:
                label_parts.append(f"{detection['confidence']:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw label background
                cv2.rectangle(image, (x1, y1 - text_height - baseline - 5), 
                             (x1 + text_width, y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(image, label, (x1, y1 - baseline - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Convert back to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image


class DetectionProcessor:
    """
    High-level processor that combines detection with image processing.
    """
    
    def __init__(self, detection_model: DetectionModel, enable_detection: bool = True):
        self.detection_model = detection_model
        self.enable_detection = enable_detection
        self.confidence_threshold = 0.25
        self.draw_labels = True
        self.draw_confidence = True
        self.text_prompt = None  # For Grounding DINO
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process an image with object detection.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
            
        Returns:
            Processed image with detections drawn
        """
        if not self.enable_detection:
            return image
        
        try:
            # Run detection
            detections = self.detection_model.detect(
                image, 
                self.confidence_threshold, 
                self.text_prompt
            )

            # Draw detections on image
            processed_image = self.detection_model.draw_detections(
                image, detections, self.draw_labels, self.draw_confidence
            )
            
            return processed_image
            
        except Exception as e:
            print(f"Error in detection processing: {e}")
            traceback.print_exc()
            return image
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for detections."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def set_draw_options(self, draw_labels: bool, draw_confidence: bool):
        """Set drawing options for detections."""
        self.draw_labels = draw_labels
        self.draw_confidence = draw_confidence
    
    def toggle_detection(self, enable: bool):
        """Enable or disable detection processing."""
        self.enable_detection = enable
    
    def set_text_prompt(self, text_prompt: str):
        """Set text prompt for Grounding DINO."""
        self.text_prompt = text_prompt 