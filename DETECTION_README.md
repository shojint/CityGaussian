# Grounding DINO Integration for CityGaussian Viewer

This integration adds real-time text-guided object detection capabilities to the CityGaussian viewer using Grounding DINO, allowing you to detect specific objects in rendered 3D scenes based on text descriptions and overlay detection results on the viewer images.

## Features

- **Text-Guided Object Detection**: Detect objects using natural language descriptions with Grounding DINO
- **Interactive UI Controls**: Enable/disable detection, adjust confidence thresholds, and customize text prompts
- **Multiple Model Support**: Support for Grounding DINO (primary), YOLO, and Faster R-CNN models
- **Configurable Visualization**: Toggle detection labels and confidence scores
- **Multi-client Support**: Detection works with multiple simultaneous viewers
- **Fallback Support**: Automatically falls back to YOLO if Grounding DINO is not available

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r detection_requirements.txt
   ```

2. **Install Detection Models** (optional):
   - YOLO models will be automatically downloaded on first use
   - For custom models, place them in the appropriate directory

## Usage

### Basic Usage

Run the viewer with Grounding DINO detection:

```bash
python example_detection_viewer.py <path_to_your_model>
```

Or use the original viewer (Grounding DINO will be available):

```bash
python viewer.py <path_to_your_model>
```

### Web Interface

1. Open your web browser and navigate to the viewer URL (default: `http://localhost:8080`)
2. Navigate to the **General** tab
3. Find the **Object Detection** folder
4. Use the controls to:
   - **Enable Detection**: Toggle object detection on/off
   - **Text Prompt**: Customize the text description for objects to detect
   - **Detection Confidence**: Adjust the minimum confidence threshold (0.1-1.0)
   - **Draw Labels**: Show/hide class labels on detections
   - **Draw Confidence**: Show/hide confidence scores on detections

### Text Prompt Examples

- `person . car . bicycle` - Detect people, cars, and bicycles
- `chair . table . bottle` - Detect furniture and objects
- `laptop . mouse . keyboard` - Detect computer equipment
- `tv . remote . cell phone` - Detect electronics
- `book . cup . bottle` - Detect common objects

### Programmatic Usage

You can also control detection programmatically:

```python
from detection_model import DetectionModel, DetectionProcessor

# Initialize Grounding DINO model
detection_model = DetectionModel(model_name="grounding_dino", device="cuda")
detection_processor = DetectionProcessor(detection_model, enable_detection=True)

# Set text prompt for specific objects
detection_processor.set_text_prompt("person . car . bicycle")

# Process an image
import numpy as np
image = np.random.rand(480, 640, 3)  # Your image here
processed_image = detection_processor.process_image(image)
```

## Supported Models

### Grounding DINO (Primary)
- **Model**: Swin Transformer backbone with DINO architecture
- **Capability**: Text-guided object detection using natural language
- **Classes**: Customizable via text prompts
- **Performance**: High accuracy with flexible object detection
- **Default Text Prompt**: Common objects like person, car, furniture, electronics

### YOLO (Fallback)
- **Default**: Uses YOLOv8n from ultralytics
- **Alternative**: Uses torchvision YOLO if ultralytics is not available
- **Classes**: 80 COCO classes (person, car, dog, etc.)
- **Usage**: Automatically used if Grounding DINO is not available

### Faster R-CNN
- **Model**: ResNet-50 FPN backbone
- **Classes**: 91 COCO classes
- **Performance**: Generally more accurate but slower than YOLO

## Configuration

### Model Selection

To use a different detection model, modify the `_initialize_detection` method in `scene/viewer/client.py`:

```python
def _initialize_detection(self):
    try:
        # Use Faster R-CNN instead of YOLO
        self.detection_model = DetectionModel(model_name="faster_rcnn", device="cuda")
        self.detection_processor = DetectionProcessor(self.detection_model, enable_detection=False)
        print("Detection model initialized successfully")
    except Exception as e:
        print(f"Failed to initialize detection model: {e}")
        self.detection_model = None
        self.detection_processor = None
```

### Custom Models

To use custom detection models:

1. **Create a custom detection class**:
   ```python
   class CustomDetectionModel(DetectionModel):
       def __init__(self, model_path: str, device: str = "cuda"):
           super().__init__("custom", device)
           # Load your custom model here
           self.model = load_your_model(model_path)
           self.class_names = your_class_names
   ```

2. **Override detection methods**:
   ```python
   def detect(self, image: np.ndarray, confidence_threshold: float = 0.5):
       # Implement your detection logic
       return detections
   ```

## Performance Considerations

- **GPU Memory**: Detection models require additional GPU memory
- **Processing Time**: Detection adds latency to rendering (typically 10-50ms per frame)
- **Resolution**: Higher resolution images take longer to process
- **Model Size**: Larger models (YOLOv8x vs YOLOv8n) are more accurate but slower

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   pip install ultralytics opencv-python torchvision
   ```

2. **CUDA Out of Memory**:
   - Reduce image resolution in viewer settings
   - Use smaller detection models (YOLOv8n instead of YOLOv8x)
   - Close other GPU applications

3. **Slow Performance**:
   - Lower detection confidence threshold
   - Reduce image resolution
   - Use CPU instead of GPU (change device to "cpu")

4. **No Detections**:
   - Check if detection is enabled in the UI
   - Lower confidence threshold
   - Verify the scene contains detectable objects

### Debug Mode

Enable debug output by modifying the detection processing:

```python
# In scene/viewer/client.py, render_and_send method
if self.enable_detection and self.detection_processor is not None:
    try:
        image_np = self.detection_processor.process_image(image_np)
        print(f"Detection processed successfully")
    except Exception as e:
        print(f"Detection processing error: {e}")
        import traceback
        traceback.print_exc()
```

## API Reference

### DetectionModel

```python
class DetectionModel:
    def __init__(self, model_name: str = "yolo", device: str = "cuda")
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[dict]
    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray
```

### DetectionProcessor

```python
class DetectionProcessor:
    def __init__(self, detection_model: DetectionModel, enable_detection: bool = True)
    def process_image(self, image: np.ndarray) -> np.ndarray
    def set_confidence_threshold(self, threshold: float)
    def set_draw_options(self, draw_labels: bool, draw_confidence: bool)
    def toggle_detection(self, enable: bool)
```

## Examples

### Basic Detection Example

```python
from detection_model import DetectionModel, DetectionProcessor
import numpy as np

# Initialize
model = DetectionModel("yolo", "cuda")
processor = DetectionProcessor(model, enable_detection=True)

# Process image
image = np.random.rand(480, 640, 3)  # Your image
result = processor.process_image(image)
```

### Custom Detection Pipeline

```python
# Custom detection with post-processing
detections = model.detect(image, confidence_threshold=0.7)

# Filter detections
filtered_detections = [
    d for d in detections 
    if d['class_name'] in ['person', 'car', 'bicycle']
]

# Draw custom visualization
result = model.draw_detections(image, filtered_detections)
```

## Contributing

To add support for new detection models:

1. Extend the `DetectionModel` class
2. Implement the required methods (`detect`, `draw_detections`)
3. Add model initialization logic
4. Update the model selection in `ClientThread._initialize_detection`

## License

This detection integration follows the same license as the main CityGaussian project. 