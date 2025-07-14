import time
import threading
import traceback
import numpy as np
import torch
import viser
import viser.transforms as vtf
from scene.cameras import ViewerCam
from utils.graphics_utils import fov2focal, focal2fov
import sys
import os
sys.path.append(os.path.dirname(__file__))
from detection_model import DetectionModel, DetectionProcessor
import traceback
from PIL import Image


class ClientThread(threading.Thread):
    def __init__(self, viewer, renderer, client: viser.ClientHandle, detection_model: str = "grounding_dino", detection_device: str = "cuda:1"):
        super().__init__()
        self.viewer = viewer
        self.renderer = renderer
        self.client = client
        self.detection_model_name = detection_model
        self.detection_device = detection_device

        self.render_trigger = threading.Event()

        self.last_move_time = 0

        self.last_camera = None  # store camera information

        self.state = "low"  # low or high render resolution

        self.stop_client = False  # whether stop this thread

        # Initialize detection model and processor
        self.detection_model = None
        self.detection_processor = None
        self.enable_detection = False
        self.selected_detection_classes = []  # Store selected classes for YOLO/SAHI
        self._initialize_detection()

        client.camera.up_direction = viewer.up_direction

        @client.camera.on_update
        def _(cam: viser.CameraHandle) -> None:
            with self.client.atomic():
                self.last_camera = cam
                self.state = "low"  # switch to low resolution mode when a new camera received
                self.render_trigger.set()

    def _initialize_detection(self):
        """Initialize the detection model and processor."""
        try:
            # Initialize detection model based on argument
            self.detection_model = DetectionModel(model_name=self.detection_model_name, device=self.detection_device)
            self.detection_processor = DetectionProcessor(self.detection_model, enable_detection=False)
            print(f"{self.detection_model_name} detection model initialized successfully on {self.detection_device}")
        except Exception as e:
            print(f"Failed to initialize {self.detection_model_name} model: {e}")
            traceback.print_exc()
            # Optionally, fallback to YOLO if not already YOLO
            if self.detection_model_name != "yolo":
                print("Falling back to YOLO...")
                try:
                    self.detection_model = DetectionModel(model_name="yolo", device=self.detection_device)
                    self.detection_processor = DetectionProcessor(self.detection_model, enable_detection=False)
                    print(f"YOLO detection model initialized successfully on {self.detection_device}")
                except Exception as e2:
                    print(f"Failed to initialize YOLO model: {e2}")
                    self.detection_model = None
                    self.detection_processor = None

    def render_and_send(self):
        with self.client.atomic():
            cam = self.last_camera

            self.last_move_time = time.time()

            # get camera pose
            R = vtf.SO3(wxyz=self.client.camera.wxyz)
            R = R @ vtf.SO3.from_x_radians(np.pi)
            R = torch.tensor(R.as_matrix())
            pos = torch.tensor(self.client.camera.position, dtype=torch.float64)
            c2w = torch.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = pos

            c2w = torch.matmul(self.viewer.camera_transform, c2w)

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = torch.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, 3]

            # calculate resolution
            aspect_ratio = cam.aspect
            max_res, jpeg_quality = self.get_render_options()
            image_height = max_res
            image_width = int(image_height * aspect_ratio)
            if image_width > max_res:
                image_width = max_res
                image_height = int(image_width / aspect_ratio)

            # construct camera
            fov_x = cam.fov
            f = fov2focal(cam.fov, image_width)
            fov_y = focal2fov(f, image_height)
            camera = ViewerCam(
                R=np.array(R),
                T=np.array(T),
                FoVx=fov_x,
                FoVy=fov_y,
                width=image_width,
                height=image_height,
                data_device=self.viewer.device
            )

            with torch.no_grad():
                image = self.renderer.get_outputs(camera, scaling_modifier=self.viewer.scaling_modifier.value)

                image = torch.clamp(image, max=1.)
                image = torch.permute(image, (1, 2, 0))

                # Convert to numpy array for detection processing
                image_np = image.cpu().numpy()

                # Apply detection if enabled
                if self.enable_detection and self.detection_processor is not None:
                    try:
                        # Pass selected classes for YOLO/SAHI
                        image_np = self.detection_processor.process_image(image_np, selected_classes=self.selected_detection_classes)
                    except Exception as e:
                        print(f"Detection processing error: {e}")

                self.client.set_background_image(
                    image_np,
                    format=self.viewer.image_format,
                    jpeg_quality=jpeg_quality,
                )

    def run(self):
        while True:
            trigger_wait_return = self.render_trigger.wait(0.2)  # TODO: avoid wasting CPU
            # stop client thread?
            if self.stop_client is True:
                break
            if not trigger_wait_return:
                # skip if camera is none
                if self.last_camera is None:
                    continue

                # if we haven't received a trigger in a while, switch to high resolution
                if self.state == "low":
                    self.state = "high"  # switch to high resolution mode
                else:
                    continue  # skip if already in high resolution mode

            self.render_trigger.clear()

            try:
                self.render_and_send()
            except Exception as err:
                print("error occurred when rendering for client")
                traceback.print_exc()
                break

        self._destroy()

    def get_render_options(self):
        if self.state == "low":
            return self.viewer.max_res_when_moving.value, int(self.viewer.jpeg_quality_when_moving.value)
        return self.viewer.max_res_when_static.value, int(self.viewer.jpeg_quality_when_static.value)

    def stop(self):
        self.stop_client = True
        # self.render_trigger.set()  # TODO: potential thread leakage?

    def _destroy(self):
        print("client thread #{} destroyed".format(self.client.client_id))
        self.viewer = None
        self.renderer = None
        self.client = None
        self.last_camera = None

    def toggle_detection(self, enable: bool):
        """Enable or disable object detection."""
        self.enable_detection = enable
        if self.detection_processor is not None:
            self.detection_processor.toggle_detection(enable)
        print(f"Detection {'enabled' if enable else 'disabled'}")

    def set_detection_confidence(self, confidence: float):
        """Set confidence threshold for detection."""
        if self.detection_processor is not None:
            self.detection_processor.set_confidence_threshold(confidence)

    def set_detection_draw_options(self, draw_labels: bool, draw_confidence: bool):
        """Set drawing options for detection results."""
        if self.detection_processor is not None:
            self.detection_processor.set_draw_options(draw_labels, draw_confidence)

    def set_detection_text_prompt(self, text_prompt: str):
        """Set text prompt for Grounding DINO detection."""
        if self.detection_processor is not None:
            self.detection_processor.set_text_prompt(text_prompt)

    def set_detection_classes(self, selected_classes):
        """Set selected detection classes for YOLO/SAHI."""
        self.selected_detection_classes = selected_classes
        if self.detection_processor is not None:
            self.detection_processor.set_selected_classes(selected_classes)

    def switch_detection_model(self, new_model_name: str):
        """Switch the detection model at runtime."""
        if self.detection_model_name == new_model_name:
            return  # No change
        self.detection_model_name = new_model_name
        prev_enable = self.enable_detection
        prev_confidence = None
        prev_draw_labels = None
        prev_draw_confidence = None
        prev_text_prompt = None
        if self.detection_processor is not None:
            try:
                prev_confidence = getattr(self.detection_processor, 'confidence_threshold', None)
                prev_draw_labels = getattr(self.detection_processor, 'draw_labels', None)
                prev_draw_confidence = getattr(self.detection_processor, 'draw_confidence', None)
                prev_text_prompt = getattr(self.detection_processor, 'text_prompt', None)
            except Exception:
                pass
        self._initialize_detection()
        # Restore previous settings if possible
        self.toggle_detection(prev_enable)
        if prev_confidence is not None:
            self.set_detection_confidence(prev_confidence)
        if prev_draw_labels is not None and prev_draw_confidence is not None:
            self.set_detection_draw_options(prev_draw_labels, prev_draw_confidence)
        if prev_text_prompt is not None:
            self.set_detection_text_prompt(prev_text_prompt)
