#!/usr/bin/env python3
"""
Example script demonstrating how to use the detection model integration with the viewer.

This script shows how to:
1. Run the viewer with detection capabilities
2. Control detection parameters through the UI
3. View detection results in real-time

Usage:
    python example_detection_viewer.py <model_path> [options]
"""

import argparse
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from viewer import Viewer

def main():
    parser = argparse.ArgumentParser(description="Run CityGaussian viewer with object detection")
    parser.add_argument("model_path", type=str, help="Path to the Gaussian model")
    parser.add_argument("--host", "-a", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--background_color", "--bkg_color", "-b",
                        type=str, nargs="+", default=["black"],
                        help="Background color (e.g.: white, black, 0 0 0)")
    parser.add_argument("--image_format", "-f", type=str, default="jpeg",
                        choices=["jpeg", "png"], help="Image format for streaming")
    parser.add_argument("--sh_degree", "--sh-degree", "--sh",
                        type=int, default=3, help="Spherical harmonics degree")
    parser.add_argument("--enable_transform", "--enable-transform",
                        action="store_true", default=False,
                        help="Enable transform options on Web UI")
    parser.add_argument("--show_cameras", "--show-cameras",
                        action="store_true", default=False,
                        help="Show camera frustums in the scene")
    parser.add_argument("--cameras-json", "--cameras_json", type=str, default=None,
                        help="Path to cameras.json file")
    parser.add_argument("--detection-model", type=str, default="grounding_dino",
                        choices=["grounding_dino", "yolo"],
                        help="Detection model to use (grounding_dino or yolo)")
    
    args = parser.parse_args()

    # Process background color
    if len(args.background_color) == 1 and isinstance(args.background_color[0], str):
        if args.background_color[0] == "white":
            args.background_color = (1., 1., 1.)
        else:
            args.background_color = (0., 0., 0.)
    else:
        args.background_color = tuple([float(i) for i in args.background_color])

    print("Starting CityGaussian viewer with Grounding DINO detection...")
    print(f"Model path: {args.model_path}")
    print(f"Server: http://{args.host}:{args.port}")
    print("\nGrounding DINO Features:")
    print("- Text-guided object detection")
    print("- Enable/disable detection via UI")
    print("- Adjust detection confidence threshold")
    print("- Customize text prompts for specific objects")
    print("- Toggle detection labels and confidence scores")
    print("- Real-time object detection on rendered images")
    print("\nInstructions:")
    print("1. Open the viewer in your web browser")
    print("2. Navigate to the 'General' tab")
    print("3. Find the 'Object Detection' folder")
    print("4. Check 'Enable Detection' to start detecting objects")
    print("5. Modify the 'Text Prompt' to detect specific objects")
    print("6. Adjust confidence threshold and display options as needed")
    print("\nText Prompt Examples:")
    print("- 'person . car . bicycle' - Detect people, cars, and bicycles")
    print("- 'chair . table . bottle' - Detect furniture and objects")
    print("- 'laptop . mouse . keyboard' - Detect computer equipment")

    # Create and start viewer
    viewer_init_args = {key: getattr(args, key) for key in vars(args)}
    viewer = Viewer(**viewer_init_args)
    
    try:
        viewer.start()
    except KeyboardInterrupt:
        print("\nShutting down viewer...")
    except Exception as e:
        print(f"Error running viewer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 