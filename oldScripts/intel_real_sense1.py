from IPython import display
from openvino.inference_engine import IECore
import collections
import cv2
import os
from utils import *
import time
import pyrealsense2 as rs
import open3d as o3d

def run_object_detection(source=0, flip=False, use_popup=False, skip_first_frames=0):
    device = None
    original_width = None
    original_height = None
    video_frames = []
    masks = None
    frame = None
    dist = []
    pcls = []

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960, height=540)
    render = vis.get_render_option()
    render.background_color = [0,0,0]
    render.point_size = 1

    try:
        # Check if Intel RealSense camera is available
        if "rs" not in globals():
            print("Error: Intel RealSense camera library 'pyrealsense2' is not available.")
            return
        
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

        profile = pipeline.start(config)

        params_o3d = o3d.camera.PinholeCameraIntrinsic()

        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(
                winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE
            )

        processing_times = collections.deque()
        visualizer = InstanceSegmentationVisualizer(classes, True, True)

        # init count depth
        t0 = time.time()
        while True:
            # Wait for the next frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("Source ended")
                break

            frame_rgb = np.array(color_frame.get_data())
            frame = frame_rgb[:, :, 0:3]

            # If the frame is larger than full HD, reduce size to improve performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            # Resize the image and change dimensions to fit neural network input.
            input_img = cv2.resize(
                src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA
            )
            # Create a batch of images (size = 1).
            input_img = input_img[np.newaxis, ...]
            input_img = np.moveaxis(input_img, -1, 1)

            # Measure processing time.
            start_time = time.time()
            # Get the results.
            results = list(compiled_model.infer(inputs={input_layer: input_img}).values())
            stop_time = time.time()
            # Get network outputs
            labels, scores, boxes, masks = process_results(frame=frame, input_img=input_img, results=results)
            frame_raw = frame.copy()
            # visualize
            frame = visualizer(frame, boxes, labels, scores, masks, dist, None, None)

            processing_times.append(stop_time - start_time)
            # Use processing times from the last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            # video_frames.append(frame)

            # Use this workaround if there is flickering.
            if use_popup:
                cv2.imshow(winname=title, mat=frame)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                _, encoded_img = cv2.imencode(
                    ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
                )
                # Create an IPython image.
                i = display.Image(data=encoded_img)
                # Display the image in this notebook.
                display.clear_output(wait=True)
                display.display(i)

    # ctrl-c
    except ImportError:
        print("Error: Failed to import one or more required libraries.")
    except KeyboardInterrupt:
        print("Interrupted")
    except RuntimeError as e:
        print(e)
    finally:
        pipeline.stop()
        vis.destroy_window()
        if use_popup:
            cv2.destroyAllWindows()
        return video_frames, pcls

# Define path to models
base_model_dir = "models/model-segmentation"
model_segmentation = "instance-segmentation-person-0007"
converted_model_path = f"models/model-segmentation/intel/{model_segmentation}/FP16/{model_segmentation}.xml"

try:
    ie = IECore()
except ImportError:
    print("Error: Failed to import 'IECore' from 'openvino.inference_engine'.")
    exit()

# Check if the model file exists
if not os.path.isfile(converted_model_path):
    print(f"Error: Model file not found at path: {converted_model_path}")
    exit()

# Read the network and corresponding weights from a file
try:
    model = ie.read_network(model=converted_model_path)
except Exception as e:
    print("Error: Failed to read the network.")
    print(e)
    exit()

# Compile the model for the CPU device
try:
    compiled_model = ie.load_network(network=model, device_name="CPU")
except Exception as e:
    print("Error: Failed to compile the model.")
    print(e)
    exit()

# Get the input and output nodes.
input_layer = next(iter(model.input_info))
output_boxes = next(iter(model.outputs))

# Get the input size.
height, width = model.input_info[input_layer].input_data.shape[2:]

classes = [
    "person", "background", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
    "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

video_frames, pcls = run_object_detection(source=0, flip=False, use_popup=False)



