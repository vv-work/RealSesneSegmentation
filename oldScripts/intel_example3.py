import pyrealsense2 as rs
import time
import cv2
import numpy as np

# Check for connected RealSense devices
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("No RealSense devices found. Please make sure the camera is connected.")
    exit(1)
else:
    print("Found the following RealSense device(s):")
    for i, device in enumerate(devices):
        print(f"Device {i+1}: {device.get_info(rs.camera_info.name)}")

# Implement two "processing" functions, each of which
# occassionally lags and takes longer to process a frame.
def slow_processing(frame):
    n = frame.get_frame_number() 
    if n % 20 == 0:
        time.sleep(1/4)
    print(n)

def slower_processing(frame):
    n = frame.get_frame_number() 
    if n % 20 == 0:
        time.sleep(1)
    print(n)

try:
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming to the slow processing function.
    # This stream will lag, causing the occasional frame drop.
    print("Slow callback")
    pipeline.start(config)
    start = time.time()
    while time.time() - start < 5:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()  # Extract the color frame
        slow_processing(color_frame)  # Process the color frame
        # Display the color frame
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('Color Stream', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pipeline.stop()

    # Start streaming to the the slow processing function by way of a frame queue.
    # This stream will occasionally hiccup, but the frame_queue will prevent frame loss.
    print("Slow callback + queue")
    queue = rs.frame_queue(50)
    pipeline.start(config, queue)
    start = time.time()
    while time.time() - start < 5:
        frames = queue.wait_for_frame()
        slow_processing(frames)
    pipeline.stop()

    # Start streaming to the the slower processing function by way of a frame queue.
    # This stream will drop frames because the delays are long enough that the backed up
    # frames use the entire internal frame pool preventing the SDK from creating more frames.
    print("Slower callback + queue")
    queue = rs.frame_queue(50)
    pipeline.start(config, queue)
    start = time.time()
    while time.time() - start < 5:
        frames = queue.wait_for_frame()
        slower_processing(frames)
    pipeline.stop()

    # Start streaming to the slower processing function by way of a keeping frame queue.
    # This stream will no longer drop frames because the frame queue tells the SDK
    # to remove the frames it holds from the internal frame queue, allowing the SDK to
    # allocate space for and create more frames .
    print("Slower callback + keeping queue")
    queue = rs.frame_queue(50, keep_frames=True)
    pipeline.start(config, queue)
    start = time.time()
    while time.time() - start < 5:
        frames = queue.wait_for_frame()
        slower_processing(frames)
    pipeline.stop()

except Exception as e:
    print(e)
except:
    print("A different Error")
else:
    print("Done")
finally:
    cv2.destroyAllWindows()