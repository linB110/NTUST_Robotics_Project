import pyrealsense2 as rs
import numpy as np

def capture_depth_value():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("Failed to capture frames")
            return None

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Calculate center pixel coordinates
        height, width = depth_image.shape[:2]
        center_x = int(width / 2)
        center_y = int(height / 2)

        # Get pixel value at the center of the image
        pixel_value = depth_image[center_y, center_x]

        return pixel_value

    finally:
        # Stop streaming
        pipeline.stop()

if __name__ == "__main__":
    depth_value = capture_depth_value()
    if depth_value is not None:
        print("Depth Value at Center Pixel:", depth_value)
