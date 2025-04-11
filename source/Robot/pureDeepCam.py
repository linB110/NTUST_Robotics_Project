import pyrealsense2 as rs
import numpy as np
import cv2

def get_center_pixel_value(image):
    height, width = image.shape[:2]
    center_x = int(width / 2)
    center_y = int(height / 2)
    pixel_value = image[center_y, center_x]
    return pixel_value, (center_x, center_y)

if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Get pixel value at the center of the image
            center_pixel_value, center_point = get_center_pixel_value(depth_image)
            
            # Print center pixel value
            print("Center Pixel Value:", center_pixel_value)
            
            # Mark center point on the color image
            cv2.circle(color_image, center_point, 5, (0, 0, 255), -1)  # Red circle with radius 5

            P1 = (center_point[0], center_point[1] + 50)  # 100 pixels below center
            P2 = (center_point[0], center_point[1] + 150)  # 100 pixels above center
            cv2.circle(color_image, P1, 5, (255, 0, 0), -1) # blue
            cv2.circle(color_image, P2, 5, (255, 0, 0), -1) # blue
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)
            
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
