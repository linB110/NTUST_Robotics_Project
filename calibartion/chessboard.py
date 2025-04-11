import cv2
import numpy as np

# open camera
cap = cv2.VideoCapture(0)

# chess board size (inner corners)
chessboard_size = (9, 7)

square_size = 10  # mm

# store corners
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# prepare object point
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

while True:
    # capture image
    ret, frame = cap.read()
    
    if not ret:
        print("無法捕獲影像")
        break

    # change image to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find corner on chess board
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # if find corner, plot it on screen
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

        # annotate corner cordinate
        corners = corners.reshape(-1, 2)
        for i, corner in enumerate(corners):
            x, y = corner
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
        
        pixelRatioArrays = [] 
        # calculate pixel to mm rate
        if len(corners) > 1:  # make sure corners is sufficient
            pixel_distance = np.linalg.norm(corners[0] - corners[1])  # distance between 2 corners
            pixel_to_mm_ratio = square_size / pixel_distance  # pixel to mm
            pixelRatioArrays.append(pixel_to_mm_ratio)
            # print(f"Pixel to mm ratio: {pixel_to_mm_ratio:.4f} mm/pixel")
        
        AvgPixelRatio = 0
        for i in range(len(pixelRatioArrays)):
            AvgPixelRatio += pixelRatioArrays[i]
        
        print("Average pixel ratio", AvgPixelRatio / len(pixelRatioArrays))
        
        width, height = frame.shape[:2]
        width = width / 2
        height = height / 2
        NewWidth = width + 10
    
    # show image
    cv2.circle(frame, (int(width), int(height)), 1, (255, 0, 0), -1)
    cv2.circle(frame, (int(NewWidth), int(height)), 1, (255, 0, 0), -1)
    cv2.imshow('Chessboard Corners Detection', frame)
    

    # press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera
cap.release()
cv2.destroyAllWindows()
