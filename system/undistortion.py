import cv2
import numpy as np
import os
from image_processing import ImageParse

# the following parameters are intrinsic parameters of the camera, and are used to undistort the fisheye view:

Intrinsic_matrix_K = np.array([[1.46078020e+03, 0.00000000e+00, 1.04335465e+03], 
                               [0.00000000e+00, 1.47945106e+03, 5.75140043e+02], 
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

Distortion_coefficients_D = np.array([[-0.77739624,  1.05602639, -0.00127229,  0.00214566, -0.78705656]])

K = np.array([[1.44365244e+03, 5.26459870e+00, 1.05698538e+03], [0.00000000e+00, 1.46141676e+03, 5.65369589e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)
D = np.array([[-0.37036341], [-0.39595657], [ 3.82754419], [-5.52770641]], dtype=np.float64)


def undistort(img, frame_num, balance=0.6):
    h, w = img.shape[:2]

    # Estimate optimal new camera matrix
    # TODO: new_K is totally off and needs to be fixed. currently it causes the entire image to go black.
    global K, D
    K = K.astype(np.float64)
    D = D.astype(np.float64)
    # new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), None, balance=balance)
    new_K = K.copy()
    # new_K[0, 0] *= 0.8  # fx
    # new_K[1, 1] *= 0.8  # fy
    if frame_num == 5:
        print("shape of D is", D.shape)
        print("new_K =\n", new_K)
        print("K =\n", K)
        print("D =\n", D)
        print ("height = ", h, "width = ", w)

    # Generate rectification maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )

    # Remap image
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted


def direct_undistort(img):
    h, w = img.shape[:2]
    
    # Map the source image to the normalized plane
    undistorted_img = np.zeros_like(img)
    
    # Create mapping function for each pixel
    mapx = np.zeros((h, w), dtype=np.float32)
    mapy = np.zeros((h, w), dtype=np.float32)
    
    # Get normalized pixel coordinates
    for y in range(h):
        for x in range(w):
            # Convert to normalized coordinates
            x_normalized = (x - K[0, 2]) / K[0, 0]
            y_normalized = (y - K[1, 2]) / K[1, 1]
            
            # Calculate radius
            r = np.sqrt(x_normalized**2 + y_normalized**2)
            
            # Skip if the radius is too large (might be beyond the fisheye)
            if r > 1.0:
                continue
                
            # Apply fisheye distortion model
            theta_d = r
            theta = theta_d * (1 + D[0, 0]*theta_d**2 + D[1, 0]*theta_d**4 + 
                               D[2, 0]*theta_d**6 + D[3, 0]*theta_d**8)
            
            # Scale by theta/r to get the undistorted point
            scale = 1.0 if r == 0 else theta/r
            x_undistorted = scale * x_normalized
            y_undistorted = scale * y_normalized
            
            # Convert back to pixel coordinates
            mapx[y, x] = x_undistorted * K[0, 0] + K[0, 2]
            mapy[y, x] = y_undistorted * K[1, 1] + K[1, 2]
    
    # Use remap to create the undistorted image
    undistorted = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, 
                           borderMode=cv2.BORDER_CONSTANT)
    
    return undistorted


def improved_undistort(img, fov_scale=0.7, center_offset_x=0.0, center_offset_y=0.0):
    h, w = img.shape[:2]
    
    # Create a custom camera matrix with controlled parameters
    new_K = K.copy()
    
    # 1. Scale the focal length to control field of view
    new_K[0, 0] = K[0, 0] * fov_scale
    new_K[1, 1] = K[1, 1] * fov_scale
    
    # 2. Adjust the principal point to shift the center of projection
    # This can help with the smearing on one side
    new_K[0, 2] = K[0, 2] + (center_offset_x * w)  # Shift center point horizontally
    new_K[1, 2] = K[1, 2] + (center_offset_y * h)  # Shift center point vertically
    
    # Create maps with explicit dimensions
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_32FC1)
    
    # Apply remapping with optimized border handling
    undistorted = cv2.remap(img, map1, map2, 
                           interpolation=cv2.INTER_LANCZOS4,  # Higher quality interpolation
                           borderMode=cv2.BORDER_REFLECT)  # Better border handling
    
    return undistorted


def find_distortion_params():
    # Prepare object points (the 3D points of the chessboard pattern)
    obj_points = []  # 3D points in the world coordinate system
    img_points = []  # 2D points in the image coordinate system

    # Define the chessboard pattern size (number of inner corners per chessboard row and column)
    pattern_size = (7, 7)  # 7 internal corners (pattern_size-1) along both axes

    # Define square size (in some units, e.g., millimeters or centimeters)
    square_size = 1.0  # Set this based on your chessboard square size

    # List of images in the folder
    images = os.listdir("homography_pics")

    # Loop through each image file
    for image_name in images:
        # Construct the full path to the image file
        image_path = os.path.join("homography_pics", image_name)
        
        # Read the image
        img = cv2.imread(image_path)

        # Check if the image was loaded successfully
        if img is None:
            print(f"Error: Could not load image {image_path}")
            continue
        
        # Display the image for 1 second before detecting corners
        cv2.imshow(f"Image: {image_name}", img)
        cv2.waitKey(1000)  # Wait for 1000 ms (1 second)
        cv2.destroyAllWindows()  # Close the window after 1 second
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size)
        
        if ret:
            # If corners are found, refine their positions for accuracy
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            
            # Draw the chessboard corners on the image
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            
            # Display the image with detected corners for 1 second
            cv2.imshow(f"Chessboard with corners - {image_name}", img)
            cv2.waitKey(1000)  # Wait for 1000 ms (1 second)
            cv2.destroyAllWindows()  # Close the window after 1 second

            # Add the corners to the image and object points for calibration
            img_points.append(corners)
            
            # Create the corresponding 3D object points (in real-world coordinates)
            obj_point = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
            obj_point[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
            obj_point *= square_size  # Scale by the actual square size
            obj_points.append(obj_point)
        else:
            print(f"No corners detected in image: {image_name}")

    # Perform camera calibration to obtain the camera matrix and distortion coefficients
    if obj_points and img_points:
        print(f"Number of obj_points: {len(obj_points)}")
        print(f"Number of img_points: {len(img_points)}")
        
        # Perform camera calibration using the object and image points
        # Reshape objectPoints and imagePoints
        obj_points_reshaped = [objp.reshape(1, -1, 3) for objp in obj_points]
        img_points_reshaped = [imgp.reshape(1, -1, 2) for imgp in img_points]

        K = np.zeros((3, 3))
        D = np.zeros((4, 1))

        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

        ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_points_reshaped,
            img_points_reshaped,
            gray.shape[::-1],  # image size (width, height)
            K,
            D,
            None,
            None,
            flags=flags,
            criteria=criteria
        )
        
        if ret:
            print("Camera calibration successful!")
            print("Intrinsic matrix (K):\n", K)
            print("Distortion coefficients (D):\n", D)
        else:
            print("Camera calibration failed!")
    else:
        print("No valid chessboard corners found in images.")

    # Close all windows after the process
    cv2.destroyAllWindows()


# find_distortion_params()

# print("K shape:", K.shape, "dtype:", K.dtype)
# print("D shape:", D.shape, "dtype:", D.dtype)

# image_for_experiments = ImageParse.resize_proportionally(cv2.imread('homography_pics\WIN_20250407_21_25_44_Pro.jpg'),0.5)
# directly_undistorted = improved_undistort(image_for_experiments)
# undistorted = undistort(image_for_experiments, 5)

# while True:
#     cv2.imshow("originial", image_for_experiments)
#     cv2.imshow("directly undistorted", directly_undistorted)
#     cv2.imshow("undistorted", undistorted)    
#     # Press Escape to exit
#     if cv2.waitKey(1) == 27:
#         break
