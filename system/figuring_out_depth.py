import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_resize_image(image_path, max_width=1600, max_height=900):
    """
    Load an image and resize it if it's too large
    
    Args:
        image_path: str, path to the image file
        max_width: maximum width for the displayed image
        max_height: maximum height for the displayed image
        
    Returns:
        tuple (original_image, display_image, scale_factor)
    """
    try:
        # Load original image
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Get original dimensions
        height, width = original.shape[:2]
        
        # Calculate scale factor if image is too large
        scale_w = max_width / width if width > max_width else 1
        scale_h = max_height / height if height > max_height else 1
        scale = min(scale_w, scale_h)
        
        # Resize if necessary
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            display = cv2.resize(original, (new_width, new_height), 
                               interpolation=cv2.INTER_AREA)
            print(f"Image resized from {width}x{height} to {new_width}x{new_height}")
        else:
            display = original.copy()
            scale = 1
            
        return original, display, scale
        
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None, None, None

def mouse_callback(event, x, y, flags, param):
    """Callback function to handle mouse events"""
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates = param['coordinates']
        if len(coordinates) < 2:
            # Convert display coordinates to original image coordinates
            orig_x = int(x / param['scale'])
            orig_y = int(y / param['scale'])
            coordinates.append((orig_x, orig_y))
            print(f"Clicked at position ({orig_x}, {orig_y}) in original image")
            
def fit_and_draw_line(y1, y2, target_y_diff):
    """
    Fit and draw a line between two points with a specified y difference
    
    Args:
        image: numpy array representing the image
        points: list of two (x, y) coordinate tuples
        target_y_diff: desired y-difference between points
    """
    # if len(points) != 2:
    #     return
    
    # x1, y1 = points[0]
    # x2, y2 = points[1]
    
    slope = target_y_diff/np.abs(y1-y2)
    return slope

def get_two_coordinates(original_image, display_image, scale, target_y_diff):
    window_name = "Click two points (ESC to exit)"
    cv2.namedWindow(window_name)
    
    param = {
        'coordinates': [],
        'original_image': original_image,
        'display_image': display_image,
        'scale': scale,
        'window_name': window_name,
        'y_diff': target_y_diff
    }
    
    cv2.setMouseCallback(window_name, mouse_callback, param)
    
    # Keep window open until we have 2 coordinates or ESC is pressed
    while True:
        cv2.imshow(window_name, display_image)
        key = cv2.waitKey(1) & 0xFF
        
        # Exit if ESC pressed or we have 2 coordinates
        if key == 27 or len(param['coordinates']) >= 2:
            break
    
    cv2.destroyAllWindows()
    return param['coordinates']

def get_z_values(y1, z1, y2, z2, y):
    # Set the image path and y-difference
    # image_path = "WIN_20250105_20_42_05_Pro.jpg"
    target_z_diff = np.abs(z1-z2)  # Change this value to adjust the line's y-difference

    # # Load and resize image if necessary
    # original_img, display_img, scale = load_and_resize_image(image_path)
    # if original_img is None:
    #     return

    # Get coordinates and fit line
    # coords = get_two_coordinates(original_img, display_img, scale, target_y_diff)
    # print("Selected coordinates (in original image):", coords)
    # x1, y1 = coords[0]
    # x2, y2 = coords[1]
    slope = fit_and_draw_line(y1, y2, target_z_diff)
    print("slope, real distance", slope, target_z_diff)
    return slope*y
    # y = np.linspace(-5,5,100)
    # z = slope*y
    # plt.plot(y, z, '-r', label='plot')
    # plt.title('Graph')
    # plt.xlabel('y', color='#1C2833')
    # plt.ylabel('z', color='#1C2833')
    # plt.legend(loc='upper left')
    # plt.grid()
    # plt.show()