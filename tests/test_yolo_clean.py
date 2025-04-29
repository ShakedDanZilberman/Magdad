import numpy as np
from PIL import Image

# Generate a random 128x128 image
random_image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)  # Single channel for grayscale

# Convert to a PIL image and save or display
image = Image.fromarray(random_image, mode='L')  # 'L' mode for grayscale
image.save("random_image_grayscale.png")  # Save the image
image.show()  # Display the image