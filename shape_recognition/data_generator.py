import cv2
import numpy as np
import os



def generate_shapes(data_split: str, sample_num: int):
    """"""
    shapes = ["circle", "square", "triangle"]
    colors = [0, 255]

    for i in range(sample_num):
        # Pick a random shape
        shape = np.random.choice(shapes)

        # Generate an empty image
        color = int(np.random.choice(colors))
        image = np.full((128, 128), color, dtype = np.uint8)

        # Coordinations
        center_x = np.random.randint(44, 84)
        center_y = np.random.randint(44, 84)
        size = np.random.randint(10, 40)

        # Generate circle
        if shape == "circle":
            cv2.circle(image, (center_x, center_y), size, 255 - color, -1)

        # Generate square
        elif shape == "square":
            x1, y1 = center_x - size, center_y - size
            x2, y2 = center_x + size, center_y + size
            cv2.rectangle(image, (x1, y1), (x2, y2), 255 - color, -1)

        # Generate triangle
        else:
            points = np.array([
                [center_x - size, center_y + size], 
                [center_x + size, center_y - size],
                [center_x - size // 2, center_y - size // 3]
                ], np.int32)
            cv2.fillPoly(image, [points], 255 - color)
            
        # Save the image
        cv2.imwrite(filename = f"data/{data_split}/{shape}_{i}.png", img = image)


os.makedirs("data/train", exist_ok = True)
os.makedirs("data/test", exist_ok = True)

generate_shapes("train", 1500)
generate_shapes("test", 400)