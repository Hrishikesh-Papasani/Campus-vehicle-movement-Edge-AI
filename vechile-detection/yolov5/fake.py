import os
from PIL import Image

def verify_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    img.verify()  # Verify if image can be opened
                except (IOError, SyntaxError) as e:
                    print(f'Bad file: {img_path}')

# Verify training images
verify_images('C:\\Users\\yamin\\OneDrive\\Desktop\\yk\\dataset\\vehicle dataset\\train\\images')

# Verify validation images
verify_images('C:\\Users\\yamin\\OneDrive\\Desktop\\yk\\dataset\\vehicle dataset\\valid\\images')
