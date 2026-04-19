import numpy as np
import matplotlib.pyplot as plt
import os

import image_io as io
import geometric_transformation as gt


OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# read image
img = io.read_image(r"C:\Users\Mohammed_AlJamal\OneDrive\Desktop\milestone 1\minicv\me_and_friends.jpeg", as_gray=True)


# reduce size for faster processing
img_small = gt.resize_image(img, 0.3, 0.3)

print("Reduced size:", img_small.shape)


# transformations
rotated_img = gt.rotate_image(img_small, 30)

resized_img = gt.resize_image(img_small, 0.7, 0.7)

translated_img = gt.translate_image(img_small, 30, 20)


# show results
plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.title("Original")
plt.imshow(img_small, cmap="gray")
plt.axis("off")


plt.subplot(1,4,2)
plt.title("Rotated")
plt.imshow(rotated_img, cmap="gray")
plt.axis("off")


plt.subplot(1,4,3)
plt.title("Resized")
plt.imshow(resized_img, cmap="gray")
plt.axis("off")


plt.subplot(1,4,4)
plt.title("Translated")
plt.imshow(translated_img, cmap="gray")
plt.axis("off")


plt.tight_layout()
plt.show()


# save images
plt.imsave(os.path.join(OUTPUT_DIR,"figure4_original.jpeg"), img_small, cmap="gray")

plt.imsave(os.path.join(OUTPUT_DIR,"figure4_rotated.jpeg"), rotated_img, cmap="gray")

plt.imsave(os.path.join(OUTPUT_DIR,"figure4_resized.jpeg"), resized_img, cmap="gray")

plt.imsave(os.path.join(OUTPUT_DIR,"figure4_translated.jpeg"), translated_img, cmap="gray")


print("\nFigure 4 saved in:", OUTPUT_DIR)