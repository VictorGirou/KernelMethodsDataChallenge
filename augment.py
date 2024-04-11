import numpy as np
from skimage import transform

def augment(data):
    n = data.shape[0]
    images = np.reshape(data,(n,32,32,3))
    augmented_images = []

    for image in images:
        # Randomly rotate the image between -10 and 10 degrees
        rotated_image = transform.rotate(image, np.random.uniform(-10, 10), mode='edge')

        # Randomly flip the image horizontally
        flipped_image = np.fliplr(rotated_image) if np.random.random() < 0.5 else rotated_image

        # Randomly shift the image horizontally and vertically by up to 10% of the image size
        shift_x = np.random.uniform(-0.1, 0.1) * image.shape[1]
        shift_y = np.random.uniform(-0.1, 0.1) * image.shape[0]
        shifted_image = transform.warp(flipped_image, transform.SimilarityTransform(translation=(shift_x, shift_y)))

        augmented_images.append(shifted_image)

    augmented_images = np.array(augmented_images)
    data = np.concatenate((data,np.reshape(augmented_images,(n,3072))))

    return data