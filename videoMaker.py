import cv2
import os

image_folder = 'graphs'
video_name = 'VmRec-75-encoder-1-decoder-1-epochs-80000-window_size-75.avi'

images = [img for img in os.listdir(image_folder) if (img.startswith("VmRec-75-encoder-1-decoder-1-epochs-") and img.endswith("window_size-75.png"))]
images.sort()
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()