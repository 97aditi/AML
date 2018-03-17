import cv2
vidcap = cv2.VideoCapture('name of video')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  cv2.imwrite("frame%d.jpg" % count, image)
  count += 1
