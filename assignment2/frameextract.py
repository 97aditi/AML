import cv2
vidcap = cv2.VideoCapture('\youtubevids\maheshwari\m_vid1.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  img = cv2.resize(image, (256,256))
  cv2.imwrite("\youtubevids\mvid1frames\mr%d.jpg" % count, img)
  count += 1
