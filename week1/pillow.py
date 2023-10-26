import  pillow
# 屏幕抓取
import time
from PIL import ImageGrab



# ImageGrab.grab().save("outimg.png")
img = ImageGrab.grab(bbox=(90,369,430,756))
img.save("outimg.png")