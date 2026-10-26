import os
import shutil
import urllib.request

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)

HAAR_NAME = "haarcascade_frontalface_default.xml"
YUNET_NAME = "face_detection_yunet_2023mar.onnx"

HAAR_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/" + HAAR_NAME,
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/" + HAAR_NAME,
]
YUNET_URLS = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/" + YUNET_NAME,
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + YUNET_NAME,
]


def download(urls, dest, min_bytes):
    for url in urls:
        try:
            print("Trying:", url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            if os.path.getsize(dest) >= min_bytes:
                print("  OK", os.path.getsize(dest), "bytes")
                return True
            print("  too small, trying next...")
        except Exception as e:
            print("  failed:", e)
    if os.path.exists(dest) and os.path.getsize(dest) < min_bytes:
        os.remove(dest)
    return False


def haar_ok(path):
    try:
        return os.path.exists(path) and os.path.getsize(path) > 10000 and not cv2.CascadeClassifier(path).empty()
    except Exception:
        return False


def main():
    print("OpenCV", cv2.__version__)
    haar_dest = os.path.join(HERE, HAAR_NAME)
    yunet_dest = os.path.join(HERE, YUNET_NAME)

    if not haar_ok(haar_dest):
        parent_file = os.path.join(PARENT, HAAR_NAME)
        if haar_ok(parent_file):
            shutil.copyfile(parent_file, haar_dest)
            print("Copied working Haar file from", parent_file)
        else:
            download(HAAR_URLS, haar_dest, 10000)
    print("Haar cascade :", "READY" if haar_ok(haar_dest) else "NOT available")

    if not (os.path.exists(yunet_dest) and os.path.getsize(yunet_dest) > 100000):
        download(YUNET_URLS, yunet_dest, 100000)
    print("YuNet model  :", "READY" if os.path.exists(yunet_dest) and os.path.getsize(yunet_dest) > 100000
          else "not downloaded (optional)")

    if not haar_ok(haar_dest) and not os.path.exists(yunet_dest):
        print("\nNo model available. Check your internet connection and run this script again.")
    else:
        print("\nDone. Now run: python chronos_companion.py --demo")


if __name__ == "__main__":
    main()