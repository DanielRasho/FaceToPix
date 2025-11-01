import os
import os.path as path
import random
import subprocess

NEW_PATH = "sample_pix/"
if not path.exists(NEW_PATH):
    os.makedirs(NEW_PATH)

DATA_PATH = "./upload_data/pyxel/"
imgs = [i for i in os.listdir(DATA_PATH) if path.isfile(path.join(DATA_PATH, i))]

for _ in range(1500):
    while True:
        imgName = random.choice(imgs)
        targetPath = path.join(NEW_PATH, imgName)
        srcPath = path.join(DATA_PATH, imgName)

        try:
            subprocess.run(["cp", srcPath, targetPath])
            break
        except Exception as ex:
            print(f"Failed to copy: {srcPath}->{targetPath}\n{ex}")
            pass
