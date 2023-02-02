import os

from PIL import Image
from tqdm import tqdm

img_folder = "/shared/aifiles/disk1/media/twitter/v10/covid19-twitter-images"
with os.scandir(img_folder) as it:
    for file in tqdm(it):
        file_path = os.path.join(img_folder, file)
        try:
            Image.open(file_path)
        except Exception as ex:
            print(ex)
            os.remove(file_path)
print("DONE")
