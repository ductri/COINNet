import glob
import os
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    IDs = ['n02056570', 'n02085936', 'n02128757', 'n02690373', 'n02692877', 'n02123159', 'n03095699', 'n02687172', 'n04285008', 'n04467665', 'n04254680', 'n07747607', 'n07749582', 'n03393912', 'n03895866']

    my_dict = {}
    root = '/scratch/tri/datasets/imagenet/batch_0/'
    # for class_id in IDs:
    #     os.mkdir(f'{root}/{class_id}')
    for class_id in tqdm(IDs):
        all_images = glob.glob(f"/scratch/tri/datasets/imagenet/batch_0/{class_id}_*")
        # print(f'#images: {len(all_images)} for {class_id}')
        for image_path in all_images:
            image_path = Path(image_path)
            parent_path = image_path.parent
            os.rename(image_path, f'{str(image_path.parent)}/{class_id}/{image_path.name}')
        print()


