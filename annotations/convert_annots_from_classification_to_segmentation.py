import json
import os
import fire
from loguru import logger
import numpy as np
from PIL import Image
import pycocotools.mask as _mask

@logger.catch(reraise=True)
def convert_annots_from_classification_to_segmentation(
    annotations_file: str,
    manipulated_masks_folder: str,
    output_file: str,
) -> None:
    """This function can be used to reorganize the set of categories of a
    dataset.

    Concretely, it can be use to merge sets of categories. The new annotations
    are stored into the specified output .json file.

    Args:
        annotations_file (str): file with classification annotation
        manipulated_masks_folder (str): Path where the masks are stored
        output_file (str): new segmentation annotations file

    Returns:
        metadata: Dictionary with the annotations.

    """

    obj = json.load(open(annotations_file))
    images_obj = obj["images"]
    annotations_new = []


    # masks_path = args.masks_folder

    previous_image_id = -1
    new_image = 0

    count_images_gt_not_found = 0
    for image in images_obj:

        annotation = {}
        annotation["id"] = image["id"]
        annotation["image_id"] = image["id"]
        annotation["area"] = image["width"]*image["height"]    
        annotation["bbox"] = [0, 0, image["width"], image["height"] ]
        annotation["iscrowd"] = 0
        annotation["score"] = 1.0
        annotation["category_id"] = image["category_id"]

        size = {"w": 0, "h": 0}
        image_id = annotation["image_id"]
        if image_id == previous_image_id:
            continue
        else:
            new_image += 1

        file_name = ""
        find_values = False
        for image in images_obj:
            if image["id"] == image_id:
                head, tail = os.path.split(image["file_name"])
                file_name = tail
                size["w"] = image["height"]
                size["h"] = image["width"]
                find_values = True

        if find_values is False:
            logger.info("Error reading information")
            raise Exception("Data not found")

        logger.info(
            f"Updating information for file  {file_name} ... {new_image} images created"
        )

        if annotation["category_id"] == 0:
            mask_image = np.zeros([size["w"], size["h"]],dtype=np.uint8)
            mask_file = file_name
        else:
            filename = os.path.splitext(file_name)[0]
            mask_file = os.path.join(
                manipulated_masks_folder, filename + "_gt.png"
            )
            print(mask_file)

            if not os.path.isfile(mask_file):
                count_images_gt_not_found += 1
                continue

            logger.info(f"Images not found {count_images_gt_not_found} ******************")
            # load the image
            mask_image_dataset = np.array(Image.open(mask_file))

            if (len(mask_image_dataset.shape)==3):
                mask_image = mask_image_dataset[:,:,1] 
            else:
                mask_image = mask_image_dataset
            mask_image[mask_image >= 100] = 255
            mask_image[mask_image < 100] = 0

            #########################################

        print((mask_image.shape))
        assert len(mask_image.shape) == 2
        data = np.asfortranarray(mask_image)
        w, h = data.shape
        data = np.asfortranarray(data).astype("uint8")
        annotation["bbox"] = [0, 0, size["w"], size["h"]]

        compute_rle = True
        if compute_rle is True:
            logger.info(f" .... Computing RLE encoding for mask  {mask_file} ...")
            rle = _mask.encode(data)

            annotation["segmentation"] = {}
            annotation["segmentation"]["size"] = [size["w"], size["h"]]
            # annotation["segmentation"]['counts'] = codecs.decode(rle['counts'], 'utf-8')
            annotation["segmentation"]["counts"] = rle["counts"].decode("ascii")

        else:
            logger.info(f" .... Skipping RLE encoding for mask  {mask_file} ...")
            annotation["segmentation"] = {
                "size": [size["w"], size["h"]],
                "counts": "000",
            }
        annotations_new.append(annotation)

        previous_image_id = image_id

    obj["annotations"] = annotations_new
    with open(output_file, "w") as json_output:
        json.dump(obj, json_output)



if __name__ == "__main__":
    fire.Fire(convert_annots_from_classification_to_segmentation)
