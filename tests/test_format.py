import json
from pathlib import Path

import pytest
from pycocotools.coco import COCO

from annotations.SCHEMA import COCOSegmentationDataset


ANNOTATION_FILE = str(Path(__file__).parent.parent / "annotations" / "CASIA_ITDE_SEGMENTATION.json")


@pytest.mark.skipif(not Path(ANNOTATION_FILE).exists(), reason="No access to DVC cache")
def test_coco_format():
    dataset = COCOSegmentationDataset(**json.load(open(ANNOTATION_FILE)))

    image_ids = [x.id for x in dataset.images]
    image_ids_set = set(image_ids)
    # No repeated image ids
    assert len(image_ids) == len(image_ids_set)

    categories = [x.id for x in dataset.categories]
    categories_set = set(categories)
    # No repeated categories
    assert len(categories) == len(categories_set)

    annotation_ids = [x.id for x in dataset.annotations]
    annotation_ids_set = set(annotation_ids)
    # No repeated annotation ids
    assert len(annotation_ids) == len(annotation_ids_set)

    for annotation in dataset.annotations:
        assert annotation.image_id in image_ids_set
        assert annotation.category_id in categories_set


@pytest.mark.skipif(not Path(ANNOTATION_FILE).exists(), reason="No access to DVC cache")
def test_segmentation_mask_decoding():
    coco_annotations = COCO(ANNOTATION_FILE)
    for image_id, image_data in coco_annotations.imgs.items():
        anns_ids = coco_annotations.getAnnIds(imgIds=image_id)
        image_annotations = coco_annotations.loadAnns(anns_ids)
        for image_annotation in image_annotations:
            coco_annotations.annToMask(image_annotation)