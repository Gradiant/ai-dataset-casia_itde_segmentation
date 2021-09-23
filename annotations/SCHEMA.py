from typing import get_args, List, Literal, Union
from pydantic import BaseModel, conint, confloat, conlist


CATEGORY_NAMES = Literal["apple", "banana"] 
CATEGORY_NAMES_LIST = get_args(CATEGORY_NAMES)


class COCOCategory(BaseModel):
    id: conint(ge=0, le=len(CATEGORY_NAMES_LIST))
    name: CATEGORY_NAMES
    supercategory: str = "object"


class COCOImage(BaseModel):
    id: int
    width: int
    height: int
    file_name: str


class COCORLE(BaseModel):
    size: conlist(int, min_items=2, max_items=2)
    counts: Union[str, List[int]]


class COCOAnnotation(BaseModel):
    id: int
    image_id: int
    area: confloat(gt=0.0)
    bbox: conlist(int, min_items=4, max_items=4)
    iscrowd: conint(ge=0, le=1) = 0.
    score: confloat(ge=0, le=1) = 1.
    category_id: conint(ge=0, le=len(CATEGORY_NAMES_LIST))
    segmentation: Union[COCORLE, List[List[float]]]


class COCOSegmentationDataset(BaseModel):
    images: List[COCOImage]
    annotations: List[COCOAnnotation]
    categories: List[COCOCategory]
