import os
import numpy as np
from pathlib import Path
from dlib import rectangle


def remove_rotated_faces(paths: list[Path]) -> None:
    for path in paths:
        lbl_name = Path(str(path).replace(str(path.suffix), '.pts'))
        if lbl_name.is_file():
            with open(lbl_name, 'r') as f:
                points = f.read().replace('{','').replace('}','').split('\n')[3:-1]
            points = list(filter(None, points))
            if len(points) != 68:
                os.remove(path)
                os.remove(lbl_name)
        else:
            os.remove(path)

def bb_from_kpoints(path: str) -> np.array:
    with open (str(path).replace(str(path.suffix), '.pts')) as f:
        points_list = []
        points = f.read().replace('{','').replace('}','').split('\n')[3:-1]
        points = list(filter(None, points))
        for cords in points:
            points_list.append(list(map(float, cords.split(' ')[:2])))
    return np.hstack((np.array(points_list).min(axis=0), np.array(points_list).max(axis=0)))

def convert_and_trim_bb(image: np.array, rect) -> np.array:
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	return np.array((startX, startY, endX, endY))

def bbox_iou(box1: np.array, box2: np.array, xywh: bool = False, eps: float = 1e-7) -> float:
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        w1, h1 = np.clip(b1_x2 - b1_x1, eps, np.inf),  np.clip(b1_y2 - b1_y1, eps, np.inf)
        w2, h2 = np.clip(b2_x2 - b2_x1, eps, np.inf), np.clip(b2_y2 - b2_y1, eps, np.inf)

    # Intersection area
    inter = np.clip(np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1), 0, np.inf) * \
            np.clip(np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1), 0, np.inf)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    return iou

def dir_trees(dir: str) -> None:
    datasets = ['300W','Menpo']
    modes = ['train','test']
    rectangles = ['', '_rects']
    for dataset in datasets:
        for mode in modes:  
            for rect in rectangles:
                Path(f'{dir}{rect}/{dataset}/{mode}').mkdir(parents=True, exist_ok=True) 

def get_files(extensions: str, path: str) -> list[Path]:
    all_files = []
    for ext in extensions:
        all_files.extend(Path(path).rglob(ext))
    return all_files

def recalculate_points(path, new_path, shift_box):
    with open (str(path).replace(str(path.suffix), '.pts'), 'r') as f:
        points_list = []
        points_default_str = f.read()
        points = points_default_str.replace('{','').replace('}','').split('\n')[3:-1]
        points = list(filter(None, points))
        for cords in points:
            points_list.append(list(map(float, cords.split(' ')[:2])))
    recalc_np_X = np.array(points_list)[:, 0] - shift_box[0]
    recalc_np_Y = np.array(points_list)[:, 1] - shift_box[1]
    recalc_list = np.column_stack((recalc_np_X, recalc_np_Y)).tolist()
    recalct_full_str = '\n'.join(points_default_str.split('\n')[0:3] + 
                        [str(x)[1:-1].replace(',', '') for x in recalc_list] + 
                        list(['}']))
    with open(new_path.replace(str(Path(new_path).suffix), '_rect_box_points.txt'), 'w+') as f:
        f.write(recalct_full_str)
    