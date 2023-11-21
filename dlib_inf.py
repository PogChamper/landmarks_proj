import dlib
import cv2
from tqdm import tqdm
from utils import *
from pathlib import Path
import numpy as np

def get_bbox_preds(path: Path, detector, iou_threshold: float) -> np.array:
    print(path)
    try:
        img = dlib.load_rgb_image(str(path))
    except:
        print('Corrupted image file')
        return None
    dets = detector(img, 1)
    if dets:
        bb_kpoins = bb_from_kpoints(Path(str(path).replace(str(path.suffix), '.pts')))
        if len(dets) > 1:
            boxes = np.array([convert_and_trim_bb(img, r) for r in dets if len(dets) > 1])
            ious = bbox_iou(bb_kpoins, boxes)
            if max(ious) > iou_threshold:
                ok_box = boxes[np.argmax(ious)]
            else:
                print('Box from dlib dont correspond to points')
                return None
        else:
            ok_box = convert_and_trim_bb(img, dets[0])
        x1, x2 = int(ok_box[0] - (ok_box[2]-ok_box[0])*1.75 / 2) , int(ok_box[2] + (ok_box[2]-ok_box[0])*1.75 / 2)
        y1, y2 = int(ok_box[1] - (ok_box[3]-ok_box[1])*1.75 / 2) , int(ok_box[3] + (ok_box[3]-ok_box[1])*1.75 / 2)
        ext_ok_box = convert_and_trim_bb(img, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)) #np.array((x1, y1, x2, y2))
        if not (bb_kpoins[0] > ext_ok_box[0]) & (bb_kpoins[1] > ext_ok_box[1]) & (bb_kpoins[2] < ext_ok_box[2]) & (bb_kpoins[3] < ext_ok_box[3]):
            print('DOTS DONT FIT THE BOX EVEN AFTER EXTENTION')
        else:
            return ext_ok_box, ok_box

def main():
    PATH = '/home/baishev/projects/landmarks/'
    PATH_DATA = PATH+str('data/landmarks_task')
    
    paths = get_files(('*.png', '*.jpg'), PATH_DATA)
    a1 = len(paths)
    remove_rotated_faces(paths)
    paths = get_files(('*.png', '*.jpg'), PATH_DATA)
    a2 = len(paths)
    c = 0
    dir_trees(PATH_DATA)
    detector = dlib.get_frontal_face_detector()
    for path in tqdm(paths):
        ok_box = get_bbox_preds(path, detector, 0.3)
        new_path = str(path).replace(f'{PATH_DATA.split("/")[-1]}', f'{PATH_DATA.split("/")[-1]}_rects')
        img = cv2.imread(str(path))
        if ok_box is not None:
            c+=1
            cv2.imwrite(new_path, img[ok_box[0][1]:ok_box[0][3], ok_box[0][0]:ok_box[0][2]])
            with open(new_path.replace(str(Path(new_path).suffix),'_rect_box.txt'),'w') as f:
                f.write(' '.join(list(map(str, ok_box[0]))))
            with open(new_path.replace(str(Path(new_path).suffix),'_rect_box_old.txt'),'w') as f:
                f.write(' '.join(list(map(str, ok_box[1]))))
            recalculate_points(path, new_path, ok_box[0])
    print(a1, a2, c)

if __name__ == '__main__':
    main()