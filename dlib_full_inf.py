from dlib import rectangle
import dlib
import argparse
from imutils import face_utils
import os
import glob
from tqdm import tqdm
from pathlib import Path

def make_all_dirs(res_dir):
    datasets = ['300W','Menpo']
    modes = ['train','test']
    for dataset in datasets:
        for mode in modes:
            if not os.path.exists(f'data/{res_dir}/{dataset}/{mode}'):
                os.makedirs(f'data/{res_dir}/{dataset}/{mode}')

def main():
    parser = argparse.ArgumentParser(description='Onet train and inference script',
                                     add_help=True)
    
    parser.add_argument('--result_dir', action='store', type=str, help='',
                        default='result_dlib')
    parser.add_argument('--dataset', action='store', type=str, help='',
                        default='300W')
    args = parser.parse_args()
    make_all_dirs(args.result_dir)
    predictor = dlib.shape_predictor('data/dlib_model/shape_predictor_68_face_landmarks.dat')
    paths1 = glob.glob(f'data/landmarks_task_rects/{args.dataset}/test/*.jpg')
    paths2 = glob.glob(f'data/landmarks_task_rects/{args.dataset}/test/*.png')
    paths = paths1 + paths2
    #print(paths)
    
    count = 0
    print(len(paths))
    for path in tqdm(paths):
        new_path = path.replace('landmarks_task_rects','landmarks_task')
        rgb = dlib.load_rgb_image(new_path)
        with open(path.replace(str(Path(path).suffix),'_rect_box_old.txt'),'r') as f:
            txt = f.read()
        det = rectangle(*list(map(int,txt.split(' '))))
        shape = predictor(rgb, det)
        shape = face_utils.shape_to_np(shape)
        if len(shape.reshape(-1).tolist())!=136:
            count+=1
        else:
            with open(path.replace('landmarks_task_rects',args.result_dir).replace(str(Path(path).suffix),'.pts'),'w') as f:
                #f.write(' '.join(list(map(str,shape.reshape(-1).tolist()))))
                f.write('\n'.join('version: 1\nn_points:  68\n{'.split('\n') + 
                        [str(x)[1:-1].replace(',', '') for x in shape.tolist()] + 
                        list(['}'])))
    print(f"NUMBER IMAGES WHERE DLIB DIDN'T FOUND 68 POINTS: {count}")
        

if __name__ == '__main__':
    main()