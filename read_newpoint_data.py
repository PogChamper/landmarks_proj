from utils import get_files
from tqdm import tqdm

import numpy as np
import pandas as pd
from utils import get_files
from pathlib import Path

def read_newpointdata(paths):
    points_dict = {}
    for path in paths:
        with open(str(Path(path)).replace(str(Path(path).suffix),
                                          '_rect_box_points.txt'),'r') as cur_file:
            points = cur_file.read().replace('{','').replace('}','').split('\n')[3:-1]
            points = list(filter(None, points))
            points_list = []
            for cords in points:
                points_list.append(list(map(float, cords.split(' ')[:2])))
            
        with open(str(str(Path(path))).replace(str(Path(path).suffix), 
            '_rect_box.txt'),'r') as cur_file:
            lines = cur_file.read()

        
        xses = (np.array(points_list)[:, 0] / (list(map(int, lines.split(' ')))[2] - 
                                               list(map(int, lines.split(' ')))[0])).reshape(-1, 1)
        yses = (np.array(points_list)[:, 1] / (list(map(int, lines.split(' ')))[3] - 
                                               list(map(int, lines.split(' ')))[1])).reshape(-1, 1)
        
        box_w = list(map(int, lines.split(' ')))[2] - list(map(int, lines.split(' ')))[0] # на это делим сейчас, чтобы восстановить умножим
        box_h = list(map(int, lines.split(' ')))[3] - list(map(int, lines.split(' ')))[1] # на это делим сейчас, чтобы восстановить умножим

        box_corn_x = list(map(int, lines.split(' ')))[0] # это прибавим, чтобы конверт точки в старые корд
        box_corn_y = list(map(int, lines.split(' ')))[1] # это прибавим, чтобы конверт точки в старые корд

        points_list = np.hstack((xses, yses))
        points_list = np.concatenate((np.squeeze(np.array(points_list).reshape(1, -1)), np.array((box_w, box_h, box_corn_x, box_corn_y))))
        points_dict[str(Path(path)).replace(
            'landmarks_task_rects', 'landmarks_task_rects'
            )] = points_list
            
    df = pd.DataFrame.from_dict(points_dict, orient='index')
    df.reset_index(inplace=True)
    return df

# def main():
#     PATH = '/home/baishev/projects/landmarks/'
#     PATH_DATA = PATH+str('data/landmarks_task_rects')
    
#     paths = get_files(('*.png', '*.jpg'), PATH_DATA)
#     df_prepared = read_newpointdata(paths)
#     df_prepared.to_csv(f'{PATH}/df.csv')

# if __name__ == '__main__':
#     main()