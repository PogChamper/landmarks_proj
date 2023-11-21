# landmarks_proj
Facial Landmark Detection task

cd landmarks_proj
mkdir data
распакуйте landmarks_task архив в data

Получение ограничивающих прямоугольников с лицами:
python3 dlib_inf.py 

Обучение модели:
python3 train_onet.py --max_epoch 150 --output_path 'Onet-default-train' --batch_size 16

Инференс dlib на датасете Menpo:
python3 dlib_full_inf.py --dataset 'Menpo'

Инференс O-Net на датасете Menpo:
python3 train_onet.py --mode 'test' --dataset 'Menpo' --output_path 'Onet-default-train'

Инференс O-Net на датасете 300W:
python3 train_onet.py --mode 'test' --dataset '300W' --output_path 'Onet-default-train'

Подсчет метрик и построение графиков для результатов dlib и O-Net на датасете Menpo:
python3 count_ced_for_points.py --gt_path 'data/landmarks_task' --predictions_path 'data/result_onet' --predictions_path 'data/result_dlib' --output_path 'Menpo-dlib-onet' --max_points_to_read 100000 --dataset 'Menpo' --error_thr 0.08

Подсчет метрик и построение графиков для результатов O-Net на датасете 300W:
python3 count_ced_for_points.py --gt_path 'data/landmarks_task' --predictions_path 'data/result_onet' --output_path '300W-onet' --max_points_to_read 100000 --dataset '300W' --error_thr 0.08
