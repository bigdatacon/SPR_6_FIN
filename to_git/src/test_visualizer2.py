import os
from mmseg.datasets import TrainDatasetForStudentsC
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from mmseg.visualization import SegLocalVisualizer
from mmengine.registry import init_default_scope
import numpy as np

# Инициализация области поиска mmseg
init_default_scope('mmseg')

# Пути к данным
DATA_ROOT = "../data/train_dataset_for_students"
IMG_DIR = "img/train"
LABEL_DIR = "labels/train"

# Папка для сохранения визуализаций
SAVE_DIR = os.path.abspath("./viz_outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

# Метаданные классов
CLASS_NAMES = ["FON", "CAT", "DOG"]
PALETTE = [
    [0, 0, 0],       # фон
    [255, 0, 0],     # кошка — красная
    [0, 255, 0]      # собака — зелёная
]

def load_custom_dataset():
    data_prefix = dict(img_path=IMG_DIR, seg_map_path=LABEL_DIR)

    pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
    ]

    dataset = TrainDatasetForStudentsC(
        data_root=DATA_ROOT,
        data_prefix=data_prefix,
        pipeline=pipeline,
        img_suffix=".jpg",
        seg_map_suffix=".png"
    )
    return dataset

def visualize_dataset(ds):
    print(f"Загружен датасет длиной {len(ds)} элементов")

    visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend', save_dir=SAVE_DIR)],
        alpha=0.5,
    )
    visualizer.dataset_meta = dict(classes=CLASS_NAMES, palette=PALETTE)

    for idx in range(len(ds)):
        sample = ds[idx]
        img = sample["img"]
        mask = sample["gt_seg_map"]

        data_sample = SegDataSample()
        data_sample.gt_sem_seg = PixelData(data=mask)

        # Попробуем получить оригинальное имя файла
        file_name = sample.get("img_path", None)
        if file_name is not None:
            file_name = os.path.splitext(os.path.basename(file_name))[0]
        else:
            file_name = f"sample_{idx}"

        # Добавляем уникальный индекс на случай дубликатов
        file_name = f"{file_name}_{idx}"

        visualizer.add_datasample(
            name=file_name,
            image=img,
            data_sample=data_sample,
            show=False,
            draw_pred=False,
            draw_gt=True
        )

    print(f"Визуализация завершена. Файлы сохранены в: {SAVE_DIR}")

if __name__ == "__main__":
    ds = load_custom_dataset()
    visualize_dataset(ds)