import os
from mmseg.datasets import DummyDataset
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from mmseg.visualization import SegLocalVisualizer
from mmengine.registry import init_default_scope

# Инициализация области поиска mmseg
init_default_scope('mmseg')


def load_dummy_ds() -> DummyDataset:
    mmseg_root = os.path.dirname(os.path.abspath(__file__))

    data_root = os.path.join(mmseg_root, "..", "data", "train_dataset_for_students")
    data_prefix = dict(
        img_path=os.path.join("img", "train"),
        seg_map_path=os.path.join("labels", "train")
    )

    reading_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
    ]

    dataset = DummyDataset(
        data_root=data_root,
        data_prefix=data_prefix,
        pipeline=reading_pipeline,
        img_suffix=".jpg",
        seg_map_suffix=".png"
    )
    return dataset


def plot_sample_with_classes(ds):
    print(f"Загружен датасет длиной {len(ds)} элементов")
    ds_meta = ds.metainfo

    # Создаем абсолютный путь к папке для сохранения
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz_outputs")
    os.makedirs(save_dir, exist_ok=True)

    # Настройка визуализатора с LocalVisBackend
    seg_local_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend', save_dir=save_dir)],
        alpha=0.5,
    )
    seg_local_visualizer.dataset_meta = dict(
        classes=ds_meta["classes"],
        palette=ds_meta["palette"]
    )

    for idx in range(len(ds)):
        sample = ds[idx]
        img = sample["img"]

        data_sample = SegDataSample()
        data_sample.gt_sem_seg = PixelData(data=sample["gt_seg_map"])

        # Каждый файл будет иметь уникальное имя
        seg_local_visualizer.add_datasample(
            name=f"sample_{idx}",
            image=img,
            data_sample=data_sample,
            show=False,
            draw_pred=False,
            draw_gt=True
        )

    print(f"Визуализация завершена. Файлы сохранены в {save_dir}")


if __name__ == "__main__":
    ds = load_dummy_ds()
    plot_sample_with_classes(ds)