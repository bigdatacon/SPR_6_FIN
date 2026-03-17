dataset_type = 'TrainDatasetForStudentsC'
data_root = "/home/ubuntu/SPR_6_PETROV_E/mmsegmentation/data/train_dataset_for_students"
# data_root = "/home/ubuntu/SPR_6_PETROV_E/mmsegmentation/data/practice_dataset"
# data_root = "C:/Users/Admin/NEIRO/SPR_6_MMDETECTION/mmsegmentation/data/train_dataset_for_students"




# ==== Определяем обуающий пайплайн данных ======
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # Аугментацции 
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomRotFlip', degree=(-45, 45)),
    dict(type='RandomCutOut', prob=0.4, n_holes=(7, 15), cutout_ratio=(0.1, 0.15)),
    dict(type='Albu', transforms=[dict(type="GridDistortion", num_steps=10, p=1)]),
    # ===
    dict(type='PackSegInputs')
]
train_dataset=dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='img/train',
        seg_map_path='labels/train'),
    pipeline=train_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset   
)


# ==== Определяем валидационный пайплайн данных ======
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs")
]
test_pipeline = val_pipeline

val_dataset = dataset=dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='img/val',
        seg_map_path='labels/val'),
    pipeline=val_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset
)


# ==== Определяем тестовый пайплайн данных ======
test_dataset = dataset=dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='img/test',
        seg_map_path='labels/test'),
    pipeline=test_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)
test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset
)


# Здесь же в пайплайне данных создаются объекты для подсчета метрик
val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
test_evaluator = val_evaluator

# Run::
# C:/Users/Admin/NEIRO/SPR_6_MMDETECTION/practicum_venv/Scripts/python.exe tools/train.py configs/unet_practice/unet_256x256.py