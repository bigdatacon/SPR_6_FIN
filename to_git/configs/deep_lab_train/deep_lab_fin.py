_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', 
    '../_base_/datasets/train_dataset_for_students_fin.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_sanity_check1.py'
]



visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),      # сохраняем логи локально
        dict(
            type='ClearMLVisBackend',      # дублируем всё в ClearML
            init_kwargs=dict(
                project_name='YaPracticum',
                task_name='unet-s5-d16_fcn_4xb4-practice_dataset-512x512',
                reuse_last_task_id=False,
                continue_last_task=False,
                output_uri=None,
                auto_connect_arg_parser=True,
                auto_connect_frameworks=True,
                auto_resource_monitoring=True,
                auto_connect_streams=True,
            )
        )     
    ]
)


# Определим размер входа 
# input_suze = (256, 256)
# data_preprocessor = dict(size=input_suze)


model = dict(
        data_preprocessor=dict(
        type='SegDataPreProcessor',
        size=(256, 256)
    ),
    decode_head=dict(
        # num_classes=3,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=1.0
            ),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)
        ]
    ),
    auxiliary_head=dict(
        # num_classes=3,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=1.0
            ),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)
        ]
    )
)




# model = dict(
#     data_preprocessor=data_preprocessor, # добавил эту строку относительно того что было на уроке иначе не работало
#     test_cfg=dict(mode="whole"),
#     decode_head=dict(
#         num_classes=3,
#         loss_decode=[
#             dict(
#                 type='CrossEntropyLoss',
#                 loss_name='loss_ce',
#                 use_sigmoid=False,
#                 loss_weight=1.0
#             ),
#             dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)
#         ]
#     ),
#     auxiliary_head=dict(
#         num_classes=3,
#         loss_decode=[
#             dict(
#                 type='CrossEntropyLoss',
#                 loss_name='loss_ce',
#                 use_sigmoid=False,
#                 loss_weight=1.0
#             ),
#             dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)
#         ]
#     )
# )
# Run
# C:/Users/Admin/NEIRO/SPR_6_MMDETECTION/practicum_venv/Scripts/python.exe tools/train.py configs/unet_train_ds_1/unet_train_1_256x256.py

# python3.10 tools/train.py /home/ubuntu/SPR_6_PETROV_E/mmsegmentation/configs/deep_lab_train/deep_lab_fin.py 