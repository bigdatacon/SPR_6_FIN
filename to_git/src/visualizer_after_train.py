from mmseg.apis import inference_model, init_model, show_result_pyplot
import os
import glob

# === Пути к конфигу и чекпоинту ===
config_file = r"C:\Users\Admin\NEIRO\SPR_6_MMDETECTION\mmsegmentation\configs\unet_practice\unet_256x256.py"
checkpoint_file = r"C:\Users\Admin\NEIRO\SPR_6_MMDETECTION\mmsegmentation\work_dirs\unet_train_1_256x256\epoch_6.pth"

# Берем предпоследнюю папку из пути к конфигу (например, 'unet_practice')
config_folder_name = os.path.basename(os.path.dirname(config_file))

# Загружаем модель на GPU (или 'cpu', если нет GPU)
model = init_model(config_file, checkpoint_file, device='cpu')

# Папка с изображениями для инференса
test_img_dir = r"C:\Users\Admin\NEIRO\SPR_6_MMDETECTION\mmsegmentation\data\train_dataset_for_students_c\img\test"

# Папка, куда сохраняем результаты: добавляем имя конфиг-папки
out_dir = os.path.join(
    r"C:\Users\Admin\NEIRO\SPR_6_MMDETECTION\mmsegmentation\src\train_dataset_for_students_c",
    config_folder_name
)
os.makedirs(out_dir, exist_ok=True)

# Получаем список всех jpg изображений и берём первые 5
img_list = glob.glob(os.path.join(test_img_dir, "*.jpg"))
img_list = img_list[:5]

for img_path in img_list:
    # Выполняем инференс
    result = inference_model(model, img_path)

    # Имя файла результата
    base_name = os.path.basename(img_path)
    out_file = os.path.join(out_dir, f"infer_{base_name}")

    # Сохраняем изображение с оверлеем
    show_result_pyplot(
        model,
        img_path,
        result,
        opacity=0.5,
        with_labels=True,
        draw_gt=False,
        show=False,
        out_file=out_file
    )
    print(f"Сохранено: {out_file}")