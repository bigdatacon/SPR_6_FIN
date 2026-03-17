import shutil
from pathlib import Path
import random

# Оригинальный датасет
src_root = Path(r"C:\Users\Admin\NEIRO\SPR_6_MMDETECTION\mmsegmentation\data\train_dataset_for_students")

# Новый датасет
dst_root = Path(r"C:\Users\Admin\NEIRO\SPR_6_MMDETECTION\mmsegmentation\data\train_dataset_for_students_c")
dst_root.mkdir(parents=True, exist_ok=True)

# Обрабатываем train / val / test
splits = ["train", "val", "test"]

for split in splits:

    # Пути к картинкам и маскам
    img_folder = src_root / "img" / split
    mask_folder = src_root / "labels" / split

    # Если такой папки нет — пропускаем
    if not img_folder.exists():
        continue

    # Получаем имена файлов без расширения
    img_files = {f.stem: f for f in img_folder.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]}
    mask_files = {f.stem: f for f in mask_folder.glob("*.png")}

    # Формируем корректные пары
    matched = [name for name in img_files if name in mask_files]

    # Выбираем 2 случайные пары
    if len(matched) > 2:
        matched = random.sample(matched, 2)

    # Создаём новые папки
    dst_img_folder = dst_root / "img" / split
    dst_mask_folder = dst_root / "labels" / split
    dst_img_folder.mkdir(parents=True, exist_ok=True)
    dst_mask_folder.mkdir(parents=True, exist_ok=True)

    # Копируем выбранные пары
    for name in matched:
        shutil.copy(img_files[name], dst_img_folder / img_files[name].name)
        shutil.copy(mask_files[name], dst_mask_folder / mask_files[name].name)
        print(f"{split}: скопирована пара {img_files[name].name} + {mask_files[name].name}")

print(f"\nГотово! Новый датасет создан в {dst_root}")