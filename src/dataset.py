from wildlife_datasets.datasets import AnimalCLEF2025
from config import PROCESSED_DIR
import pandas as pd
from torchvision.transforms import functional as TF
import os

'''
AnimalCLEF2025 로드하고, query/database/calibration 분리까지 담당
'''
def salamander_orientation_transform(image, metadata):
    # Only apply to SalamanderID2025 dataset
    if metadata.get("dataset") == "SalamanderID2025":
        orientation = metadata.get("orientation", "top")
        # Align 'right' orientation to 'top' by rotating -90 degrees
        if orientation == "right":
            return TF.rotate(image, -90)
        # Align 'left' orientation to 'top' by rotating +90 degrees
        elif orientation == "left":
            return TF.rotate(image, 90)
        # 'top' orientation needs no change
    return image

# ✅ 수정된 전체 로딩 함수
def load_datasets(root, calibration_size=100):
    metadata_path = os.path.join(root, "metadata.csv")
    metadata = pd.read_csv(metadata_path)

    # ✅ 전처리된 .png 파일 경로로 변경
    metadata["path"] = metadata.apply(
        lambda row: os.path.join(PROCESSED_DIR, row["split"], f"{row['image_id']}.png"), axis=1
    )

    # ✅ 수정된 metadata 사용하여 전체 dataset 생성
    dataset = AnimalCLEF2025(root, df=metadata, load_label=True, transform=salamander_orientation_transform)

    # ✅ split 기준으로 쪼갬
    dataset_db = dataset.get_subset(dataset.metadata['split'] == 'database')
    dataset_query = dataset.get_subset(dataset.metadata['split'] == 'query')

    # ✅ calib: DB에서 일부 샘플링
    calib_meta = metadata[metadata['split'] == 'database'].sample(
        n=min(calibration_size, len(metadata[metadata['split'] == 'database'])),
        random_state=42
    )
    calib_meta["path"] = calib_meta.apply(
    lambda row: os.path.join(PROCESSED_DIR, "database", f"{row['image_id']}.png"), axis=1
    )
    dataset_calib = AnimalCLEF2025(root, df=calib_meta, load_label=True, transform=salamander_orientation_transform)

    return dataset, dataset_db, dataset_query, dataset_calib

# ✅ 종별로 나눠서 불러오는 함수 (optional)
def load_datasets_by_species(root, calibration_size=100):
    metadata_path = os.path.join(root, "metadata.csv")
    metadata = pd.read_csv(metadata_path)

    # ✅ .png 경로 반영
    metadata["path"] = metadata.apply(
    lambda row: os.path.join(PROCESSED_DIR, row["split"], f"{row['image_id']}.png"), axis=1
    )


    species_groups = {}
    for dataset_name in metadata['dataset'].unique():
        is_dataset = metadata['dataset'] == dataset_name
        db_df = metadata[is_dataset & (metadata['split'] == 'database')]
        query_df = metadata[is_dataset & (metadata['split'] == 'query')]

        print(f"[INFO] Dataset: {dataset_name} | DB: {len(db_df)}, Query: {len(query_df)}")

        calib_df = db_df.sample(n=min(calibration_size, len(db_df)), random_state=42)
        db_df = db_df.drop(calib_df.index)

        dataset_db = AnimalCLEF2025(root, df=db_df, load_label=True, transform=salamander_orientation_transform)
        dataset_query = AnimalCLEF2025(root, df=query_df, load_label=True, transform=salamander_orientation_transform)
        dataset_calib = AnimalCLEF2025(root, df=calib_df, load_label=True, transform=salamander_orientation_transform)

        species_groups[dataset_name] = {
            'db': dataset_db,
            'query': dataset_query,
            'calib': dataset_calib
        }

    return species_groups
