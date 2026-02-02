import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import numpy as np
import pandas as pd
import torch
import timm

from config import (
    ROOT,
    MEGAD_NAME,
    DEVICE,
    THRESHOLD,
    MARGIN_12_THRESHOLD,   # ✅ config에서 margin threshold import
)

from src.dataset import load_datasets
from src.utils import set_seed
from src.transforms import transform_tta_mega, transforms_aliked
from src.factory import build_megadescriptor, build_aliked, build_eva02


def main():
    set_seed(42)

    # ---------------------------------------------------------
    # 1. 데이터셋 로드
    # ---------------------------------------------------------
    print("[Main] Loading datasets...")
    dataset, dataset_db, dataset_query, dataset_calib = load_datasets(
        ROOT, calibration_size=1000
    )

    db_mask = np.ones(len(dataset_db), dtype=bool)
    calib_mask = np.ones(len(dataset_calib), dtype=bool)

    # ---------------------------------------------------------
    # 2. 파이프라인 구축
    # ---------------------------------------------------------
    print("[Main] Building pipelines...")

    print(f" - MegaDescriptor ({MEGAD_NAME})")
    model_mega = timm.create_model(
        MEGAD_NAME, num_classes=0, pretrained=True
    ).to(DEVICE)
    pipeline_mega = build_megadescriptor(
        model=model_mega, transform=transform_tta_mega, device=DEVICE
    )

    print(" - ALIKED")
    pipeline_aliked = build_aliked(
        transform=transforms_aliked, device=DEVICE
    )

    print(" - EVA02")
    pipeline_eva = build_eva02(device=DEVICE)

    # ---------------------------------------------------------
    # 3. Calibration
    # ---------------------------------------------------------
    print("[Main] Starting calibration...")

    calib = dataset_calib.get_subset(calib_mask)

    calib.transform = transform_tta_mega
    pipeline_mega.fit_calibration(calib, calib)

    calib.transform = transforms_aliked
    pipeline_aliked.fit_calibration(calib, calib)

    calib.transform = pipeline_eva.transform
    pipeline_eva.fit_calibration(calib, calib)

    # ---------------------------------------------------------
    # 4. Inference & Ensemble
    # ---------------------------------------------------------
    predictions_all = []
    image_ids_all = []

    db_mega = dataset_db.get_subset(db_mask)
    db_mega.transform = transform_tta_mega

    db_aliked = dataset_db.get_subset(db_mask)
    db_aliked.transform = transforms_aliked

    db_eva = dataset_db.get_subset(db_mask)
    db_eva.transform = pipeline_eva.transform

    printed_stats = False  # ✅ final_scores 분포 로그 1회만 출력

    for dataset_name in dataset_query.metadata["dataset"].unique():
        query_mask = dataset_query.metadata["dataset"] == dataset_name
        query_subset = dataset_query.get_subset(query_mask)

        print(f"[Main] Processing {dataset_name} ({len(query_subset)})")

        full_mask = np.ones(len(query_subset), dtype=bool)

        # -----------------------------------------------------
        # A. MegaDescriptor
        # -----------------------------------------------------
        query_mega = query_subset.get_subset(full_mask)
        query_mega.transform = transform_tta_mega
        scores_mega = pipeline_mega(query_mega, db_mega)

        # -----------------------------------------------------
        # B. ALIKED Reranking
        # -----------------------------------------------------
        B = 25
        _, topk_indices = torch.topk(
            torch.from_numpy(scores_mega),
            k=min(B, scores_mega.shape[1]),
            dim=1,
        )

        pairs = [(r, c) for r, cols in enumerate(topk_indices.numpy()) for c in cols]

        query_aliked = query_subset.get_subset(full_mask)
        query_aliked.transform = transforms_aliked

        scores_aliked_sparse = pipeline_aliked(
            query_aliked, db_aliked, pairs=pairs
        )

        scores_aliked_full = np.full_like(scores_mega, -np.inf)
        if scores_aliked_sparse.ndim == 1:
            for (r, c), s in zip(pairs, scores_aliked_sparse):
                scores_aliked_full[r, c] = s
        else:
            scores_aliked_full = scores_aliked_sparse

        scores_fusion = scores_mega.copy()
        valid = scores_aliked_full > -999
        scores_fusion[valid] = (
            scores_mega[valid] + scores_aliked_full[valid]
        ) / 2

        # -----------------------------------------------------
        # C. EVA02
        # -----------------------------------------------------
        query_eva = query_subset.get_subset(full_mask)
        query_eva.transform = pipeline_eva.transform
        scores_eva = pipeline_eva(query_eva, db_eva)

        # -----------------------------------------------------
        # D. Final Ensemble
        # -----------------------------------------------------
        final_scores = 0.5 * scores_fusion + 0.5 * scores_eva

        # ✅ final_scores 분포 로그 (1회)
        if not printed_stats:
            fs = final_scores.astype(np.float32)
            print(f"[Debug] final_scores min/max: {fs.min():.6f} / {fs.max():.6f}")
            print(
                "[Debug] percentiles:",
                np.percentile(fs, [0, 1, 5, 50, 95, 99, 100]),
            )
            print(
                f"[Debug] in [0,1] range?: {fs.min() >= 0 and fs.max() <= 1}"
            )
            print(
                f"[Debug] THRESHOLD={THRESHOLD}, "
                f"MARGIN_12_THRESHOLD={MARGIN_12_THRESHOLD}"
            )
            printed_stats = True

        # -----------------------------------------------------
        # E. Prediction + Top-1 / Top-2 Margin
        # -----------------------------------------------------
        top2_idx = np.argpartition(final_scores, -2, axis=1)[:, -2:]
        top2_scores = np.take_along_axis(final_scores, top2_idx, axis=1)

        order = np.argsort(top2_scores, axis=1)
        top2_scores = np.take_along_axis(top2_scores, order, axis=1)
        top2_idx = np.take_along_axis(top2_idx, order, axis=1)

        p_top2 = top2_scores[:, 0]
        p_top1 = top2_scores[:, 1]
        top_idx = top2_idx[:, 1]

        margin_12 = p_top1 - p_top2

        pred_labels = dataset_db.labels_string[top_idx].copy()
        is_new = (p_top1 < THRESHOLD) | (margin_12 < MARGIN_12_THRESHOLD)
        pred_labels[is_new] = "new_individual"

        predictions_all.extend(pred_labels)
        image_ids_all.extend(query_subset.metadata["image_id"])

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    df = pd.DataFrame(
        {"image_id": image_ids_all, "identity": predictions_all}
    )
    df.to_csv("sample_submission.csv", index=False)
    print("✅ sample_submission.csv saved!")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
