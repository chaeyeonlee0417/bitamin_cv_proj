import os

# Dataset root path
'''
    define your root directory
'''
ROOT = './dataset'

 

# Model settings
MEGAD_NAME = 'hf-hub:BVRA/MegaDescriptor-L-384'
EVA_NAME = 'EVA02-L-14-336'
EVA_WEIGHT_NAME = 'merged2b_s6b_b61k'
DEVICE = 'cuda'

# Threshold
THRESHOLD = 0.35

# ✅ Top-1 vs Top-2 margin threshold (top1 - top2)
# margin이 작으면 모델이 확신하지 못하는 상태로 보고 new_individual로 보내는 보강 규칙
MARGIN_12_THRESHOLD = 0.03
