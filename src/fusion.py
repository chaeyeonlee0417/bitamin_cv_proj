from wildlife_tools.similarity.wildfusion import WildFusion

'''
WildFusion 객체 생성 + calibration 수행
'''
def build_wildfusion(
    matcher_aliked, 
    matcher_mega, 
    calibration_query, 
    calibration_db,
    fusion_type="weighted", 
    w1=0.8, 
    w2=0.2
):
    if fusion_type == "weighted":
        fusion = WildFusion(
            calibrated_pipelines=[matcher_aliked, matcher_mega],
            priority_pipeline=matcher_mega,
            w1=w1,
            w2=w2
        )
        fusion.fit_calibration(calibration_query, calibration_db)
        return fusion
    else:
        raise NotImplementedError(f"Fusion type '{fusion_type}' is not supported.")
