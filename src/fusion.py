'''
MegaDescriptor + ALIKED matcher를 가중치 기반으로 융합하는 Fusion 객체 생성
'''
def build_weighted_fusion(
    matcher_aliked,
    matcher_mega,
    calibration_query=None,
    calibration_db=None,
    w1=0.6,
    w2=0.4
):
    class WeightedWildFusion:
        def __init__(self, matcher_mega, matcher_aliked, weight1=0.5, weight2=0.5):
            self.matcher_mega = matcher_mega
            self.matcher_aliked = matcher_aliked
            self.weight1 = weight1
            self.weight2 = weight2
            
            if calibration_query is not None and calibration_db is not None:
                self.matcher_mega.fit_calibration(calibration_query, calibration_db)
                self.matcher_aliked.fit_calibration(calibration_query, calibration_db)

        def __call__(self, query, db, B=25):
            sim_mega = self.matcher_mega(query, db)
            sim_aliked = self.matcher_aliked(query, db)
            return self.weight1 * sim_mega + self.weight2 * sim_aliked

    return WeightedWildFusion(matcher_mega, matcher_aliked, w1, w2)

        
