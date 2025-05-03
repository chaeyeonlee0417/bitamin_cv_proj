class WeightedWildFusion:
    def __init__(self, matcher_mega, matcher_aliked, weight1=0.5, weight2=0.5):
        self.matcher_mega = matcher_mega
        self.matcher_aliked = matcher_aliked
        self.weight1 = weight1
        self.weight2 = weight2

    def match(self, query_feats, db_feats):
        sim_mega = self.matcher_mega.match(query_feats, db_feats)
        sim_aliked = self.matcher_aliked.match(query_feats, db_feats)
        return self.weight1 * sim_mega + self.weight2 * sim_aliked

from wildlife_tools.similarity.wildfusion import WildFusion

'''
WildFusion 객체 생성 + calibration 수행
'''
def build_wildfusion(matcher_aliked, matcher_mega, calibration_query, calibration_db):
    fusion = WildFusion(
        calibrated_pipelines=[matcher_aliked, matcher_mega],
        priority_pipeline=matcher_mega
    )
    fusion.fit_calibration(calibration_query, calibration_db)
    return fusion
