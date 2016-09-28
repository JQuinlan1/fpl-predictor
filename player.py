import numpy as np
from tools import Utils


class Player:

    def __init__(self, bootstrap_data, log):
        self.log = log
        self.id = unicode(bootstrap_data["id"])
        self.metadata = PlayerMetadata(bootstrap_data)
        self.team = None
        self.predictions = {}
        self.features = {}
        self.team_features = {}
        self.labels = {}
        self.training_data = {}
        self.prediction_fvs = {}

    def __str__(self):
        return "[%3s] %30s %s" % (self.id, self.metadata.name, sum(self.predictions.values()))

    def update(self, gw_id, gw_data):
        self.features[gw_id] = PlayerFeatureVector(self.metadata.position, gw_id, gw_data).load(self.features)
        self.labels[gw_id] = float(gw_data["stats"]["total_points"])


class PlayerMetadata:

    def __init__(self, bootstrap_data):
        self.team_id = bootstrap_data["team"]
        self.position = bootstrap_data["element_type"]
        self.price = bootstrap_data["now_cost"]
        self.name = bootstrap_data["first_name"] + " " + bootstrap_data["second_name"]


class PlayerFeatureVector:

    def __init__(self, p_type, gw_id, gw_data):

        self.gw_id = gw_id
        self.type = p_type

        if gw_data is None:   # do default impl
            self.minutes = 0
            self.bps = 0
            self.goals = 0
            self.assists = 0
            self.reds = 0
            self.yellows = 0
            self.clean_sheets = 0
            self.saves = 0
            self.conceded = 0
        else:
            stats = gw_data["stats"]
            self.minutes = stats["minutes"]
            self.bps = stats["bps"]
            self.goals = stats["goals_scored"]
            self.assists = stats["assists"]
            self.reds = stats["red_cards"]
            self.yellows = stats["yellow_cards"]
            self.clean_sheets = stats["clean_sheets"]
            self.saves = stats["saves"]
            self.conceded = stats["goals_conceded"]

        self.feature_vector = None

    def load(self, context):

        # minutes, bps, assists, goals
        goals_3 = self._get_time_sum_feature(context, lambda x: x.goals, 6)
        goals_avg = self._get_avg_feature(context, lambda x: x.goals)

        assists_3 = self._get_time_sum_feature(context, lambda x: x.assists, 6)
        assists_avg = self._get_avg_feature(context, lambda x: x.assists)

        minutes_3 = self._get_time_sum_feature(context, lambda x: x.minutes, 6)
        minutes_avg = self._get_avg_feature(context, lambda x: x.minutes)

        bps_3 = self._get_time_sum_feature(context, lambda x: x.bps, 6)
        bps_avg = self._get_avg_feature(context, lambda x: x.bps)

        yrs_3 = self._get_time_sum_feature(context, lambda x: 3*x.reds + 1*x.yellows, 6)
        yrs_avg = self._get_avg_feature(context, lambda x: 3*x.reds + 1*x.yellows)

        cs_3 = self._get_time_sum_feature(context, lambda x: x.clean_sheets, 6)
        cs_avg = self._get_avg_feature(context, lambda x: x.clean_sheets)

        saves_3 = self._get_time_sum_feature(context, lambda x: x.saves, 6)
        saves_avg = self._get_avg_feature(context, lambda x: x.saves)

        conceded_3 = self._get_time_sum_feature(context, lambda x: x.conceded, 6)
        conceded_avg = self._get_avg_feature(context, lambda x: x.conceded)

        self.feature_vector = [
            minutes_3,
            minutes_avg,
            bps_3,
            bps_avg,
            cs_3,
            cs_avg,
            saves_3,
            saves_avg,
            conceded_3,
            conceded_avg
        ] if self.type == 1 else [
            minutes_3,
            minutes_avg,
            bps_3,
            bps_avg,
            cs_3,
            cs_avg,
            conceded_3,
            conceded_avg,
            assists_avg,
            yrs_3,
            yrs_avg
        ] if self.type == 2 else [
            goals_3,
            goals_avg,
            assists_3,
            assists_avg,
            bps_3,
            bps_avg,
            yrs_3,
            yrs_avg
        ] if self.type == 3 else [
            goals_3,
            goals_avg,
            assists_3,
            assists_avg,
            bps_3,
            bps_avg
        ] if self.type == 4 else None

        if self.feature_vector is None:
            raise ValueError("Unrecognised player type [%s], could not set feature vector" % self.type)

        return self

    def _get_time_sum_feature(self, previous_data, extractor_fn, tau):
        previous_in_range = sorted(previous_data.iteritems(), key=lambda x: x[0], reverse=True)[:tau]
        if len(previous_in_range) < tau:
            previous_in_range.append((-1, PlayerFeatureVector(self.type, -1, None)))
        tot = 0
        counted = 0
        for t, dat in [(self.gw_id, self)] + previous_in_range:
            tot += extractor_fn(dat)
            counted += 1
        counted = 1 if counted == 0 else counted
        multiplier = float(tau) / counted
        return multiplier * tot

    def _get_avg_feature(self, previous_data, extractor_fn):
        tot = 0
        counted = 0
        for t, dat in [(self.gw_id, self)] + sorted(previous_data.iteritems()):
            tot += extractor_fn(dat)
            counted += 1
        counted = 1 if counted == 0 else counted
        return float(tot) / counted


class PlayerTrainingPair:

    def __init__(self, features, label, log):

        # check FV and label types
        self.log = log
        Utils.assert_type(features, np.ndarray, self.log)
        Utils.assert_type(label, float, self.log)

        self.features = features
        self.label = label
