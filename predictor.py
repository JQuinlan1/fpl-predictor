# coding=utf-8
from player import Player, PlayerTrainingPair
from model import LinearRegressionModel, LogisticRegressionModel, RFRegressionModel
from selector import TeamSelector, CompleteSelector
from tools import Utils
from datasource import NativeFplDataSource, CustomCsvDataSource
import numpy as np
import csv

NUMBER_WEEKS_PREDICT = 1
DEFAULT_LOG_LEVEL = 3


class Runner:

    def __init__(self, log, data_source, model, output_prospects=False):

        # initialize vars
        self.log = log
        self.data_source = data_source
        self.model = model
        self.output_prospects = output_prospects

        # setup logging
        # self.log.set_log_level(DEFAULT_LOG_LEVEL)
        self.log.info("Initializing runner")

        # initialize player list
        self.players = [Player(p, self.log) for p in self.data_source.get_players_data()]
        self.log.info("Created player set of length: %s" % len(self.players))

    def run(self):
        self.update()   # prepare FVs and labels
        self.analyse()  # run model
        return self.select()   # perform linear programming algorithm

    def update(self):

        for gw in self.data_source.get_training_gameweeks():
            self.log.info("Commencing update stage for training GW%s" % gw)
            gw_data = self.data_source.get_gameweek_data(gw)
            self._update_team_fts(gw, gw_data)
            self._update_player_fts(gw, gw_data)

        for gw in self.data_source.get_prediction_gameweeks(NUMBER_WEEKS_PREDICT):
            self.log.info("Commencing update stage for prediction GW%s" % gw)
            self._update_team_fts(gw, self.data_source.get_gameweek_data(gw))

        for gw in [w for w in self.data_source.get_training_gameweeks() if w+1 not in self.data_source.get_prediction_gameweeks(NUMBER_WEEKS_PREDICT)]:
            self.log.info("Commencing finalise stage for training GW%s" % gw)
            self._finalise_training_week(gw)

        for gw in self.data_source.get_prediction_gameweeks(NUMBER_WEEKS_PREDICT):
            self.log.info("Commencing finalise stage for prediction GW%s" % gw)
            self._finalise_prediction_week(gw)

    def _update_team_fts(self, gw, gw_data):
        team_features = {
                tid: [
                    sum(player.labels[gw-1] for player in self.players if player.metadata.team_id == tid and gw in player.labels.keys()),
                    sum(float(sum(player.labels.values()))/len(player.labels.values()) for player in self.players if player.metadata.team_id == tid and len(player.labels.values()) > 0)
                ] for tid in range(1, 21)
            }
        for fixture in gw_data["fixtures"]:
            team_features[fixture["team_h"]] += [
                team_features[fixture["team_a"]][0],
                team_features[fixture["team_a"]][1],
                1.
            ]
            team_features[fixture["team_a"]] += [
                team_features[fixture["team_h"]][0],
                team_features[fixture["team_h"]][1],
                0.
            ]
        for player in self.players:
            player.team_features[gw] = team_features[player.metadata.team_id]

    def _update_player_fts(self, gw, gw_data):

        for player in self.players:

                if not Utils.player_played_this_week(player, gw_data):
                    self.log.debug("Missing player [%s] %s for GW%s" % (player.id, player.metadata.name, gw))
                    continue

                player.update(gw, gw_data["elements"][player.id])
                self.log.debug("Updated player [%s] %s for GW%s" % (player.id, player.metadata.name, gw))

    def _finalise_training_week(self, gw):

        for player in self.players:

            if gw not in player.features.keys():
                self.log.info("No feature data for GW%s for player %s, skipping" % (gw, player))
                continue

            if gw+1 not in player.team_features.keys():
                self.log.info("No team feature data for GW%s for player %s, skipping" % (gw+1, player))
                continue

            player_features_for_week = player.features[gw].feature_vector
            team_features_for_week = player.team_features[gw+1]
            features_for_week = np.array(player_features_for_week + team_features_for_week)
            label_for_week1 = player.labels[gw+1] if gw+1 in player.labels.keys() else None
            label_for_week2 = player.labels[gw+1] if gw+1 in player.labels.keys() else None
            label_for_week3 = player.labels[gw+1] if gw+1 in player.labels.keys() else None
            labels_for_week = (label_for_week1, label_for_week2, label_for_week3)
            if None in labels_for_week:
                player.log.debug("No 3-match label available for GW% for player %s" % (gw, player))
            else:
                player.training_data[gw+1] = PlayerTrainingPair(features_for_week, sum(labels_for_week), player.log)

    def _finalise_prediction_week(self, gw):

        last_gw = max(self.data_source.get_training_gameweeks())

        for player in self.players:

            if last_gw not in player.features.keys():
                self.log.info("No current feature vector for player %s, skipping" % player)
                continue

            player_features_for_week = player.features[last_gw].feature_vector
            team_features_for_week = player.team_features[gw]
            features_for_week = np.array(player_features_for_week + team_features_for_week)
            player.prediction_fvs[gw] = features_for_week

    def analyse(self):

        self.log.info("Beginning analyse stage")

        self.model.train(self.players)
        self.model.predict(self.players)

    def select(self):

        self.log.info("Beginning select stage")

        self.output_predictions()

        # selector = TeamSelector(self.players, self.log)
        selector = CompleteSelector(self.players, self.log)

        solution = list(selector.solve())
        solution_pick = sorted(solution[0], key=lambda x: x.metadata.position)
        solution_bench = sorted(solution[1], key=lambda x: x.metadata.position)

        self.log.info("Best predicted selection")
        self.log.info("Picked:")
        Utils.print_indented_players(solution_pick, self.log)
        self.log.info("Bench:")
        Utils.print_indented_players(solution_bench, self.log)

        predict_gws = self.data_source.get_prediction_gameweeks(NUMBER_WEEKS_PREDICT)
        for gw in predict_gws:
            predicted_gw_score = sum(p.predictions[gw] for p in solution_pick)
            self.log.info("Predicted GW%s score for picked players: %s" % (gw, predicted_gw_score))
        predicted_avg_score = float(sum(sum(p.predictions.values()) for p in solution_pick))/len(predict_gws)
        self.log.info("Average GW score over next %s for picked players: %s" % (len(predict_gws), predicted_avg_score))
        self.log.info("Total cost of team: %s" % sum(p.metadata.price for p in solution_pick + solution_bench))

        if self.output_prospects:
            prospects = {pos: [(p, sum(p.predictions[gw] for gw in predict_gws if gw in p.predictions.keys()), p.metadata.price)
                               for p in self.players if p.metadata.position == pos]
                         for pos in set(p.metadata.position for p in self.players)}
            for pos, player_data in prospects.iteritems():
                sorted_player_data = sorted(player_data, key=lambda x: float(x[1])/x[2], reverse=True)
                self.log.info("Best prospects for position %s" % pos)
                for entry in sorted_player_data[:10]:
                    self.log.info("%50s: %15s,%15s,%15s" % (entry[0], entry[1], entry[2], entry[1]/entry[2]))
            # for entry in sorted([(p, sum(p.predictions[gw] for gw in predict_gws if gw in p.predictions.keys()), p.metadata.price) for p in self.players], key=lambda x: float(x[1])/x[2], reverse=True)[:50]:
            #     self.log.info("%50s: %15s,%15s,%15s" % (entry[0], entry[1], entry[2], entry[1]/entry[2]))

        return solution_pick, solution_bench

    def output_predictions(self):
        with open("player_predictions.csv", "wb") as f:
            w = csv.DictWriter(f, ['id', 'name', 'prediction'])
            for player in self.players:
                pd = {
                    'id': player.id,
                    'name': player.metadata.name.encode('utf-8'),
                    'prediction': sum(player.predictions.values())
                }
                w.writerow(pd)

if __name__ == "__main__":
    # out_log = Log(open("log", "a"))
    out_log = Utils.get_logger()
    # Runner(out_log, NativeFplDataSource(out_log), LinearRegressionModel(out_log), output_prospects=True).run()
    # Runner(out_log, NativeFplDataSource(out_log), LogisticRegressionModel(out_log), output_prospects=True).run()
    Runner(out_log, NativeFplDataSource(out_log), RFRegressionModel(out_log), output_prospects=True).run()
    # tot = 0
    # for wk in range(4, 39):
    # for wk in range(34, 35):
    #     source = CustomCsvDataSource(wk)
    #     pick, bench = Runner(out_log, source, LinearRegressionModel(out_log)).run()
    #     cpn = max(pick, key=lambda p: p.predictions[wk])
    #     scs = source.get_scores_for_current_gw()
    #     print "%4s: %s / %s   (%s: %s)" % (wk, sum(scs[p.id] for p in pick) + scs[cpn.id], sum(p.predictions[wk] for p in pick) + cpn.predictions[wk], cpn, cpn.predictions[wk])
    #     for p in pick:
    #         print "    %4s / %4s  %s" % (scs[p.id], p.predictions[wk], p)
    #     tot += sum(scs[p.id] for p in pick) + scs[cpn.id]
    # print "Overall score: %s" % tot
