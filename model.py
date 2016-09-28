from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import NotFittedError
from sklearn.metrics import r2_score
from itertools import chain
import numpy as np
from tools import Utils
import abc
from sklearn.svm import SVR


class Model:

    POSITIONS = [1, 2, 3, 4]

    @abc.abstractmethod
    def _new_model(self):
        '''
        :return: model instantiation
        '''

    def __init__(self, log):
        self.regressions = {k: self._new_model() for k in self.POSITIONS}
        self.log = log

    def train(self, players):

        self.log.info("Beginning training data assembly")

        training_data = {k: [] for k in self.POSITIONS}

        for player in players:

            for player_train_week in player.training_data.keys():
                self.log.debug("Adding FV/label pair for GW%s for player %s" % (player_train_week, player))
                training_data[player.metadata.position].append(player.training_data[player_train_week])

            # for gw in player.features.keys():
            #
            #
            #     # check for label match
            #     lgw = gw+1
            #     if lgw not in player.labels.keys():  # we have no label match, so can't use this feature vector to train
            #         self.log.debug("Player [%s] %s has no label for GW%s, " % (player.id, player.metadata.name, gw) +
            #                        "despite having a feature vector for GW%s, so skipping" % lgw)
            #         continue
            #
            #     # explicitly define candidate FV and label
            #     feature_vector_to_add = player.features[gw].feature_vector + player.team_features[gw]
            #     label_to_add = player.labels[lgw]
            #
            #     # add FV and label to our training set
            #     self.log.debug("Adding FV for GW%s and label for GW%s for " % (gw, lgw) +
            #                    "player [%s] %s" % (player.id, player.metadata.name))
            #     training_data[player.metadata.position][0].append(feature_vector_to_add)
            #     training_data[player.metadata.position][1].append(label_to_add)

        self.log.info("Finished assembling training data, found %s label matches" %
                      sum([len(v) for v in training_data.values()]))

        for pos in self.POSITIONS:

            # get fit input in correct format
            pos_features = np.array([train_pair.features for train_pair in training_data[pos]])
            pos_labels = np.array([train_pair.label for train_pair in training_data[pos]])

            # check fit prerequisites
            if len(pos_features) != len(pos_labels):
                self.log.error("Length of FV set (%s) for position %s different to length of label set (%s)" %
                               (len(pos_features), pos, len(pos_labels)))
                raise Exception()
            elif len(pos_features) == 0:
                self.log.warn("No FVs defined for position %s, not training model" % pos)
                continue

            # cross validation
            self.cross_validate(pos_features, pos_labels, 3)

            # perform fit
            self.log.info("Starting to train for position %s" % pos)
            self.regressions[pos].fit(pos_features, pos_labels)
            self.log.info("Finished train for position %s" % pos)

            predictions = self.regressions[pos].predict(pos_features)
            self.log.info("R^2 score for position %s: %s" % (pos, r2_score(pos_labels, predictions)))

    def predict(self, players):

        try:
            for player in players:

                # order FVs by game-week in reverse
                # single_player_fvs = sorted(player.features.iteritems(), key=lambda x: x[0], reverse=True)
                # single_player_team_fvs = sorted(player.team_features.iteritems(), key=lambda x: x[0], reverse=True)  # todo

                # don't predict if no FVs for player
                # if len(single_player_fvs) == 0:
                #     self.log.info("Not predicting for player [%s] %s because no feature vectors defined" %
                #                   (player.id, player.metadata.name))
                #     continue
                prediction_fvs = player.prediction_fvs
                if len(prediction_fvs.keys()) == 0:
                    self.log.info("Not predicting for player %s because no feature vectors defined" % player)
                    continue

                #
                # # pre-format the prediction FV
                # fgw, raw_prediction_fv = single_player_fvs[-1]
                # fgw2, raw_pred_team_fv = single_player_team_fvs[-1]
                # if fgw != fgw2:
                #     self.log.warn("Inconsistent player (%s) and team (%s) max GW in fv for player %s" % (fgw, fgw2, player))
                # prediction_fv = np.array(raw_prediction_fv.feature_vector + raw_pred_team_fv)

                # perform prediction
                for gw in prediction_fvs.keys():
                    self.log.debug("Predicting for player [%s] %s based on feature vector %s" %
                                   (player.id, player.metadata.name, prediction_fvs[gw]))
                    player.predictions[gw] = self.regressions[player.metadata.position].predict([prediction_fvs[gw]])[0]
                    self.log.debug("Predicted score of %s for player [%s] %s" %
                                   (player.predictions[gw], player.id, player.metadata.name))

        except NotFittedError, e:
            self.log.warn("Model is not fitted, so not making predictions")
            self.log.warn(e.message)
            return

        # print some stats on the model(s)
        # for pos in self.POSITIONS:
        #     self.log.info("%s Intercept: %s" % (pos, self.regressions[pos].intercept_))
        #     self.log.info("%s Coefs: %s" % (pos, self.regressions[pos].coef_))

    def cross_validate(self, features, labels, folds):
        self.log.info("Beginning %s-fold cross validation on training set of size %s" % (folds, len(features)))
        cutoffs = [(len(features)/folds) * i for i in range(folds+1)]
        split_fts = [features[cutoffs[i]:cutoffs[i+1]] for i in range(len(cutoffs)-1)]
        split_lbs = [labels[cutoffs[i]:cutoffs[i+1]] for i in range(len(cutoffs)-1)]
        for i in range(len(split_fts)):
            current_split_ft = split_fts[i]
            current_split_lb = split_lbs[i]
            other_split_fts = np.array(list(chain(*[x for j, x in enumerate(split_fts) if j != i])))
            other_split_lbs = np.array(list(chain(*[x for j, x in enumerate(split_lbs) if j != i])))
            reg = self._new_model()
            reg.fit(other_split_fts, other_split_lbs)
            current_r2 = r2_score(current_split_lb, reg.predict(current_split_ft))
            self.log.info("R^2 score for cross-section %s: %s" % (i+1, current_r2))


class LinearRegressionModel(Model):

    def _new_model(self):
        return LinearRegression(normalize=True)


class LogisticRegressionModel(Model):

    def _new_model(self):
        return LogisticRegression()


class RFRegressionModel(Model):

    def _new_model(self):
        return RandomForestRegressor(min_samples_split=8, min_samples_leaf=2, max_features=6, max_depth=6, n_estimators=1000)