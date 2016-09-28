import abc
from tools import Data, Utils
import csv
from collections import defaultdict


class DataSource:

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_gameweek_data(self, gw):
        '''
        :param gw: gameweek for which to return data
        :return: dictionary in particular gameweek format
        '''

    @abc.abstractmethod
    def get_players_data(self):
        '''
        :return: dictionary for all players
        '''

    @abc.abstractmethod
    def get_training_gameweeks(self):
        '''
        :return: list of gameweek numbers to use for training
        '''

    @abc.abstractmethod
    def get_prediction_gameweeks(self, n_predictions):
        '''
        :param n_predictions: number of prediction gameweeks to return
        :return: list of gameweek numbers to predict for
        '''


class NativeFplDataSource(DataSource):

    BOOTSTRAP_STATIC_URL = r"https://fantasy.premierleague.com/drf/bootstrap-static"
    GAMEWEEK_URL = r"https://fantasy.premierleague.com/drf/event/__GAMEWEEK__/live"

    def __init__(self, log):
        DataSource.__init__(self)
        log.info("Beginning gameweek data load")
        self.all_gameweek_data = {
            gw: Data().download(self.GAMEWEEK_URL.replace("__GAMEWEEK__", str(gw))).data for gw in range(1, 39)
        }
        log.info("Finished gameweek data load")

    def get_players_data(self):
        bs_data = Data().download(self.BOOTSTRAP_STATIC_URL)
        return bs_data.data["elements"]

    def get_gameweek_data(self, gw):
        return self.all_gameweek_data[gw]

    def get_training_gameweeks(self):
        return [k for k, v in self.all_gameweek_data.iteritems() if Utils.is_gameweek_finished(v)]

    def get_prediction_gameweeks(self, n_predictions):
        unstarted_weeks = {k: v for k, v in self.all_gameweek_data.iteritems() if not Utils.is_gameweek_started(v)}
        sorted_cut = sorted(unstarted_weeks.iteritems(), key=lambda x: x[0])[:n_predictions]
        return [wk for wk, data in sorted_cut]


class CustomCsvDataSource(DataSource):

    PLAYER_DATA = r"data/FPL 2014-15/Player_Data.csv"
    PLAYER_DETAILS = r"data/FPL 2014-15/Player_Details.csv"

    element_type_map = {"Goalkeeper": 1, "Defender": 2, "Midfielder": 3, "Forward": 4}
    team_name_map = {"Arsenal": 1, "Aston Villa": 2, "Burnley": 3, "Chelsea": 4, "Crystal Palace": 5, "Everton": 6,
                      "Hull": 7, "Leicester": 8, "Liverpool": 9, "Man City": 10, "Man Utd": 11, "Newcastle": 12,
                      "Southampton": 13, "Stoke": 14, "Sunderland": 15, "Swansea": 16, "Spurs": 17, "QPR": 18,
                      "West Brom": 19, "West Ham": 20}
    short_team_map = {"ARS": 1, "AVL": 2, "BUR": 3, "CHE": 4, "CRY": 5, "EVE": 6,
                      "HUL": 7, "LEI": 8, "LIV": 9, "MCI": 10, "MUN": 11, "NEW": 12,
                      "SOU": 13, "STK": 14, "SUN": 15, "SWA": 16, "TOT": 17, "QPR": 18,
                      "WBA": 19, "WHU": 20}

    def __init__(self, current_gw):
        DataSource.__init__(self)
        self.player_data = {}
        self.current_gw = current_gw
        gw_data = defaultdict(list)

        with open(self.PLAYER_DATA) as f:
            for entry in csv.DictReader(f):
                gw_data[int(entry["Week"])].append(entry)

        with open(self.PLAYER_DETAILS) as f:
            for entry in csv.DictReader(f):
                cost = Utils.filter_list_for_attribute_condition_value(gw_data[current_gw], "Value", "ID", str(entry["ID"]))
                if cost is None:
                    cost = 999
                self.player_data[str(entry["ID"])] = {
                    "id": str(entry["ID"]),
                    "team": self.team_name_map[entry["Team"]],
                    "element_type": self.element_type_map[entry["Position"]],
                    "now_cost": float(cost),
                    "first_name": entry["Name"].split(" ")[0].decode('utf-8', 'ignore'),
                    "second_name": " ".join(entry["Name"].split(" ")[1:]).decode('utf-8', 'ignore')
                }

        self.all_gameweek_data = {}
        for gw in gw_data.keys():
            elements = {str(e["ID"]): {"stats": {
                "minutes": int(e["Mins"]),
                "bps": int(e["BPS"]),
                "goals_scored": int(e["Goals"]),
                "assists": int(e["Assists"]),
                "red_cards": int(e["Red_Cards"]),
                "yellow_cards": int(e["Yellow_Cards"]),
                "clean_sheets": int(e["Clean_Sheets"]),
                "saves": int(e["Saves"]),
                "goals_conceded": int(e["Goals_Conceded"]),
                "total_points": int(e["Points"])
            }} for e in gw_data[gw]}
            team_pairs = [(self.player_data[str(e["ID"])]["team"], self.short_team_map[e["Opponent"]])
                                 for e in gw_data[gw] if e["Venue"] == "H"]

            fixtures = []
            for h, a in team_pairs:
                if Utils.attribute_condition_in_list(fixtures, "team_h", h):
                    continue
                h_fixt_opp = max({a1: team_pairs.count((h1, a1)) for h1, a1 in team_pairs if h1 == h}.iteritems(),
                                 key=lambda x: x[1])[0]
                if team_pairs.count((h, h_fixt_opp)) < 10:
                    continue
                fixtures.append({"team_h": h, "team_a": h_fixt_opp})
            # fixtures = dict(set([(self.player_data[e["ID"]]["team"], self.short_team_map[e["Opponent"]])
            #                      for e in gw_data[str(gw)] if e["Venue"] == "H"]))
            # todo investigate not being able to extract fixtures # due to transfers
            self.all_gameweek_data[gw] = {"elements": elements, "fixtures": fixtures}

    def get_players_data(self):
        return self.player_data.values()

    def get_gameweek_data(self, gw):
        return self.all_gameweek_data[gw]

    def get_training_gameweeks(self):
        return sorted(self.all_gameweek_data.keys())[:self.current_gw-1]

    def get_prediction_gameweeks(self, n_predictions):
        return sorted(self.all_gameweek_data.keys())[self.current_gw-1:self.current_gw+n_predictions-1]

    def get_scores_for_current_gw(self):
        return {p: v["stats"]["total_points"] for p, v in self.all_gameweek_data[self.current_gw]["elements"].iteritems()}


# todo: make this read from our csv format :)
# todo: pick better api for data-source
class NormalCsvDataSource(DataSource):

    '''
    how do we want to expose the data?
    we want to take the data and turn it into fv/l pairs and a current fv
    '''

    DATAFILE = r"data/FPL_player_data_2016-17.csv"

    def __init__(self):
        DataSource.__init__(self)
        self.data = list(csv.DictReader(open(self.DATAFILE)))

    def get_prediction_gameweeks(self, n_predictions):
        pass

    def get_training_gameweeks(self):
        pass

    def get_players_data(self):
        pass

    def get_gameweek_data(self, gw):
        pass
