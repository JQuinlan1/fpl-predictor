import json
import urllib2
import logging


class Data:

    def __init__(self):
        self.data = None

    def download(self, url):
        self.data = self.download_data_from_url(url)
        return self

    @staticmethod
    def download_data_from_url(url):
        return json.loads(urllib2.urlopen(url).read())


class Utils:

    def __init__(self):
        pass

    @staticmethod
    def filter_list_for_attribute_condition_value(li, to_get, k, v):
        for entry in li:
            if entry[k] == v:
                return entry[to_get]
        return None

    @staticmethod
    def filter_list_for_attribute_condition(li, k, v):
        for entry in li:
            if entry[k] == v:
                return entry
        return None

    @staticmethod
    def attribute_condition_in_list(li, k, v):
        return Utils.filter_list_for_attribute_condition(li, k, v) is not None

    @staticmethod
    def assert_type(inst, typ, log):
        if not isinstance(inst, typ):
            msg = "Value %s is not of expected type %s but instead of type %s" % (inst, typ, type(inst))
            log.error(msg)
            raise TypeError(msg)
        log.debug("Value %s confirmed to be of type %s" % (inst, typ))

    @staticmethod
    def is_gameweek_started(gw_data):
        return any(fixture["started"] for fixture in gw_data["fixtures"])

    @staticmethod
    def is_gameweek_finished(gw_data):
        return all(fixture["finished"] for fixture in gw_data["fixtures"])

    @staticmethod
    def player_played_this_week(player, gw_data):
        return player.id in gw_data["elements"].keys()

    @staticmethod
    def print_indented_players(players, log):
        for player in players:
            log.info("    %s" % player)

    @staticmethod
    def get_logger():
        logger = logging.getLogger("FPLPredictor")
        logger.setLevel(logging.INFO)
        # create file handler which logs even debug messages
        # fh = logging.FileHandler('spam.log')
        # fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        # logger.addHandler(fh)
        logger.addHandler(ch)
        return logger