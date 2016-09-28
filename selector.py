from pulp import *
from time import time
from collections import Counter


class SingleFormationSelector:

    def __init__(self, players, g_log, selection):
        self.log = g_log
        self.players = players
        self.selection = selection
        self.capital = 1000.0
        self.log.debug("Sub-problem defined for selection: %s" % self.selection)

    def solve(self):

        prices = {p.id: p.metadata.price for p in self.players}
        positions = {p.id: p.metadata.position for p in self.players}
        label = {p.id: sum(p.predictions.values()) for p in self.players}
        teams = {p.id: p.metadata.team_id for p in self.players}
        ids = [p.id for p in self.players if len(p.predictions.keys()) > 0]  # only include players with predictions

        for attempt in range(10):

            self.log.debug("Starting attempt %s at solving SingleFormationSelector for bench %s" %
                           (attempt, self.selection))

            # setup - declare the problem
            prob1 = LpProblem("MinimumValueSelector", LpMinimize)
            assign_vars1 = LpVariable.dicts("MVS_players", ids, 0, 1, LpBinary)

            # objective - minimize cost
            prob1.objective = lpSum(assign_vars1[ID] * prices[ID] for ID in ids)

            # selection constraint
            prob1.addConstraint(lpSum(assign_vars1[ID] for ID in ids if positions[ID] == 1) == self.selection[1])
            prob1.addConstraint(lpSum(assign_vars1[ID] for ID in ids if positions[ID] == 2) == self.selection[2])
            prob1.addConstraint(lpSum(assign_vars1[ID] for ID in ids if positions[ID] == 3) == self.selection[3])
            prob1.addConstraint(lpSum(assign_vars1[ID] for ID in ids if positions[ID] == 4) == self.selection[4])

            # solve the problem
            self.log.debug("Attempting bench problem solution for attempt %s" % attempt)
            prob1.solve()
            if prob1.status != 1:
                msg = "Linear programming algorithm was unsuccessful for selection %s," % self.selection +\
                      " constraints failed with status %s" % LpStatus[prob1.status]
                self.log.error(msg)
                raise Exception(msg)

            # solution determines our bench
            bench = [self.get_player_from_id(ID) for ID, v in assign_vars1.items() if value(v) == 1.0]
            self.log.debug("Determined bench of %s" % [(p.id, p.metadata.name) for p in bench])
            bench_cost = sum(p.metadata.price for p in bench)
            self.log.debug("Determined bench has total cost of %s" % bench_cost)

            # now, given bench, setup the actual problem
            prob2 = LpProblem("SingleFormationSelector", LpMaximize)
            assign_vars2 = LpVariable.dicts("SFS_players", ids, 0, 1, LpBinary)

            # objective: to maximize the total points
            prob2.objective = lpSum(assign_vars2[ID] * label[ID] for ID in ids)

            # cost constraint
            prob2.addConstraint(lpSum(assign_vars2[ID] * prices[ID] for ID in ids) <= (self.capital - bench_cost))

            # positions constraint
            prob2.addConstraint(lpSum(assign_vars2[ID] for ID in ids if positions[ID] == 1) == (2 - self.selection[1]))
            prob2.addConstraint(lpSum(assign_vars2[ID] for ID in ids if positions[ID] == 2) == (5 - self.selection[2]))
            prob2.addConstraint(lpSum(assign_vars2[ID] for ID in ids if positions[ID] == 3) == (5 - self.selection[3]))
            prob2.addConstraint(lpSum(assign_vars2[ID] for ID in ids if positions[ID] == 4) == (3 - self.selection[4]))

            # teams constraint
            for tid in range(1, 21):
                prob2.addConstraint(lpSum(assign_vars2[ID] for ID in ids if teams[ID] == tid) <= 3)

            # custom constraints
            # prob.addConstraint(assign_vars[242] == 1)  # de gea
            # prob.addConstraint(assign_vars[147] == 1)  # Jakupovic (crap keeper)
            # prob.addConstraint(assign_vars[425] == 0)  # deeney
            # prob.addConstraint(assign_vars[426] == 0)  # ighalo
            # prob.addConstraint(assign_vars[184] == 0)  # vardy
            # prob.addConstraint(assign_vars[134] == 0)  # barkley
            # prob.addConstraint(assign_vars[51] == 1)  # grabban (crap striker)
            # prob.addConstraint(assign_vars[128] == 1)  # stones
            # prob.addConstraint(assign_vars[272] == 1)  # ibrahimovic
            # prob2.addConstraint(assign_vars2["293"] == 0)  # stuani

            # solve the problem
            self.log.debug("Attempting picked problem solution for attempt %s" % attempt)
            prob2.solve()
            if prob2.status != 1:
                msg = "Linear programming algorithm was unsuccessful for selection %s," % self.selection +\
                      " constraints failed with status %s" % LpStatus[prob1.status]
                self.log.error(msg)
                raise Exception(msg)

            # check that subs and players don't collide on teams
            # self.log.debug("Ensuring team condition met between benched and picked players")
            pick = [self.get_player_from_id(ID) for ID, v in assign_vars2.items() if value(v) == 1.0]
            # t15 = bench + pick
            # t15_teams = [p.metadata.team_id for p in t15]
            # if Counter(t15_teams).most_common(1)[0][1] <= 3:
            #     self.log.debug("Team condition met, returning %s, %s" % (pick, bench))
            return pick, bench

            # self.log.info("Team condition not met for attempt %s" % attempt)

        # if reached here, had 10 attempts with no solution
        # raise Exception("Failure to obtain solution for selection %s after 10 attempts")

    def get_player_from_id(self, pid):
        for player in self.players:
            if player.id == pid:
                return player
        return None


class TeamSelector:
    def __init__(self, players, log):
        self.log = log
        self.log.info("Initializing TeamSelector")
        self.players = players
        self.log.debug("Creating selection sub-problems")
        self.sub_problems = [
            SingleFormationSelector(self.players, self.log, {1: 1, 2: 2, 3: 1, 4: 0}),
            SingleFormationSelector(self.players, self.log, {1: 1, 2: 2, 3: 0, 4: 1}),
            SingleFormationSelector(self.players, self.log, {1: 1, 2: 1, 3: 2, 4: 0}),
            SingleFormationSelector(self.players, self.log, {1: 1, 2: 1, 3: 1, 4: 1}),
            SingleFormationSelector(self.players, self.log, {1: 1, 2: 1, 3: 0, 4: 2}),
            SingleFormationSelector(self.players, self.log, {1: 1, 2: 0, 3: 3, 4: 0}),
            SingleFormationSelector(self.players, self.log, {1: 1, 2: 0, 3: 2, 4: 1}),
            SingleFormationSelector(self.players, self.log, {1: 1, 2: 0, 3: 1, 4: 2})
        ]
        self.solutions = []

    def solve(self):
        t0 = time()

        for problem in self.sub_problems:
            self.log.debug("Solving sub-problem %s" % self.sub_problems.index(problem))
            self.solutions.append(problem.solve())

        self.log.debug("Calculating best solution from sub-problem solutions")
        best_solution = max(self.solutions, key=lambda s: sum(sum(p.predictions.values()) for p in s[0] + s[1]))

        self.log.info("Linear programming team selection problem solved in %s seconds." % (time() - t0))
        return best_solution


class CompleteSelector:

    def __init__(self, players, log):
        self.log = log
        self.log.info("Initializing TeamSelector")
        self.players = players
        self.capital = 1000.

    def solve(self):
        t0 = time()

        prices = {p.id: p.metadata.price for p in self.players}
        positions = {p.id: p.metadata.position for p in self.players}
        label = {p.id: sum(p.predictions.values()) for p in self.players}
        teams = {p.id: p.metadata.team_id for p in self.players}
        ids = [p.id for p in self.players if len(p.predictions.keys()) > 0]  # only include players with predictions

        prob = LpProblem("SingleFormationSelector", LpMaximize)
        assign_vars = LpVariable.dicts("SFS_players", ids, 0, 1, LpBinary)

        def selection_function(formation):
            return sum(sorted([assign_vars[ID] * label[ID] for ID in ids if positions[ID] == 1])[-formation[1]:]) + \
                   sum(sorted([assign_vars[ID] * label[ID] for ID in ids if positions[ID] == 2])[-formation[2]:]) + \
                   sum(sorted([assign_vars[ID] * label[ID] for ID in ids if positions[ID] == 3])[-formation[3]:]) + \
                   sum(sorted([assign_vars[ID] * label[ID] for ID in ids if positions[ID] == 4])[-formation[4]:])

        formations = [
            {1: 1, 2: 3, 3: 4, 4: 3},
            {1: 1, 2: 3, 3: 5, 4: 2},
            {1: 1, 2: 4, 3: 3, 4: 3},
            {1: 1, 2: 4, 3: 4, 4: 2},
            {1: 1, 2: 4, 3: 5, 4: 1},
            {1: 1, 2: 5, 3: 2, 4: 3},
            {1: 1, 2: 5, 3: 3, 4: 2},
            {1: 1, 2: 5, 3: 4, 4: 1}
        ]

        # objective: to maximize the total points
        prob.objective = max([selection_function(formation) for formation in formations])

        # cost constraint
        prob.addConstraint(lpSum(assign_vars[ID] * prices[ID] for ID in ids) <= self.capital)

        # positions constraint
        prob.addConstraint(lpSum(assign_vars[ID] for ID in ids if positions[ID] == 1) == 2)
        prob.addConstraint(lpSum(assign_vars[ID] for ID in ids if positions[ID] == 2) == 5)
        prob.addConstraint(lpSum(assign_vars[ID] for ID in ids if positions[ID] == 3) == 5)
        prob.addConstraint(lpSum(assign_vars[ID] for ID in ids if positions[ID] == 4) == 3)

        # teams constraint
        for tid in range(1, 21):
            prob.addConstraint(lpSum(assign_vars[ID] for ID in ids if teams[ID] == tid) <= 3)

        # custom constraints
        prob.addConstraint(assign_vars["239"] == 1)  # aguero
        prob.addConstraint(assign_vars["235"] == 0)  # kdb injury

        # solve the problem
        self.log.debug("Attempting to solve linear programming problem")
        prob.solve()
        if prob.status != 1:
            msg = "Linear programming algorithm was unsuccessful," +\
                  " constraints failed with status %s" % LpStatus[prob.status]
            self.log.error(msg)
            raise Exception(msg)

        self.log.info("Linear programming team selection problem solved in %s seconds." % (time() - t0))

        players = [self.get_player_from_id(ID) for ID, v in assign_vars.items() if value(v) == 1.0]

        # we now have to work out which are benched. This is in itself a linear programming problem
        # but we can solve it ourselves because it's quite limited ;)
        position_split = {k: sorted([player for player in players if player.metadata.position == k], key=lambda x: sum(x.predictions.values())) for k in range(1, 5)}
        return max([self.best_in_formation(formation, position_split) for formation in formations], key=lambda x: x[0])[1:]
        # goalkeepers = sorted([player for player in players if player.metadata.position == 1], key=lambda x: sum(x.predictions.values()))
        # outfield = sorted([player for player in players if player.metadata.position != 1], key=lambda x: sum(x.predictions.values()))
        # return [goalkeepers[-1]] + outfield[-10:], [goalkeepers[0]] + outfield[:3]

    def get_player_from_id(self, pid):
        for player in self.players:
            if player.id == pid:
                return player
        return None

    @staticmethod
    def best_in_formation(formation, psplit):
        play, bench = [], []
        for p in range(1, 5):
            play += psplit[p][-formation[p]:]
            bench += psplit[p][:-formation[p]]
        return sum([sum(x.predictions.values()) for x in play]), play, bench
