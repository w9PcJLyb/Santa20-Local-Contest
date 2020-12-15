import numpy as np
from enum import IntEnum
from ndarray import NDArrayField
from django.db import models
from django.db.models import Q

DECAY_RATE = 0.97
SAMPLE_RESOLUTION = 100


class Agent(models.Model):
    name = models.CharField(max_length=50, blank=True)
    source = models.URLField(null=True, blank=True)
    rating = models.FloatField(blank=True, null=False, default=600)
    file = models.FileField(blank=True, upload_to="uploads/")
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)

    def __str__(self):
        return f"Agent '{self.name}' {round(self.rating, 1)}"

    def games_qs(self):
        return Game.objects.filter(
            Q(left_agent=self) | Q(right_agent=self), status=GameStatus.FINISHED
        )

    def num_games(self):
        return self.games_qs().count()


class GameStatus(IntEnum):
    STARTED = 0
    FINISHED = 10
    DELETED = 20


class GameResult(IntEnum):
    LEFT_WON = 0
    RIGHT_WON = 1
    DRAW = 10
    UNKNOWN = 20


class Game(models.Model):
    started = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    finished = models.DateTimeField(null=True, blank=True)
    status = models.IntegerField(
        choices=[(x.value, x.name) for x in GameStatus], default=GameStatus.STARTED
    )
    configuration = models.JSONField(null=True, blank=True)

    left_agent = models.ForeignKey(
        Agent, blank=True, related_name="games_l", on_delete=models.CASCADE
    )
    right_agent = models.ForeignKey(
        Agent, blank=True, related_name="games_r", on_delete=models.CASCADE
    )

    left_current_rating = models.FloatField()
    right_current_rating = models.FloatField()

    left_new_rating = models.FloatField(null=True)
    right_new_rating = models.FloatField(null=True)

    initial_thresholds = NDArrayField(blank=True, null=True)
    left_actions = NDArrayField(blank=True, null=True)
    right_actions = NDArrayField(blank=True, null=True)
    left_rewards = NDArrayField(blank=True, null=True)
    right_rewards = NDArrayField(blank=True, null=True)

    result = models.IntegerField(
        choices=[(x.value, x.name) for x in GameResult],
        null=True,
        default=GameResult.UNKNOWN,
    )

    def __str__(self):
        return (
            f"Game '{self.left_agent.name}' {round(self.left_current_rating, 1)} vs "
            f"'{self.right_agent.name}' {round(self.right_current_rating, 1)} - {GameResult(self.result).name}"
        )

    def execution_time(self):
        if self.started is None or self.finished is None:
            return "-"

        return self.finished - self.started

    def total_rewards(self):
        return self.left_rewards[-1], self.right_rewards[-1]

    def expected_rewards_estimation(self):
        th = np.array(self.initial_thresholds, dtype="float32")
        l_expected = np.zeros(len(self.left_actions), dtype="float32")
        r_expected = np.zeros(len(self.right_actions), dtype="float32")
        for i, (la, ra) in enumerate(zip(self.left_actions, self.right_actions)):
            l_expected[i] = th[la]
            r_expected[i] = th[ra]
            th[la] *= DECAY_RATE
            th[ra] *= DECAY_RATE
        return l_expected / SAMPLE_RESOLUTION, r_expected / SAMPLE_RESOLUTION

    def total_expected_rewards(self):
        le, re = self.expected_rewards_estimation()
        return le.sum(), re.sum()

    def thresholds_at_the_end(self):
        th = np.array(self.initial_thresholds, dtype="float32")
        for i in self.left_actions:
            th[i] *= DECAY_RATE
        for i in self.right_actions:
            th[i] *= DECAY_RATE
        return th
