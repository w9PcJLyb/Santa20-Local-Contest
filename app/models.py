import numpy as np
from enum import IntEnum
from ndarray import NDArrayField
from django.db import models
from django.db.models import Q
from django.utils.functional import cached_property

DECAY_RATE = 0.97
SAMPLE_RESOLUTION = 100


class Agent(models.Model):
    name = models.CharField(max_length=50, blank=True)
    source = models.URLField(null=True, blank=True)
    rating = models.FloatField(blank=True, null=False, default=600)
    file = models.FileField(blank=True, upload_to="uploads/")
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    enabled = models.BooleanField(blank=True, null=False, default=True)
    description = models.CharField(max_length=511, null=True, blank=True)

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
            f"Game '{self.left_name}' {round(self.left_current_rating, 1)} vs "
            f"'{self.right_name}' {round(self.right_current_rating, 1)} - {GameResult(self.result).name}"
        )

    @property
    def left_name(self):
        return self.left_agent.name

    @property
    def right_name(self):
        return self.right_agent.name

    def execution_time(self):
        if self.started is None or self.finished is None:
            return "-"

        return self.finished - self.started

    def total_rewards(self):
        return self.left_rewards[-1], self.right_rewards[-1]

    @cached_property
    def steps(self):
        th = np.array(self.initial_thresholds, dtype="float32")
        total_left_reward, total_right_reward = 0, 0
        data = []
        for la, ra, lr, rr in zip(
            self.left_actions, self.right_actions, self.left_rewards, self.right_rewards
        ):
            left_expected_reward = th[la] / SAMPLE_RESOLUTION
            right_expected_reward = th[ra] / SAMPLE_RESOLUTION

            th[la] *= DECAY_RATE
            th[ra] *= DECAY_RATE

            lr -= total_left_reward
            rr -= total_right_reward
            total_left_reward += lr
            total_right_reward += rr

            data.append(
                {
                    "left_action": la,
                    "right_action": ra,
                    "left_reward": lr,
                    "right_reward": rr,
                    "total_left_reward": total_left_reward,
                    "total_right_reward": total_right_reward,
                    "left_expected_reward": left_expected_reward,
                    "right_expected_reward": right_expected_reward,
                    "thresholds": np.array(th),
                }
            )

        return data

    def expected_rewards_estimation(self):
        l_expected, r_expected, th = [], [], []
        for d in self.steps:
            l_expected.append(d["left_expected_reward"])
            r_expected.append(d["right_expected_reward"])
            th.append(d["thresholds"] / SAMPLE_RESOLUTION)
        return l_expected, r_expected, th

    def total_expected_rewards(self):
        l_expected, r_expected = 0, 0
        for d in self.steps:
            l_expected += d["left_expected_reward"]
            r_expected += d["right_expected_reward"]
        return l_expected, r_expected

    def thresholds_at_the_end(self):
        th = self.initial_thresholds
        for d in self.steps:
            th = d["thresholds"]
        return th
