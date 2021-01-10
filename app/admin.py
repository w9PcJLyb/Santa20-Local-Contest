import io
import base64
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict
from django.urls import path
from django.contrib import admin
from django.utils.safestring import mark_safe

from app.views import visualization_view
from app.models import Agent, Game, GameResult


class AgentAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "rating", "created_at", "enabled")
    readonly_fields = (
        "id",
        "created_at",
        "num_games",
        "elo_rating",
        "rank",
        "win_ratio",
        "statistics",
        "last_games",
    )
    list_filter = ("enabled",)
    fields = (
        "id",
        "name",
        # "description",
        "enabled",
        "source",
        "file",
        "created_at",
        "num_games",
        "elo_rating",
        "rank",
        "win_ratio",
        "statistics",
        "last_games",
    )

    @staticmethod
    def elo_rating(obj):
        return round(obj.rating, 1)

    @staticmethod
    def rank(obj):
        agents = Agent.objects.all().order_by("-rating").values_list("id", flat=True)
        for rank, agent_id in enumerate(agents, start=1):
            if agent_id == obj.id:
                return rank

    @staticmethod
    def win_ratio(obj):
        def side_stat(side, win_result):
            games = obj.games_qs().filter(**{f"{side}_agent": obj})
            num_games = games.count()
            if num_games:
                num_win_games = games.filter(result=win_result).count()
                ratio = round(num_win_games / num_games * 100)
            else:
                ratio = "Nan"
            return f"{side}: <strong>{ratio}%</strong> ({num_games} games)"

        left = side_stat("left", GameResult.LEFT_WON)
        right = side_stat("right", GameResult.RIGHT_WON)
        return mark_safe(f"{left} - {right}")

    @staticmethod
    def statistics(obj):
        competitor_to_stats = defaultdict(lambda: {"win": 0, "lose": 0, "draw": 0})

        for left_id, right_id, result in (
            obj.games_qs()
            .values_list("left_agent_id", "right_agent_id", "result")
            .iterator()
        ):
            if left_id == right_id:
                continue

            k = None
            competitor_id = None
            if left_id == obj.id:
                competitor_id = right_id
                if result == GameResult.LEFT_WON:
                    k = "win"
                elif result == GameResult.DRAW:
                    k = "draw"
                elif result == GameResult.RIGHT_WON:
                    k = "lose"
            elif right_id == obj.id:
                competitor_id = left_id
                if result == GameResult.RIGHT_WON:
                    k = "win"
                elif result == GameResult.DRAW:
                    k = "draw"
                elif result == GameResult.LEFT_WON:
                    k = "lose"

            if not k or not competitor_id:
                continue

            competitor_to_stats[competitor_id][k] += 1

        if not competitor_to_stats:
            return ""

        agent_id_to_name = {}
        agent_id_to_rating = {}
        for agent_id, name, rating in Agent.objects.filter(
            id__in=competitor_to_stats
        ).values_list("id", "name", "rating"):
            agent_id_to_name[agent_id] = name
            agent_id_to_rating[agent_id] = int(rating)

        data = []
        for a_id, stats in competitor_to_stats.items():
            data.append(
                {
                    "id": f"<a href=/admin/app/agent/{a_id}>{a_id}</a>",
                    "name": agent_id_to_name.get(a_id),
                    "rating": agent_id_to_rating.get(a_id),
                    "num_win": stats["win"],
                    "num_draw": stats["draw"],
                    "num_lose": stats["lose"],
                }
            )

        df = pd.DataFrame(data)
        df["num_games"] = df["num_win"] + df["num_draw"] + df["num_lose"]
        df["win_ratio"] = df["num_win"] / df["num_games"]
        df.sort_values("rating", inplace=True, ascending=False)
        df = df.append(
            {
                "id": "-",
                "name": "<strong>Total</strong>",
                "rating": "-",
                "num_win": df["num_win"].sum(),
                "num_draw": df["num_draw"].sum(),
                "num_lose": df["num_lose"].sum(),
                "num_games": df["num_games"].sum(),
                "win_ratio": df["num_win"].sum() / df["num_games"].sum(),
            },
            ignore_index=True,
        )

        return mark_safe(
            df.to_html(
                index=False,
                float_format=lambda x: f"{round(100 * x, 2)}%",
                border=0,
                escape=False,
            )
        )

    @staticmethod
    def last_games(obj, num_games=50):
        data = []
        for g in (
            obj.games_qs()
            .order_by("-started")[:num_games]
            .values(
                "id",
                "started",
                "right_agent",
                "right_agent__name",
                "right_new_rating",
                "right_current_rating",
                "left_agent",
                "left_agent__name",
                "left_new_rating",
                "left_current_rating",
            )
        ):
            if g["left_agent"] == obj.id:
                opponent_name = g["right_agent__name"]
                opponent_score = g["right_current_rating"]
                new_rating = g["left_new_rating"]
                old_rating = g["left_current_rating"]
            elif g["right_agent"] == obj.id:
                opponent_name = g["left_agent__name"]
                opponent_score = g["left_current_rating"]
                new_rating = g["right_new_rating"]
                old_rating = g["right_current_rating"]
            else:
                continue

            if new_rating is None or old_rating is None:
                rating_change = "-"
            else:
                rating_change = new_rating - old_rating
                if rating_change > 0:
                    rating_change = (
                        f"<span style='color: green'>+{round(rating_change, 1)}</span>"
                    )
                else:
                    rating_change = f"<span style='color: red'>-{round(abs(rating_change), 1)}</span>"

            data.append(
                {
                    "result": rating_change,
                    "name": opponent_name,
                    "rating": int(opponent_score),
                    "date": g["started"].strftime("%Y-%m-%d %H:%M"),
                    "link": f"<a href=/admin/app/game/{g['id']}>link</a>",
                }
            )

        df = pd.DataFrame(data)
        return mark_safe(df.to_html(index=False, border=0, escape=False))


class GameAdmin(admin.ModelAdmin):
    readonly_fields = (
        "id",
        "started",
        "finished",
        "rating",
        "execution_time",
        "rewards",
        "expected_rewards",
        "rewards_over_time",
        "expected_rewards_graph",
        "threshold_distribution",
        "visualization",
    )
    list_filter = ("left_agent", "right_agent", "status", "result", "started")
    list_display = ("id", "left_agent", "right_agent", "started", "status", "result")
    fields = (
        "id",
        "started",
        "finished",
        "execution_time",
        "left_agent",
        "right_agent",
        "status",
        "result",
        "rating",
        "rewards",
        "expected_rewards",
        "rewards_over_time",
        "expected_rewards_graph",
        "threshold_distribution",
        "visualization",
    )

    def has_add_permission(self, request):
        return False

    def get_urls(self):
        visualization_url = path(
            "<path:object_id>/video/",
            self.admin_site.admin_view(visualization_view),
            name="video",
        )
        return [visualization_url] + super().get_urls()

    @staticmethod
    def rating(obj):
        l_name, r_name = obj.left_name, obj.right_name
        l_old, r_old = obj.left_current_rating, obj.right_current_rating
        l_new, r_new = obj.left_new_rating, obj.right_new_rating

        if l_old is None or r_old is None or l_new is None or r_new is None:
            return ""

        l_diff = l_new - l_old
        r_diff = r_new - r_old

        def to_str(name, old, diff):
            if diff > 0:
                return (
                    f"<strong>'{name}'</strong> {round(old, 1)} "
                    f"(<span style='color: green'>+{round(diff, 1)}</span>)"
                )
            else:
                return (
                    f"'{name}' {round(old, 1)} "
                    f"(<span style='color: red'>-{round(abs(diff), 1)}</span>)"
                )

        return mark_safe(
            f"{to_str(l_name, l_old, l_diff)} - {to_str(r_name, r_old, r_diff)}"
        )

    @staticmethod
    def rewards(obj):
        l, r = obj.total_rewards()
        return f"{l} - {r}"

    @staticmethod
    def expected_rewards(obj):
        l, r = obj.total_expected_rewards()
        return f"{int(l)} - {int(r)}"

    @staticmethod
    def threshold_distribution(obj):
        if obj.initial_thresholds is None:
            return "-"

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        ax[0].hist(obj.initial_thresholds, bins=20, range=(0, 100), color="gray")
        ax[0].set_xlabel("threshold")
        ax[0].set_ylabel("number of bandits")
        ax[0].set_title("at the beginning")

        ax[1].hist(obj.thresholds_at_the_end(), bins=20, range=(0, 100), color="gray")
        ax[1].set_xlabel("threshold")
        ax[1].set_title("at the end")

        return fig_to_html(fig)

    @staticmethod
    def rewards_over_time(obj):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        ax[0].plot(obj.left_rewards, label=obj.left_name, color="red")
        ax[0].plot(obj.right_rewards, label=obj.right_name, color="blue")
        ax[0].legend()
        ax[0].set_title("reward")
        ax[0].set_xlabel("time")

        gap = obj.left_rewards.astype("int16") - obj.right_rewards
        if gap[-1] < 0:
            gap = -gap
        ax[1].plot(gap, color="gray")
        ax[1].set_title("gap")
        ax[1].set_xlabel("time")

        return fig_to_html(fig)

    @staticmethod
    def expected_rewards_graph(obj):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        l_expected, r_expected, th = obj.expected_rewards_estimation()
        ax[0].plot(l_expected, ".", alpha=0.5, color="gray")
        ax[0].plot(np.max(th, axis=1), label="max", color="black")
        ax[0].plot(
            np.percentile(th, q=50, axis=1), label="50% percentile", color="blue"
        )
        ax[0].legend()
        ax[0].set_title(obj.left_name)
        ax[0].set_xlabel("time")

        l_expected, r_expected, th = obj.expected_rewards_estimation()
        ax[1].plot(r_expected, ".", alpha=0.5, color="gray")
        ax[1].plot(np.max(th, axis=1), label="max", color="black")
        ax[1].plot(
            np.percentile(th, q=50, axis=1), label="50% percentile", color="blue"
        )
        ax[1].legend()
        ax[1].set_title(obj.right_name)
        ax[1].set_xlabel("time")

        return fig_to_html(fig)

    @staticmethod
    def visualization(obj):
        return mark_safe(
            f'<a class="button" href=/admin/app/game/{obj.id}/video>VIDEO</a>'
            f'<small><i><span style="color: red;"> <--- It can take a while!</span></i></small>'
        )


def fig_to_html(fig):
    encoded = __fig_to_base64(fig)
    content = __prepare_content(
        f'<img src="data:image/png;base64, {encoded.decode("utf-8")}">'
    )
    return content


def __fig_to_base64(fig):
    # https://stackoverflow.com/a/49016797
    img = io.BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    return base64.b64encode(img.getvalue())


def __prepare_content(content):
    return mark_safe(
        f'<div style="overflow-y: hidden; overflow-x: auto;">{content}</div>'
    )


admin.site.register(Agent, AgentAdmin)
admin.site.register(Game, GameAdmin)
