import io
import base64
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
from django.contrib import admin
from django.utils.safestring import mark_safe

from app.models import Agent, Game, GameResult


class AgentAdmin(admin.ModelAdmin):
    list_display = ("name", "rating", "created_at")
    readonly_fields = (
        "id",
        "created_at",
        "num_games",
        "elo_rating",
        "statistics",
        "last_games",
    )
    fields = (
        "id",
        "name",
        "source",
        "file",
        "created_at",
        "num_games",
        "elo_rating",
        "statistics",
        "last_games",
    )

    @staticmethod
    def elo_rating(obj):
        return round(obj.rating, 1)

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
                    "id": a_id,
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
                    "url": f"<a href=/admin/app/game/{g['id']}>url</a>",
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
        "threshold_distribution",
    )
    list_filter = (
        "left_agent__name",
        "right_agent__name",
        "status",
        "result",
        "started",
    )
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
        "threshold_distribution",
    )

    @staticmethod
    def rating(obj):
        info = (
            Game.objects.filter(id=obj.id)
            .values_list(
                "left_agent__name",
                "right_agent__name",
                "left_current_rating",
                "right_current_rating",
                "left_new_rating",
                "right_new_rating",
            )
            .last()
        )
        if not info:
            return ""

        if any(x is None for x in info):
            return ""

        l_name, r_name, l_old, r_old, l_new, r_new = info
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
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))

        ax[0].hist(obj.initial_thresholds, bins=20, range=(0, 100))
        ax[0].set_xlabel("threshold")
        ax[0].set_ylabel("number of bandits")
        ax[0].set_title("at the beginning")

        ax[1].hist(obj.thresholds_at_the_end(), bins=20, range=(0, 100))
        ax[1].set_xlabel("threshold")
        ax[1].set_title("at the end")

        return mark_safe(f'<img src="data:image/png;base64, {fig_to_base64(fig)}">')

    @staticmethod
    def rewards_over_time(obj):
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))

        ax[0].plot(obj.left_rewards, label=obj.left_agent.name)
        ax[0].plot(obj.right_rewards, label=obj.right_agent.name)
        ax[0].legend()
        ax[0].set_title("reward")
        ax[0].set_xlabel("time")

        gap = obj.left_rewards.astype("int16") - obj.right_rewards
        if gap[-1] < 0:
            gap = -gap
        ax[1].plot(gap)
        ax[1].set_title("gap")
        ax[1].set_xlabel("time")

        l_expected, r_expected = obj.expected_rewards_estimation()
        ax[2].plot(l_expected, label=obj.left_agent.name, alpha=0.5)
        ax[2].plot(r_expected, label=obj.right_agent.name, alpha=0.5)
        ax[2].legend()
        ax[2].set_title("expected reward")
        ax[2].set_xlabel("time")

        return mark_safe(f'<img src="data:image/png;base64, {fig_to_base64(fig)}">')


def fig_to_base64(fig):
    # https://stackoverflow.com/a/49016797
    img = io.BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    b = base64.b64encode(img.getvalue())
    return b.decode("utf-8")


admin.site.register(Agent, AgentAdmin)
admin.site.register(Game, GameAdmin)
