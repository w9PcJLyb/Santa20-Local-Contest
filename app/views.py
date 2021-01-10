from matplotlib import animation, pyplot as plt
from django.http import HttpResponse

from app.models import Game


def visualization_view(request, object_id):
    game = Game.objects.filter(id=object_id).first()
    if not game:
        return HttpResponse(f"Can't find game with id {object_id}.")

    if (
        game.initial_thresholds is None
        or game.left_actions is None
        or game.right_actions is None
        or game.left_rewards is None
        or game.right_rewards is None
    ):
        return HttpResponse("Can't create a video.")

    fig, ax = plt.subplots(2, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [3, 1]})

    th_animation = ThresholdsAnimation(ax[0, 0], game)
    bandit_rewards_animation = BanditRewardsAnimation(ax[1, 0], game)
    agent_rewards_animation = AgentRewardsAnimation(ax[0, 1], game)
    WithoutAnimation(ax[1, 1], game)

    def animate(step):
        return (
            th_animation.animate(step)
            + bandit_rewards_animation.animate(step)
            + agent_rewards_animation.animate(step)
        )

    ani = animation.FuncAnimation(
        fig, animate, interval=50, blit=True, save_count=len(game.steps)
    )
    return HttpResponse(f"<h1>{game_info(game)}</h1>{ani.to_html5_video()}")


class AnimationABC:
    def __init__(self, ax, game, left_color="red", right_color="blue"):
        self.game = game
        self.left_agent_name = game.left_name
        self.right_agent_name = game.right_name
        self.left_color = left_color
        self.right_color = right_color
        self.num_bandits = len(game.initial_thresholds)

        self.plot_objects = self.init_func(ax)

    def init_func(self, ax):
        raise NotImplementedError()

    def animate(self, step):
        raise NotImplementedError()


class WithoutAnimation(AnimationABC):
    def init_func(self, ax):
        ax.set_axis_off()

    def animate(self, step):
        return ()


class ThresholdsAnimation(AnimationABC):
    def __init__(self, ax, game, *args, **kwargs):
        bandit_to_th = dict(enumerate(game.initial_thresholds.tolist()))
        self.sorted_bandits = sorted(bandit_to_th, key=lambda x: -bandit_to_th[x])
        self.bandit_to_order = {b: i for i, b in enumerate(self.sorted_bandits)}

        super().__init__(ax, game, *args, **kwargs)

    def order_bandits(self, values):
        return [values[i] for i in self.sorted_bandits]

    def init_func(self, ax):
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel("thresholds")

        th, *_ = ax.plot(
            self.order_bandits(self.game.initial_thresholds), "o", color="gray"
        )
        left_bandit, *_ = ax.plot([], "o", ms=10, color=self.left_color, alpha=0.5)
        right_bandit, *_ = ax.plot([], "o", ms=10, color=self.right_color, alpha=0.5)

        previous_left_bandit, *_ = ax.plot(
            [], "o", ms=8, color=self.left_color, alpha=0.25
        )
        previous_right_bandit, *_ = ax.plot(
            [], "o", ms=8, color=self.right_color, alpha=0.25
        )

        step_info = ax.text(70, 80, "Step 0")
        left_agent_info = ax.text(
            70, 70, f"{self.left_agent_name} - {0}", color=self.left_color
        )
        right_agent_info = ax.text(
            70, 60, f"{self.right_agent_name} - {0}", color=self.right_color
        )
        return (
            th,
            left_bandit,
            right_bandit,
            previous_left_bandit,
            previous_right_bandit,
            step_info,
            left_agent_info,
            right_agent_info,
        )

    def animate(self, step):
        (
            thresholds,
            left_bandit,
            right_bandit,
            previous_left_bandit,
            previous_right_bandit,
            step_info,
            left_agent_info,
            right_agent_info,
        ) = self.plot_objects

        info = self.game.steps[step]
        th = info["thresholds"]

        thresholds.set_ydata(self.order_bandits(th))

        left_bandit.set_data(
            [self.bandit_to_order[info["left_action"]], th[info["left_action"]]]
        )
        right_bandit.set_data(
            [self.bandit_to_order[info["right_action"]], th[info["right_action"]]]
        )

        step_info.set_text(f"Step {step}")
        left_agent_info.set_text(
            f"{self.left_agent_name} - {info['total_left_reward']}"
        )
        right_agent_info.set_text(
            f"{self.right_agent_name} - {info['total_right_reward']}"
        )

        if step > 0:
            info = self.game.steps[step - 1]
            th = info["thresholds"]
            previous_left_bandit.set_data(
                [self.bandit_to_order[info["left_action"]], th[info["left_action"]]]
            )
            previous_right_bandit.set_data(
                [self.bandit_to_order[info["right_action"]], th[info["right_action"]]]
            )

        return (
            thresholds,
            left_bandit,
            right_bandit,
            previous_left_bandit,
            previous_right_bandit,
            step_info,
            left_agent_info,
            right_agent_info,
        )


class BanditRewardsAnimation(AnimationABC):
    def __init__(self, ax, game, *args, **kwargs):
        bandit_to_th = dict(enumerate(game.initial_thresholds.tolist()))
        self.sorted_bandits = sorted(bandit_to_th, key=lambda x: -bandit_to_th[x])
        self.bandit_to_order = {b: i for i, b in enumerate(self.sorted_bandits)}
        self.bandit_distribution = self.get_bandit_distribution(game)

        super().__init__(ax, game, *args, **kwargs)

    def order_bandits(self, values):
        return [values[i] for i in self.sorted_bandits]

    def get_bandit_distribution(self, game):
        num_bandits = len(game.initial_thresholds)
        left_action_count = [0] * num_bandits
        right_action_count = [0] * num_bandits
        left_reward_count = [0] * num_bandits
        right_reward_count = [0] * num_bandits
        out = []
        for d in game.steps:
            left_action_count[d["left_action"]] += 1
            right_action_count[d["right_action"]] += 1
            left_reward_count[d["left_action"]] += d["left_reward"]
            right_reward_count[d["right_action"]] += d["right_reward"]
            out.append(
                {
                    "left_action_count": self.order_bandits(left_action_count),
                    "right_action_count": self.order_bandits(right_action_count),
                    "left_reward_count": self.order_bandits(left_reward_count),
                    "right_reward_count": self.order_bandits(right_reward_count),
                }
            )
        return out

    def init_func(self, ax):
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel("rewards")
        ax.set_ylim(
            0,
            max(
                self.bandit_distribution[-1]["left_action_count"]
                + self.bandit_distribution[-1]["right_action_count"]
            ),
        )
        left_rewards = ax.bar(
            range(self.num_bandits),
            [0] * self.num_bandits,
            color=self.left_color,
            alpha=0.5,
        )
        right_rewards = ax.bar(
            range(self.num_bandits),
            [0] * self.num_bandits,
            color=self.right_color,
            alpha=0.5,
        )
        left_turns = ax.bar(
            range(self.num_bandits),
            [0] * self.num_bandits,
            color=self.left_color,
            linewidth=0,
            alpha=0.2,
            width=1,
        )
        right_turns = ax.bar(
            range(self.num_bandits),
            [0] * self.num_bandits,
            color=self.right_color,
            linewidth=0,
            alpha=0.2,
            width=1,
        )
        return left_rewards, right_rewards, left_turns, right_turns

    def animate(self, step):
        left_rewards, right_rewards, left_turns, right_turns = self.plot_objects
        bd = self.bandit_distribution[step]

        for patches, data in [
            (left_rewards.patches, bd["left_reward_count"]),
            (right_rewards.patches, bd["right_reward_count"]),
            (left_turns.patches, bd["left_action_count"]),
            (right_turns.patches, bd["right_action_count"]),
        ]:
            for p, d in zip(patches, data):
                p.set_height(d)

        return ()


class AgentRewardsAnimation(AnimationABC):
    def init_func(self, ax):
        ax.set_ylabel("reward")
        ax.set_xlabel("time")
        ax.plot(self.game.left_rewards, color=self.left_color)
        ax.plot(self.game.right_rewards, color=self.right_color)
        time_line, *_ = ax.plot(
            [0, 0], [0, max(self.game.total_rewards())], color="black"
        )
        return (time_line,)

    def animate(self, step):
        time_line, *_ = self.plot_objects
        time_line.set_xdata([step, step])
        return (time_line,)


def game_info(game):
    l_name, r_name = game.left_name, game.right_name
    l_old, r_old = game.left_current_rating, game.right_current_rating
    l_new, r_new = game.left_new_rating, game.right_new_rating

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

    return f"{to_str(l_name, l_old, l_diff)} - {to_str(r_name, r_old, r_diff)}"
