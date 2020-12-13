import numpy as np
from kaggle_environments import make
from django.utils import timezone
from django.core.management.base import BaseCommand

from app.models import Game, Agent, GameStatus, GameResult


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            "-n", "--num_games", default=np.inf, type=int, dest="num_games"
        )

    def handle(self, *args, **options):
        num_games = options["num_games"]
        agent_id_list = list(
            Agent.objects.filter(file__isnull=False).values_list("id", flat=True)
        )
        if len(agent_id_list) < 2:
            self.stdout.write(self.style.ERROR("There must be at least 2 agents."))
            return

        c = 1
        env = make("mab", debug=True)
        while c <= num_games:
            left_agent_id, right_agent_id = choice_agents_for_game(agent_id_list)

            try:
                game = run_game(env, left_agent_id, right_agent_id)
            except Exception as e:
                self.stdout.write(self.style.ERROR(e))
                return

            self.stdout.write(f"{c} - {game}")
            c += 1


def choice_agents_for_game(agent_id_list):
    return np.random.choice(agent_id_list, size=2, replace=False)


def run_game(env, left_agent_id, right_agent_id):
    if left_agent_id == right_agent_id:
        raise ValueError("Agents must be different")

    left_agent = Agent.objects.filter(id=left_agent_id).first()
    if not left_agent:
        raise ValueError(f"Can't find agent with id {left_agent_id}.")

    right_agent = Agent.objects.filter(id=right_agent_id).first()
    if not right_agent:
        raise ValueError(f"Can't find agent with id {right_agent_id}.")

    env.reset()
    game = Game.objects.create(
        left_agent=left_agent,
        right_agent=right_agent,
        left_current_rating=left_agent.rating,
        right_current_rating=right_agent.rating,
        configuration=env.configuration,
    )
    env.run([left_agent.file.path, right_agent.file.path])

    num_steps = len(env.steps) - 1
    left_actions = np.zeros(num_steps, dtype=np.uint8)
    right_actions = np.zeros(num_steps, dtype=np.uint8)
    left_rewards = np.zeros(num_steps, dtype=np.uint16)
    right_rewards = np.zeros(num_steps, dtype=np.uint16)
    for i, s in enumerate(env.steps[1:]):
        left_env, right_env = s
        left_actions[i], right_actions[i] = left_env["action"], right_env["action"]
        left_rewards[i], right_rewards[i] = left_env["reward"], right_env["reward"]

    left_total_reward, right_total_reward = left_rewards[-1], right_rewards[-1]

    if left_total_reward > right_total_reward:
        result = GameResult.LEFT_WON
    elif left_total_reward < right_total_reward:
        result = GameResult.RIGHT_WON
    else:
        result = GameResult.DRAW

    left_new_score, right_new_score = find_new_scores(
        left_agent.rating, right_agent.rating, result
    )

    game.initial_thresholds = np.array(
        env.steps[0][0]["observation"]["thresholds"], dtype=np.uint8
    )
    game.left_actions = left_actions
    game.right_actions = right_actions
    game.left_rewards = left_rewards
    game.right_rewards = right_rewards
    game.result = result
    game.status = GameStatus.FINISHED
    game.left_new_rating = left_new_score
    game.right_new_rating = right_new_score
    game.finished = timezone.now()
    game.save()

    left_agent.rating = left_new_score
    left_agent.save(update_fields=["rating"])

    right_agent.rating = right_new_score
    right_agent.save(update_fields=["rating"])

    return game


def find_new_scores(ra, rb, result, k=32):
    ea, eb = expected_scores(ra, rb)
    if result == GameResult.LEFT_WON:
        sa, sb = 1, 0
    elif result == GameResult.RIGHT_WON:
        sa, sb = 0, 1
    elif result == GameResult.DRAW:
        sa, sb = 0.5, 0.5
    else:
        raise ValueError(f"Unknown result '{result}'.")
    return ra + k * (sa - ea), rb + k * (sb - eb)


def expected_scores(a, b):
    qa = 10 ** (a / 400)
    qb = 10 ** (b / 400)
    s = qa + qb
    return qa / s, qb / s
