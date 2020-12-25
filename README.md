# Santa20-Local-Contest
Django leaderboard for Santa 2020, https://www.kaggle.com/c/santa-2020/overview

### Features:

- Create your own leaderboard with django

![agents](https://github.com/w9PcJLyb/Santa20-Local-Contest/blob/main/assets/agents.png)

- Detail statistics for each agent

![agents](https://github.com/w9PcJLyb/Santa20-Local-Contest/blob/main/assets/agent.png)

- Each game is stored in a database, and you can watch them at any time

![game](https://github.com/w9PcJLyb/Santa20-Local-Contest/blob/main/assets/game.png)

- Add your own custom metrics for agent and match analysis

### Installation:

1. Clone this repo
2. Create a virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a development database:
   ```bash
   ./manage.py migrate
   ```
5. Create a superuser:
   ```bash
   ./manage.py createsuperuser
   ```
6. Start the Django development server:
   ```bash
   ./manage.py runserver
   ```
7. Go to http://127.0.0.1:8000/admin/app/

### Local leaderboard:

1. Add at least two agents via /admin/app/agent/add/
2. Run games:
   ```bash
   ./manage.py run_games -n 100
   ```
3. You can find your local LB here http://127.0.0.1:8000/admin/app/agent/
