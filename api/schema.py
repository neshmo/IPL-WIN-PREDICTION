from pydantic import BaseModel

class PreMatchInput(BaseModel):
    team1_elo: float
    team2_elo: float
    elo_diff: float
    chasing_win_rate: float

class LiveInput(BaseModel):
    runs_remaining: int
    balls_remaining: int
    wickets_in_hand: int
    current_run_rate: float
    required_run_rate: float
