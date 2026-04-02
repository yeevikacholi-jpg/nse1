"""
Step 5 — Model Building
========================
Defines a Gym trading environment and trains a PPO agent
(stable-baselines3) to generate BUY / HOLD / SELL signals.

Actions:  0 = HOLD, 1 = BUY, 2 = SELL
Reward:   portfolio return at each step minus transaction cost
"""

import os
import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from config import CONFIG


FEATURE_COLS = [
    "ret_1d", "ret_5d", "rsi", "macd", "macd_sig",
    "bb_width", "sma_10", "sma_30", "obv",
]


# ── Environment ───────────────────────────────────────────────────────────────

class TradingEnv(gym.Env):
    """
    Single-stock discrete-action trading environment.
    State  : window of FEATURE_COLS (normalised)
    Action : 0 HOLD | 1 BUY | 2 SELL
    Reward : log return of portfolio at each step
    """
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, window: int = 10):
        super().__init__()
        self.df     = df.reset_index(drop=True)
        self.window = window
        self.n      = len(df)

        n_features  = len(FEATURE_COLS) * window
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL

        self._reset_state()

    def _reset_state(self):
        self.idx        = self.window
        self.position   = 0          # 0 = flat, 1 = long
        self.cash       = CONFIG["initial_capital"]
        self.shares     = 0
        self.portfolio  = self.cash
        self.prev_port  = self.cash

    def _get_obs(self):
        window_df = self.df.iloc[self.idx - self.window: self.idx][FEATURE_COLS]
        obs = window_df.values.flatten().astype(np.float32)
        # Normalise
        std = obs.std() + 1e-8
        obs = (obs - obs.mean()) / std
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs()

    def step(self, action):
        price = self.df.loc[self.idx, "Close"]
        cost  = CONFIG["transaction_cost"]

        if action == 1 and self.position == 0:      # BUY
            self.shares   = int(self.cash / (price * (1 + cost)))
            self.cash    -= self.shares * price * (1 + cost)
            self.position = 1
        elif action == 2 and self.position == 1:    # SELL
            self.cash    += self.shares * price * (1 - cost)
            self.shares   = 0
            self.position = 0

        self.portfolio = self.cash + self.shares * price
        reward = np.log(self.portfolio / (self.prev_port + 1e-8))
        self.prev_port = self.portfolio

        self.idx += 1
        done = self.idx >= self.n - 1

        return self._get_obs(), float(reward), done, {}


# ── Train ─────────────────────────────────────────────────────────────────────

def train(df_train: pd.DataFrame, timesteps: int = 5_000) -> PPO:
    env = DummyVecEnv([lambda: TradingEnv(df_train)])
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
    )
    print(f"[train_model] Training PPO for {timesteps:,} timesteps ...")
    model.learn(total_timesteps=timesteps)
    os.makedirs("models", exist_ok=True)
    model.save(CONFIG["model_path"])
    print(f"[train_model] Model saved to {CONFIG['model_path']}")
    return model


def run_training(timesteps: int = 5_000) -> PPO:
    df = pd.read_csv("data/features.csv")
    split = int(len(df) * CONFIG["train_split"])
    df_train = df.iloc[:split].copy()
    return train(df_train, timesteps=timesteps)


if __name__ == "__main__":
    run_training()
