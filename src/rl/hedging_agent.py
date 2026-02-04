"""RL-based hedging agent for everlasting options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

from .networks import PolicyNetwork, ValueNetwork


@dataclass
class HedgingState:
    """
    State representation for the hedging agent.

    All values are normalized for neural network input.
    """
    # Price features
    spot_price: float          # Normalized by initial price
    spot_return: float         # Recent return
    spot_volatility: float     # Rolling volatility estimate

    # Option features
    delta: float               # Option delta [-1, 1]
    gamma: float               # Normalized gamma
    moneyness: float           # log(S/K)

    # Position features
    net_position: float        # Normalized by max position
    current_hedge: float       # Current hedge ratio [0, 1]

    # PnL features
    unrealized_pnl: float      # Normalized
    funding_rate: float        # Current funding rate

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for network input."""
        return np.array([
            self.spot_price,
            self.spot_return,
            self.spot_volatility,
            self.delta,
            self.gamma,
            self.moneyness,
            self.net_position,
            self.current_hedge,
            self.unrealized_pnl,
            self.funding_rate,
        ])

    @staticmethod
    def feature_dim() -> int:
        return 10


class RLHedgingAgent:
    """
    Reinforcement Learning agent for dynamic hedging.

    Uses an actor-critic architecture with policy gradient updates.
    The agent learns to adjust hedge ratios based on market conditions
    to maximize risk-adjusted returns.

    Parameters
    ----------
    learning_rate : float
        Learning rate for both policy and value networks
    gamma : float
        Discount factor for future rewards
    exploration_std : float
        Standard deviation of exploration noise
    hidden_dims : List[int]
        Hidden layer dimensions for networks
    pretrained : bool
        Whether to load pretrained weights
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        exploration_std: float = 0.1,
        hidden_dims: List[int] = [64, 32],
        pretrained: bool = True,
    ):
        self.gamma = gamma
        self.exploration_std = exploration_std
        self.training = True

        input_dim = HedgingState.feature_dim()

        # Actor (policy) network
        self.policy = PolicyNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
        )

        # Critic (value) network
        self.value = ValueNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
        )

        # Experience buffer for batch updates
        self._states: List[np.ndarray] = []
        self._actions: List[float] = []
        self._rewards: List[float] = []
        self._values: List[float] = []

        # Normalization stats
        self._price_mean = 3000.0
        self._price_std = 500.0
        self._pnl_scale = 10000.0
        self._position_scale = 1000.0

        # Performance tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

        if pretrained:
            self._initialize_pretrained()

    def _initialize_pretrained(self):
        """Initialize with reasonable pretrained weights for delta hedging."""
        # Initialize policy to approximate: hedge_ratio ≈ |delta|
        # This gives the agent a sensible starting point

        # Reset all weights to small values first
        for i in range(len(self.policy.weights)):
            self.policy.weights[i] *= 0.1
            self.policy.biases[i] *= 0.1

        # Set up the network to primarily follow delta
        # State features: [spot_price, spot_return, spot_volatility, delta, gamma,
        #                  moneyness, net_position, current_hedge, unrealized_pnl, funding_rate]
        # Delta is index 3, we want output ≈ |delta|

        if len(self.policy.weights) >= 2:
            # First layer: extract delta and pass through
            self.policy.weights[0][3, :] = 3.0   # Strong weight on delta
            self.policy.weights[0][7, :] = 0.5   # Some weight on current hedge (smoothing)
            self.policy.biases[0][:] = 0.0

            # Output layer: combine to get hedge ratio
            self.policy.weights[-1][:, 0] = 0.5
            self.policy.biases[-1][0] = 0.0  # Bias towards 0.5 hedge ratio after sigmoid

    def get_state(
        self,
        spot_price: float,
        price_history: np.ndarray,
        delta: float,
        gamma: float,
        strike: float,
        net_position: float,
        current_hedge: float,
        pnl: float,
        funding_rate: float,
        initial_price: float,
    ) -> HedgingState:
        """
        Construct normalized state from raw market data.

        Parameters
        ----------
        spot_price : float
            Current spot price
        price_history : np.ndarray
            Recent price history
        delta : float
            Option delta
        gamma : float
            Option gamma
        strike : float
            Option strike
        net_position : float
            Net trader position
        current_hedge : float
            Current hedge ratio
        pnl : float
            Current PnL
        funding_rate : float
            Current funding rate
        initial_price : float
            Initial spot price

        Returns
        -------
        HedgingState
            Normalized state for the agent
        """
        # Compute derived features
        if len(price_history) >= 2:
            spot_return = (price_history[-1] - price_history[-2]) / price_history[-2]
        else:
            spot_return = 0.0

        if len(price_history) >= 5:
            returns = np.diff(price_history[-5:]) / price_history[-5:-1]
            spot_volatility = np.std(returns) * np.sqrt(252)
        else:
            spot_volatility = 0.5

        moneyness = np.log(spot_price / strike) if strike > 0 else 0.0

        return HedgingState(
            spot_price=spot_price / initial_price - 1.0,  # Normalized around 0
            spot_return=np.clip(spot_return * 10, -1, 1),  # Scale returns
            spot_volatility=np.clip(spot_volatility, 0, 2),
            delta=delta,
            gamma=np.clip(gamma * 100, -1, 1),  # Scale gamma
            moneyness=np.clip(moneyness, -1, 1),
            net_position=np.clip(net_position / self._position_scale, -1, 1),
            current_hedge=current_hedge,
            unrealized_pnl=np.clip(pnl / self._pnl_scale, -1, 1),
            funding_rate=np.clip(funding_rate / 10, -1, 1),
        )

    def act(self, state: HedgingState, explore: bool = True) -> float:
        """
        Select hedge ratio action given state.

        Parameters
        ----------
        state : HedgingState
            Current market state
        explore : bool
            Whether to add exploration noise

        Returns
        -------
        float
            Hedge ratio in [0, 1]
        """
        state_array = state.to_array()

        # Forward pass through policy
        hedge_ratio = self.policy.forward(state_array, store_activations=self.training)

        # Add exploration noise during training
        if explore and self.training and self.exploration_std > 0:
            noise = np.random.randn() * self.exploration_std
            hedge_ratio = np.clip(hedge_ratio + noise, 0, 1)

        # Store for learning (always store to allow end_episode to work)
        value = self.value.forward(state_array)
        self._states.append(state_array)
        self._actions.append(hedge_ratio)
        self._values.append(value)

        return float(hedge_ratio)

    def compute_reward(
        self,
        pnl_change: float,
        hedge_cost: float = 0.0,
        risk_penalty: float = 0.001,
        volatility: float = 0.0,
    ) -> float:
        """
        Compute reward signal for the agent.

        The reward encourages:
        1. Positive PnL
        2. Low variance (Sharpe-like)
        3. Efficient hedging (minimize transaction costs)

        Parameters
        ----------
        pnl_change : float
            Change in PnL this step
        hedge_cost : float
            Transaction cost from rehedging
        risk_penalty : float
            Penalty coefficient for variance
        volatility : float
            Recent PnL volatility estimate

        Returns
        -------
        float
            Reward signal
        """
        # Base reward is PnL change (normalized) - scale down to prevent extreme updates
        reward = pnl_change / (self._pnl_scale * 10)

        # Subtract hedging costs
        reward -= hedge_cost / (self._pnl_scale * 10)

        # Risk penalty (penalize high variance) - reduced
        if volatility > 0:
            reward -= risk_penalty * volatility / self._pnl_scale

        # Clip reward to prevent extreme gradient updates
        reward = np.clip(reward, -1.0, 1.0)

        return reward

    def store_reward(self, reward: float):
        """Store reward for the last action."""
        if self._rewards is not None:
            self._rewards.append(reward)

    def update(self) -> Tuple[float, float]:
        """
        Update networks using collected experience.

        Uses advantage actor-critic (A2C) style update.

        Returns
        -------
        Tuple[float, float]
            (policy_loss, value_loss)
        """
        if len(self._rewards) < 2 or len(self._states) < 2:
            # Clear buffers and return
            self._states = []
            self._actions = []
            self._rewards = []
            self._values = []
            return 0.0, 0.0

        # Compute returns and advantages
        returns = []
        advantages = []

        # Compute discounted returns
        G = 0
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns)

        # Ensure values array matches returns length
        if len(self._values) < len(returns):
            # Pad with zeros if needed
            self._values.extend([0.0] * (len(returns) - len(self._values)))
        values = np.array(self._values[:len(returns)])

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute advantages
        advantages = returns - values

        # Update networks
        policy_loss = 0.0
        value_loss = 0.0

        for i, (state, advantage, ret) in enumerate(zip(self._states, advantages, returns)):
            # Re-forward to get activations
            self.policy.forward(state, store_activations=True)

            # Policy gradient update
            policy_loss += self.policy.update(advantage)

            # Value network update
            self.value.forward(state)
            value_loss += self.value.update(ret)

        # Track episode stats
        self.episode_rewards.append(sum(self._rewards))
        self.episode_lengths.append(len(self._rewards))

        # Clear buffers
        self._states = []
        self._actions = []
        self._rewards = []
        self._values = []

        n = max(len(advantages), 1)
        return policy_loss / n, value_loss / n

    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training

    def get_stats(self) -> dict:
        """Get training statistics."""
        if not self.episode_rewards:
            return {"episodes": 0}

        recent = self.episode_rewards[-10:]
        return {
            "episodes": len(self.episode_rewards),
            "mean_reward": np.mean(recent),
            "std_reward": np.std(recent),
            "mean_length": np.mean(self.episode_lengths[-10:]),
        }

    def save(self, filepath: str):
        """Save agent to file."""
        self.policy.save(filepath + "_policy.json")
        self.value.save(filepath + "_value.json")

    def load(self, filepath: str):
        """Load agent from file."""
        self.policy.load(filepath + "_policy.json")
        self.value.load(filepath + "_value.json")


class SimpleRLHedger:
    """
    Simplified RL hedger interface for use in the simulator.

    Wraps RLHedgingAgent with easy-to-use methods.
    The agent is initialized to mimic delta hedging and learns to improve.
    """

    def __init__(
        self,
        learning_rate: float = 0.0005,
        exploration: float = 0.05,
        training: bool = True,
    ):
        self.agent = RLHedgingAgent(
            learning_rate=learning_rate,
            exploration_std=exploration if training else 0.0,
            pretrained=True,  # Start with delta-hedging baseline
        )
        self.agent.set_training(training)
        self.training = training

        self._last_pnl = 0.0
        self._pnl_history: List[float] = []
        self._initial_price = None

    def get_hedge_ratio(
        self,
        spot_price: float,
        price_history: np.ndarray,
        delta: float,
        gamma: float,
        strike: float,
        net_position: float,
        current_hedge: float,
        pnl: float,
        funding_rate: float,
    ) -> float:
        """
        Get hedge ratio from RL agent.

        The agent outputs an adjustment to the baseline delta hedge.
        This ensures stable behavior while allowing learned improvements.

        Returns
        -------
        float
            Recommended hedge ratio [0, 1]
        """
        if self._initial_price is None:
            self._initial_price = spot_price

        state = self.agent.get_state(
            spot_price=spot_price,
            price_history=price_history,
            delta=delta,
            gamma=gamma,
            strike=strike,
            net_position=net_position,
            current_hedge=current_hedge,
            pnl=pnl,
            funding_rate=funding_rate,
            initial_price=self._initial_price,
        )

        # Compute reward from last step
        if self._last_pnl != 0 or pnl != 0:
            pnl_change = pnl - self._last_pnl

            # Estimate volatility
            self._pnl_history.append(pnl)
            if len(self._pnl_history) > 5:
                vol = np.std(np.diff(self._pnl_history[-5:]))
            else:
                vol = 0.0

            reward = self.agent.compute_reward(pnl_change, volatility=vol)
            self.agent.store_reward(reward)

        self._last_pnl = pnl

        # Get RL agent's output
        rl_output = self.agent.act(state, explore=self.training)

        # Blend with delta-hedging baseline for stability
        # RL output adjusts the baseline rather than replacing it entirely
        baseline_hedge = abs(delta)  # Standard delta hedge

        if self.training:
            # During training: 70% baseline + 30% RL adjustment
            hedge_ratio = 0.7 * baseline_hedge + 0.3 * rl_output
        else:
            # During inference: 50% baseline + 50% RL
            hedge_ratio = 0.5 * baseline_hedge + 0.5 * rl_output

        return float(np.clip(hedge_ratio, 0, 1))

    def end_episode(self) -> dict:
        """
        End episode and update agent.

        Returns
        -------
        dict
            Training statistics
        """
        self.agent.update()
        stats = self.agent.get_stats()

        # Reset for next episode
        self._last_pnl = 0.0
        self._pnl_history = []
        self._initial_price = None

        return stats
