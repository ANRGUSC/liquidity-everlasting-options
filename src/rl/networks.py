"""Simple neural networks for RL hedging (numpy-based, no external dependencies)."""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional
import json


class PolicyNetwork:
    """
    Policy network that outputs hedge ratio given market state.

    Uses a simple feedforward architecture with tanh activations.
    Output is squashed to [0, 1] for hedge ratio.

    Parameters
    ----------
    input_dim : int
        Dimension of state input
    hidden_dims : List[int]
        Hidden layer dimensions
    learning_rate : float
        Learning rate for policy gradient updates
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: List[int] = [64, 32],
        learning_rate: float = 0.001,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = learning_rate

        # Initialize weights
        self.weights = []
        self.biases = []

        dims = [input_dim] + hidden_dims + [1]
        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            w = np.random.randn(dims[i], dims[i+1]) * scale
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)

        # For storing gradients
        self._activations = []
        self._z_values = []

    def forward(self, x: np.ndarray, store_activations: bool = False) -> float:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : np.ndarray
            Input state vector
        store_activations : bool
            Whether to store activations for backprop

        Returns
        -------
        float
            Hedge ratio in [0, 1]
        """
        x = np.array(x).flatten()

        if store_activations:
            self._activations = [x.copy()]
            self._z_values = []

        # Hidden layers with tanh
        for i in range(len(self.weights) - 1):
            z = x @ self.weights[i] + self.biases[i]
            x = np.tanh(z)
            if store_activations:
                self._z_values.append(z)
                self._activations.append(x.copy())

        # Output layer with sigmoid for [0, 1]
        z = x @ self.weights[-1] + self.biases[-1]
        output = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))  # Sigmoid

        if store_activations:
            self._z_values.append(z)
            self._activations.append(output.copy())

        return float(output.flatten()[0])

    def update(self, reward: float, entropy_bonus: float = 0.01) -> float:
        """
        Update weights using policy gradient (REINFORCE).

        Parameters
        ----------
        reward : float
            Reward signal (higher = better)
        entropy_bonus : float
            Entropy regularization coefficient

        Returns
        -------
        float
            Gradient magnitude for monitoring
        """
        if not self._activations:
            return 0.0

        # Compute gradient of log probability
        # For deterministic policy, use reward as gradient signal
        output = self._activations[-1]

        # Gradient through sigmoid: output * (1 - output)
        d_output = output * (1 - output) * reward

        # Add entropy bonus (encourage exploration)
        entropy_grad = entropy_bonus * (0.5 - output)
        d_output += entropy_grad

        # Backpropagate
        grad_w = []
        grad_b = []

        delta = d_output

        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient for this layer
            a_prev = self._activations[i]
            gw = np.outer(a_prev, delta)
            gb = delta.flatten()

            grad_w.insert(0, gw)
            grad_b.insert(0, gb)

            if i > 0:
                # Backprop through tanh: 1 - tanh^2
                delta = (self.weights[i] @ delta) * (1 - self._activations[i]**2)

        # Update weights
        total_grad = 0.0
        for i in range(len(self.weights)):
            self.weights[i] += self.lr * grad_w[i]
            self.biases[i] += self.lr * grad_b[i]
            total_grad += np.sum(np.abs(grad_w[i]))

        return total_grad

    def add_noise(self, std: float = 0.1) -> float:
        """
        Add exploration noise to output.

        Parameters
        ----------
        std : float
            Standard deviation of noise

        Returns
        -------
        float
            Noisy hedge ratio clipped to [0, 1]
        """
        if not self._activations:
            return 0.5

        output = self._activations[-1]
        noisy = output + np.random.randn() * std
        return float(np.clip(noisy, 0, 1))

    def save(self, filepath: str):
        """Save network weights to file."""
        data = {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath: str):
        """Load network weights from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.input_dim = data["input_dim"]
        self.hidden_dims = data["hidden_dims"]
        self.weights = [np.array(w) for w in data["weights"]]
        self.biases = [np.array(b) for b in data["biases"]]


class ValueNetwork:
    """
    Value network for estimating state value (baseline for variance reduction).

    Parameters
    ----------
    input_dim : int
        Dimension of state input
    hidden_dims : List[int]
        Hidden layer dimensions
    learning_rate : float
        Learning rate
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: List[int] = [64, 32],
        learning_rate: float = 0.001,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = learning_rate

        # Initialize weights
        self.weights = []
        self.biases = []

        dims = [input_dim] + hidden_dims + [1]
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            w = np.random.randn(dims[i], dims[i+1]) * scale
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)

        self._activations = []

    def forward(self, x: np.ndarray) -> float:
        """Forward pass returning value estimate."""
        x = np.array(x).flatten()
        self._activations = [x.copy()]

        for i in range(len(self.weights) - 1):
            z = x @ self.weights[i] + self.biases[i]
            x = np.tanh(z)
            self._activations.append(x.copy())

        # Linear output
        value = x @ self.weights[-1] + self.biases[-1]
        self._activations.append(value.copy())

        return float(value.flatten()[0])

    def update(self, target: float) -> float:
        """
        Update value network using MSE loss.

        Parameters
        ----------
        target : float
            Target value (actual return)

        Returns
        -------
        float
            Loss value
        """
        if not self._activations:
            return 0.0

        prediction = self._activations[-1]
        error = target - prediction
        loss = 0.5 * error ** 2

        # Backprop
        delta = error  # Gradient of MSE

        for i in range(len(self.weights) - 1, -1, -1):
            a_prev = self._activations[i]
            gw = np.outer(a_prev, delta)
            gb = delta.flatten()

            self.weights[i] += self.lr * gw
            self.biases[i] += self.lr * gb

            if i > 0:
                delta = (self.weights[i] @ delta) * (1 - self._activations[i]**2)

        return float(loss)
