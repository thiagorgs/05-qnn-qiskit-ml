"""
Quantum Neural Network (QNN) Binary Classifier

Implements a parameterized quantum circuit (PQC) as a binary classifier
using Qiskit's Statevector simulator. The QNN encodes classical data into
quantum states via angle encoding, processes it through variational layers,
and classifies based on expectation values of Pauli observables.

Architecture:
  1. Feature Map: Angle-encodes classical features into quantum states
  2. Variational Ansatz: Parameterized RY/CNOT layers (learnable)
  3. Measurement: Expectation value of Z operator on first qubit
  4. Classification: Threshold on expectation value

References:
  - Schuld, M., Brougham, T., & Killoran, N. (2022). QML_Zoo paper
  - Benedetti, M., et al. (2019). Quantum Machine Learning
"""

import numpy as np
from typing import Tuple, Dict, List
from scipy.optimize import minimize
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp


class QuantumNeuralNetwork:
    """
    Quantum Neural Network using a Parameterized Quantum Circuit (PQC).

    Attributes:
        n_qubits (int): Number of qubits in the quantum circuit
        n_layers (int): Number of variational ansatz layers
        simulator: Qiskit statevector simulator for exact computation
    """

    def __init__(self, n_qubits: int = 2, n_layers: int = 2):
        """
        Initialize the Quantum Neural Network.

        Args:
            n_qubits: Number of qubits (should match feature dimension)
            n_layers: Number of variational ansatz layers
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.simulator = AerSimulator(method='statevector')

    def _build_feature_map(self, features: np.ndarray) -> QuantumCircuit:
        """
        Build the feature map: angle-encode classical features into quantum state.

        Uses angle encoding (RY gates) where each feature is mapped to a rotation angle.
        For feature x_i on qubit i: RY(π * x_i) if x_i ∈ [0,1]

        Args:
            features: Classical feature vector of shape (n_features,)

        Returns:
            QuantumCircuit implementing the feature map
        """
        qc = QuantumCircuit(self.n_qubits, name='feature_map')

        # Encode each feature into corresponding qubit
        for i, feature in enumerate(features):
            if i >= self.n_qubits:
                break
            # Scale feature to [0, π] for RY rotation
            angle = np.pi * feature
            qc.ry(angle, i)

        return qc

    def _build_variational_ansatz(self, params: np.ndarray) -> QuantumCircuit:
        """
        Build the variational ansatz: parameterized RY gates + CNOT entanglement.

        Structure per layer:
          - RY(theta) on each qubit (parameterized)
          - CNOT ladder for entanglement: CNOT(i, i+1) for all i

        Args:
            params: Flat array of parameters, shape (n_qubits * n_layers,)

        Returns:
            QuantumCircuit implementing the variational ansatz
        """
        qc = QuantumCircuit(self.n_qubits, name='ansatz')

        param_idx = 0
        for layer in range(self.n_layers):
            # RY rotations (parameterized)
            for qubit in range(self.n_qubits):
                qc.ry(params[param_idx], qubit)
                param_idx += 1

            # CNOT entanglement ladder
            for qubit in range(self.n_qubits - 1):
                qc.cx(qubit, qubit + 1)

        return qc

    def _build_qnn_circuit(self, features: np.ndarray, params: np.ndarray) -> QuantumCircuit:
        """
        Build complete QNN circuit: feature map + variational ansatz.

        Args:
            features: Classical feature vector
            params: Variational parameters

        Returns:
            Complete QuantumCircuit ready for evaluation
        """
        qc = QuantumCircuit(self.n_qubits)

        # Add feature map
        feature_circuit = self._build_feature_map(features)
        qc.compose(feature_circuit, inplace=True)

        # Add variational ansatz
        ansatz_circuit = self._build_variational_ansatz(params)
        qc.compose(ansatz_circuit, inplace=True)

        return qc

    def forward(self, features: np.ndarray, params: np.ndarray) -> float:
        """
        Forward pass: compute expectation value <Z_0> (Pauli Z on qubit 0).

        The expectation value is used as the QNN output for classification.
        Output ∈ [-1, 1], where:
          - Output ≈ +1 indicates class 1
          - Output ≈ -1 indicates class 0

        Args:
            features: Single feature vector of shape (n_features,)
            params: Variational parameters of shape (n_qubits * n_layers,)

        Returns:
            Expectation value <Z_0> ∈ [-1, 1]
        """
        qc = self._build_qnn_circuit(features, params)

        # Define observable: Z on qubit 0
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1.0)])

        # Use Estimator primitive for expectation value computation
        estimator = Estimator(backend=self.simulator)
        job = estimator.run(qc, observable)
        result = job.result()

        expectation_value = result.values[0].real
        return expectation_value

    def predict(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels for input features.

        Classification rule:
          - If <Z_0> > 0 → class 1
          - If <Z_0> ≤ 0 → class 0

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            params: Variational parameters

        Returns:
            Predicted class labels of shape (n_samples,)
        """
        predictions = []
        for features in X:
            expectation = self.forward(features, params)
            # Threshold at 0
            prediction = 1 if expectation > 0 else 0
            predictions.append(prediction)

        return np.array(predictions)

    def loss_function(self, params: np.ndarray, X_train: np.ndarray,
                     y_train: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss on training data.

        Loss = -∑[y * log(p) + (1-y) * log(1-p)]

        where p = (1 + <Z_0>) / 2 (maps [-1,1] to [0,1])

        Args:
            params: Current variational parameters
            X_train: Training features
            y_train: Training labels (0 or 1)

        Returns:
            Scalar loss value
        """
        loss = 0.0

        for features, label in zip(X_train, y_train):
            # Forward pass
            expectation = self.forward(features, params)

            # Map expectation value [-1, 1] to probability [0, 1]
            prob_class_1 = (1 + expectation) / 2
            prob_class_1 = np.clip(prob_class_1, 1e-7, 1 - 1e-7)  # Numerical stability

            # Binary cross-entropy
            if label == 1:
                loss += -np.log(prob_class_1)
            else:
                loss += -np.log(1 - prob_class_1)

        return loss / len(X_train)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              maxiter: int = 100, method: str = 'COBYLA') -> Dict:
        """
        Train the QNN using classical optimization.

        Uses scipy.optimize.minimize with COBYLA or L-BFGS-B optimizer.
        COBYLA is recommended for quantum circuits as it doesn't require gradients.

        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training labels of shape (n_samples,)
            maxiter: Maximum iterations for optimizer
            method: Optimization method ('COBYLA' or 'L-BFGS-B')

        Returns:
            Dictionary containing:
              - 'params': Optimized parameters
              - 'loss_history': Loss at each iteration
              - 'result': scipy optimization result
        """
        # Initialize parameters randomly
        n_params = self.n_qubits * self.n_layers
        initial_params = np.random.uniform(0, 2 * np.pi, n_params)

        # Track loss history
        loss_history = []

        def loss_with_history(params):
            loss = self.loss_function(params, X_train, y_train)
            loss_history.append(loss)
            if len(loss_history) % 10 == 0:
                print(f"Iteration {len(loss_history)}: Loss = {loss:.6f}")
            return loss

        # Optimize
        print(f"\nTraining QNN with {self.n_qubits} qubits, {self.n_layers} layers...")
        print(f"Total parameters: {n_params}")
        print(f"Training samples: {len(X_train)}\n")

        result = minimize(
            loss_with_history,
            initial_params,
            method=method,
            options={'maxiter': maxiter, 'disp': False}
        )

        print(f"\nOptimization complete!")
        print(f"Final loss: {result.fun:.6f}")
        print(f"Success: {result.success}")

        return {
            'params': result.x,
            'loss_history': loss_history,
            'result': result
        }


def generate_dataset(n_samples: int = 100, noise: float = 0.15,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate binary classification dataset using make_moons.

    The make_moons dataset creates two interleaving half circles,
    making it a non-linearly separable 2D classification problem.
    Perfect for demonstrating quantum advantage in learning.

    Args:
        n_samples: Total number of samples
        noise: Standard deviation of noise added to data
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (features, labels) where:
          - features: shape (n_samples, 2), normalized to [0, 1]
          - labels: shape (n_samples,), binary {0, 1}
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    # Normalize features to [0, 1] for angle encoding
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    return X, y


def plot_decision_boundary(qnn: QuantumNeuralNetwork, params: np.ndarray,
                          X: np.ndarray, y: np.ndarray, title: str = "Decision Boundary"):
    """
    Plot the decision boundary learned by the QNN in 2D space.

    Args:
        qnn: Trained QuantumNeuralNetwork instance
        params: Trained parameters
        X: Feature matrix (n_samples, 2)
        y: Labels
        title: Plot title
    """
    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = qnn.predict(mesh_points, params)
    Z = predictions.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])

    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                         edgecolors='black', s=100, alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(scatter)
    plt.tight_layout()

    return plt.gcf()


def plot_training_loss(loss_history: List[float]):
    """
    Plot training loss curve.

    Args:
        loss_history: List of loss values during training
    """
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, linewidth=2, color='navy')
    plt.xlabel('Iteration')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.title('QNN Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def main():
    """
    Main execution: generate dataset, train QNN, evaluate, and visualize.
    """
    print("=" * 70)
    print("Quantum Neural Network Binary Classifier")
    print("Using Parameterized Quantum Circuits (PQC) with Qiskit")
    print("=" * 70)

    # Generate dataset
    print("\nGenerating make_moons dataset...")
    X, y = generate_dataset(n_samples=100, noise=0.15)
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Initialize QNN
    n_qubits = 2
    n_layers = 2
    qnn = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers)

    # Train
    training_result = qnn.train(X, y, maxiter=150, method='COBYLA')
    trained_params = training_result['params']
    loss_history = training_result['loss_history']

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    y_pred = qnn.predict(X, trained_params)
    accuracy = accuracy_score(y, y_pred)
    print(f"\nAccuracy on training set: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Class 0', 'Class 1']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)

    # Visualizations
    print("\nGenerating visualizations...")

    # Decision boundary plot
    fig1 = plot_decision_boundary(qnn, trained_params, X, y,
                                   title="QNN Decision Boundary (2D Moons Dataset)")
    fig1.savefig('decision_boundary.png', dpi=150, bbox_inches='tight')
    print("  - Saved: decision_boundary.png")

    # Training loss plot
    fig2 = plot_training_loss(loss_history)
    fig2.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    print("  - Saved: training_loss.png")

    # Architecture diagram (text-based)
    print("\n" + "=" * 70)
    print("QNN ARCHITECTURE")
    print("=" * 70)
    print("""
    Classical Input (2 features)
            ↓
    ┌─────────────────────┐
    │  Feature Map (RY)   │  ← Angle-encode features
    │  RY(π*x₀), RY(π*x₁)│     on qubits 0, 1
    └─────────────────────┘
            ↓
    ┌─────────────────────┐
    │ Variational Ansatz  │  ← {n_layers} × [RY gates + CNOT ladder]
    │ Parameterized RY    │     Learnable rotation angles
    │ CNOT Entanglement   │
    └─────────────────────┘
            ↓
    ┌─────────────────────┐
    │ Measurement: <Z₀>   │  ← Pauli-Z expectation on qubit 0
    └─────────────────────┘
            ↓
    Output: [-1, 1] → Binary Classification (threshold at 0)
    """)

    print("\n" + "=" * 70)
    print("QUANTUM CIRCUIT EXAMPLE")
    print("=" * 70)
    # Show a sample circuit
    sample_features = X[0]
    sample_params = trained_params
    sample_circuit = qnn._build_qnn_circuit(sample_features, sample_params)
    print(sample_circuit)

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Number of qubits: {n_qubits}")
    print(f"Number of layers: {n_layers}")
    print(f"Total parameters: {n_qubits * n_layers}")
    print(f"Training samples: {len(X)}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    print(f"Final accuracy: {accuracy:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
