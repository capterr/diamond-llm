# Diamond Engine: Asymmetric Cognitive Synthesis

**Diamond Engine** is an experimental neural architecture and attention-hook framework designed to enforce a dual-pass "Id-Ego" cognitive asymmetry within deterministic Large Language Models (specifically targeting Qwen 2.5 14B).

Unlike standard autoregressive generation which favors highly probable, safe, and often rigid outputs, the Diamond Engine intentionally induces a state of **controlled chaos** to extract highly original latent concepts, which are subsequently synthesized by a rigorous rational constraint.

## 🧠 Core Architecture: The Dual-Pass Mechanism

The engine overrides the standard transformer attention mechanism and replaces it with a two-phase cognitive process:

### 1. The Id Pass (Subconscious Perturbation)
During the first pass, the **State-Space Perturbation Kernel** is activated. The engine applies mathematical permutations (the $a/b$ matrix) directly into the model's attention weights ($W$). 
- **High Temperature ($T_{id} \approx 0.90$):** Forces the model out of local minima.
- **Goal:** Induce controlled hallucinations. The model is forced to break semantic constraints and generate highly creative, unpredictable, and often chaotic structural outputs.

### 2. The Ego Pass (Rational Synthesis)
During the second pass, the Diamond perturbation kernel is deactivated (**Diamond OFF**). The model reverts to its pure, rational state.
- **Low Temperature ($T_{ego} \approx 0.20$):** Enforces strict logical coherence.
- **Mechanism:** The model is fed the hallucinated text from the Id Pass, alongside **Internal Telemetry** (Phase space curvature, cross-layer conflict, and semantic grounding). The Ego is instructed to synthesize the Id's chaotic creativity into a mathematically and logically sound response.

## 📐 Mathematical Telemetry

The engine tracks several advanced metrics token-by-token across the transformer layers:
- **Self-Return Asymmetry ($a/b$):** Measures the permutation warping of the latent space.
- **Context Grounding ($\chi$):** Evaluates attention mass allocated strictly to the user's initial prompt.
- **Phase Space Curvature ($\nabla \times x_t$):** Measures non-linear shifts in the model's internal thought process.
- **Head Diversity (`hDiv`) & Focus:** Tracks whether attention heads are parallel-searching (high diversity) or converging onto a single semantic conclusion (high focus).

## 🚀 Usage

To integrate the Diamond Engine with your Qwen model, you must register the custom attention hook before initializing your model:

```python
from diamond_engine import DiamondConfig, DiamondEngine, chat

# Starts the interactive CLI chat with real-time layer telemetry
chat(DiamondConfig())
```

### CLI Commands (Interactive Mode)
- `/viz` - Display the layer map (Guard vs Active vs Buffer layers).
- `/health` - Check oscillation and general engine health.
- `/phase` - Display the current phase space trajectory.
- `/heads` - Monitor attention head diversity and zombie-state risks.
- `/collapse <thr>` - Adjust the Id/Ego collapse threshold dynamically.

## 📊 Empirical Evidence
This repository contains raw logs and empirical evidence (`diamond.txt`) demonstrating the engine's capability to bypass standard rigid generation. In controlled A/B tests, the Diamond Engine successfully bypasses standard "safe" conceptualizations and produces deeply original combinations without suffering from degenerative loop collapse.

## ⚠️ Disclaimer
This is an experimental research framework. The injected state-space perturbations directly modify the mathematical stability of the LLM. Heavy oscillations, "zombie states", and extreme hallucinations are expected phenomena during the Id generation phase.
