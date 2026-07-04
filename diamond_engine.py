"""
Diamond Engine: Autonomous Cognitive Architecture for LLMs
==========================================================
Author: Furkan Elmas — 2026

The Diamond Engine introduces a dynamic, state-space intervention framework 
designed to disrupt auto-regressive deterministic collapse (the "zombie state") 
in Large Language Models. By continuously tracking the L2-norm of attention 
logits (x_t) and their layer-wise gradients (nabla_t) within a specialized 
phase space, the engine computes a Master Divisor. This divisor injects 
stochastic resonance and phase-curvature-driven scaling into the attention 
mechanism (Id Pass), which is subsequently synthesized by the native model 
parameters (Ego Pass).

Key Features:
- Phase Space Curvature: Adjusts injection intensity based on sharp state transitions.
- Stochastic Resonance: Breaks deterministic locking (hyper-focus) via controlled noise.
- Layer-Specific Bell Curve Alpha: Preserves syntactic layers while perturbing semantic/reasoning layers.
- Master Equation: Normalizes input tensors to prevent saturation and ensure scale-invariant gating.
- Dual-Pass Generation: Subconscious chaotic generation (Id) followed by rational synthesis (Ego).
"""

import sys, os, math, random, re, json, time, dataclasses, argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

import re
from transformers import LogitsProcessor, LogitsProcessorList

class TurkishOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, vocab_size=152064, device="cuda"):
        self.mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        bad_pattern = re.compile(r'[一-鿿㐀-䶿぀-ヿ가-힣Ѐ-ӿ؀-ۿ]')
        vocab = tokenizer.get_vocab()
        
        cprint("  [INIT] Yabancı dil (Çince/Kiril vs) kelime filtresi oluşturuluyor...", C.YELLOW)
        banned = 0
        for word, idx in vocab.items():
            if idx < vocab_size:
                try:
                    token_str = tokenizer.convert_tokens_to_string([word])
                    if bad_pattern.search(token_str):
                        self.mask[idx] = False
                        banned += 1
                except:
                    pass
        cprint(f"  [INIT] {banned} yabancı kelime engellendi.", C.GREEN)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        v = scores.size(-1)
        scores[:, ~self.mask[:v]] = -float('inf')
        return scores

import torch.nn as nn
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2.modeling_qwen2 import repeat_kv

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False


# ═══════════════════════════════════════════════════════════════════
# ANSI RENK KODLARI
# ═══════════════════════════════════════════════════════════════════

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"

def cprint(text: str, color: str = C.WHITE, bold: bool = False):
    prefix = C.BOLD if bold else ""
    print(f"{prefix}{color}{text}{C.RESET}")


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS  —  V13: Sieve of Eratosthenes (62 asal, 2→293)
# ═══════════════════════════════════════════════════════════════════

def _sieve(limit: int) -> List[int]:
    s = bytearray([1]) * (limit + 1)
    s[0] = s[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if s[i]:
            s[i*i::i] = bytearray(len(s[i*i::i]))
    return [i for i in range(2, limit + 1) if s[i]]

_PRIMES     = _sieve(300)           # 62 asal: [2, 3, 5, ..., 293]
_SR_FACTORS = [(100,1),(50,2),(25,4),(20,5),(10,10),(5,20),(4,25),(2,50)]
_STEP_FILE  = "diamond_step.txt"
_PI_SQ_12   = (math.pi**2) / 12.0

def _load_step() -> int:
    try:
        if os.path.exists(_STEP_FILE):
            return int(open(_STEP_FILE).read().strip())
    except:
        pass
    return 0

def _save_step(s: int):
    try:
        open(_STEP_FILE, "w").write(str(s))
    except:
        pass


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DiamondConfig:
    # Architecture Defaults
    n_layers:       int   = 48       # Qwen2.5-14B default layers

    # Alpha Scaling Parameters
    alpha_base:     float = 5.0
    alpha_min:      float = 0.01
    alpha_max:      float = 9.0      # Raised ceiling to allow dynamic growth driven by conflict metrics

    # Stochastic Injection Probabilities
    base_prob:      float = 0.50     # Base probability for chaotic injection
    none_prob:      float = 0.06     # Null-state triggering probability

    # Guard Zones (Unperturbed boundary layers)
    guard_ratio:    float = 0.25

    # Cosine Annealing Temperature
    temp_max:       float = 1.0
    temp_min:       float = 0.4
    temp_total:     float = 1000.0

    # Cognitive Trackers
    entropy_window: int   = 10
    tau_decay:      float = 8.0      # Spike decay coefficient
    health_window:  int   = 50

    # Master Equation Coefficients
    harmonic_coef:  float = 0.08
    lyapunov_coef:  float = 0.75     # Maximum allowable chaos accumulation rate
    resonance_thr:  float = 0.30
    conflict_coef:  float = 0.05

    # Prime Regulator & Vocabulary Normalization
    vocab_n2:       float = 19008.0  # 152064 / 8

    # Stochastic Resonance Metrics
    sr_sigma:       float = 0.0      # Set to 0.0 per user preference for deterministic baseline
    sr_entropy_thr: float = 0.05     # Threshold for post-mask low entropy
    sr_focus_thr:   float = 0.85     # Upper bound for attention hyper-focus (Zombie-State lock)

    # State Variables
    conflict_scale: float = 7.0      # Divisor to delay hyperbolic tangent saturation
    
    # Dual-Pass Thermal Dynamics (Freudian Modeling)
    id_temperature:  float = 0.90    # Subconscious generation pass (High stochasticity)
    ego_temperature: float = 0.20    # Superconscious synthesis pass (Highly rational)

    w_momentum_decay: float = 0.72   # Exponential Moving Average for weight scaling

    # Collapse Thresholds (Id/Ego Integration)
    collapse_threshold: float = 1.30
    collapse_log_file: str = "diamond_collapse_log.jsonl"
    collapse_min_overlap: float = 0.35 # Minimum token overlap requirement between ID and Ego outputs

    # Context Grounding Floor
    grounding_floor: float = 0.30    # Absolute minimum threshold for input prompt retention during chaotic injection

    # Seed
    seed:           int   = 42

    # Buffer katman oranları (4 adet)
    buffer_ratios:  List[float] = field(
        default_factory=lambda: [0.10, 0.22, 0.44, 0.66]
    )

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)
        cprint(f"  Config kaydedildi → {path}", C.GREEN)

    @classmethod
    def from_json(cls, path: str) -> "DiamondConfig":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        valid = {fld.name for fld in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ═══════════════════════════════════════════════════════════════════
# COGNITIVE ENTROPY TRACKER
# ═══════════════════════════════════════════════════════════════════

class EntropyTracker:
    """
    Tracks and analyzes rolling thermodynamic entropy equivalents of the 
    attention manifolds. Implements harmonic-weighted rolling averages, 
    trend estimation, and high-frequency oscillation detection.
    """
    def __init__(self, max_rows: int = 1000):
        self.max_rows = max_rows
        if _PANDAS_OK:
            self._df = pd.DataFrame(
                columns=["step","nabla_t","x_t","n_t","undefined"]
            )
        else:
            self._rows: List[dict] = []

    def log(self, step: int, nabla_t: float, x_t: float,
            n_t: float, undefined: bool = False):
        row = dict(step=step, nabla_t=float(nabla_t),
                   x_t=float(x_t), n_t=float(n_t), undefined=undefined)
        if _PANDAS_OK:
            row_df = pd.DataFrame([row])
            self._df = row_df if self._df.empty else \
                pd.concat([self._df, row_df], ignore_index=True)
            if len(self._df) > self.max_rows:
                self._df = self._df.tail(self.max_rows // 2).reset_index(drop=True)
        else:
            self._rows.append(row)
            if len(self._rows) > self.max_rows:
                self._rows = self._rows[-(self.max_rows // 2):]

    def _recent_vals(self, window: int) -> List[float]:
        if _PANDAS_OK:
            return list(self._df["nabla_t"].tail(window))
        return [r["nabla_t"] for r in self._rows[-window:]]

    def rolling_mean(self, window: int = 10) -> float:
        vals = self._recent_vals(window)
        if not vals:
            return 0.0
        weights = [math.log1p(i + 1) for i in range(len(vals))]
        return sum(w * v for w, v in zip(weights, vals)) / (sum(weights) + 1e-9)

    def rolling_trend(self, window: int = 10) -> float:
        vals = self._recent_vals(window)
        if len(vals) < 2:
            return 0.0
        mid    = len(vals) // 2
        first  = sum(vals[:mid]) / max(mid, 1)
        second = sum(vals[mid:]) / max(len(vals) - mid, 1)
        return math.tanh((second - first) * 5.0)

    def rolling_var(self, window: int = 10) -> float:
        vals = self._recent_vals(window)
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        return sum((v - mean)**2 for v in vals) / len(vals)

    # ── High-Frequency Oscillation ──
    def oscillation_score(self, window: int = 16) -> float:
        """
        Measures the frequency of sign changes in the entropy gradient.
        A high score indicates model instability (rapid state oscillation), 
        triggering reduced injection intensity and expanded guard zones.
        
        Returns:
            float: Normalized oscillation score in range [0, 1].
        """
        vals = self._recent_vals(window)
        if len(vals) < 4:
            return 0.0
        diffs = [vals[i+1] - vals[i] for i in range(len(vals) - 1)]
        sign_changes = sum(
            1 for i in range(len(diffs) - 1)
            if diffs[i] * diffs[i+1] < 0
        )
        return sign_changes / max(len(diffs) - 1, 1)

    def multi_window(self) -> Tuple[float, float, float]:
        """Calculates short (3), medium (10), and long (30) term rolling entropy averages."""
        return (
            self.rolling_mean(3),
            self.rolling_mean(10),
            self.rolling_mean(30),
        )

    def tail(self, n: int = 10):
        if _PANDAS_OK:
            return self._df.tail(n)
        return self._rows[-n:]

    def __len__(self) -> int:
        if _PANDAS_OK:
            return len(self._df)
        return len(self._rows)


# ═══════════════════════════════════════════════════════════════════
# CONFLICT DETECTOR
# ═══════════════════════════════════════════════════════════════════

class ConflictDetector:
    """
    Computes pairwise gradient conflicts across specified boundary layers.
    Includes cascade amplification logic and cross-layer entanglement metrics.
    Calculates 6-pair combinations for a 4-layer buffer setup.
    """
    def __init__(self, n_layers: int, ratios: List[float], scale: float = 3.0):
        self.buffer_layers: List[int] = [
            max(0, min(n_layers - 1, int(r * n_layers)))
            for r in ratios
        ]
        self.buffers: Dict[int, float] = {}
        self.scale = scale  # Saturation delay scaling factor before tanh activation

    def update(self, layer_idx: int, nabla_t: float):
        if layer_idx in self.buffer_layers:
            self.buffers[layer_idx] = nabla_t

    def compute_conflict(self) -> float:
        vals = [(l, self.buffers[l]) for l in self.buffer_layers
                if l in self.buffers]
        if len(vals) < 2:
            return 0.0
        conflicts = [abs(vals[i+1][1] - vals[i][1])
                     for i in range(len(vals) - 1)]
        cascade = 1.0
        for i in range(len(conflicts) - 1):
            if conflicts[i] > 0.1 and conflicts[i+1] > 0.1:
                # Dynamic cascade amplification capped to prevent immediate tanh saturation
                cascade = min(cascade * 1.15, 1.4)
        return math.tanh(sum(conflicts) / len(conflicts) * cascade / self.scale)

    def resonance_score(self, threshold: float = 0.3) -> float:
        vals = list(self.buffers.values())
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        var  = sum((v - mean)**2 for v in vals) / len(vals)
        res  = math.exp(-var * 5.0)
        return res if mean > threshold else 0.0

    def lhs_extra(self) -> float:
        return self.compute_conflict() * 0.1

    def cross_layer_entanglement(self) -> float:
        """
        Calculates the aggregate entanglement across all buffer layer pairs.
        Applies a distance-weighted penalty (distant layers have weaker entanglement).
        """
        buf_vals = [(l, self.buffers[l])
                    for l in self.buffer_layers if l in self.buffers]
        bonus = 0.0
        for i in range(len(buf_vals)):
            for j in range(i + 1, len(buf_vals)):
                li, vi = buf_vals[i]
                lj, vj = buf_vals[j]
                pair_bonus  = abs(vi * vj) * 0.05
                dist_weight = 1.0 / max(abs(lj - li), 1)
                bonus += pair_bonus * dist_weight
        return min(bonus, 0.5)

    def state_str(self) -> str:
        parts = []
        for l in self.buffer_layers:
            v = self.buffers.get(l)
            parts.append(f"L{l}={v:.3f}" if v is not None else f"L{l}=–")
        return "  ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# ATTENTION HEAD CONFLICT DETECTOR
# ═══════════════════════════════════════════════════════════════════

class HeadConflictDetector:
    """
    Computes per-head entropy, attention divergence (head diversity), and 
    hyper-focus metrics across the attention manifold.

    Metrics:
    - head_diversity: Measures divergence in attention distributions across different heads.
                      High diversity amplifies the state-space entanglement signal.
    - focus_score: Mean of the maximum attention weights. Values > 0.65 indicate 
                   autoregressive lock (hyper-focus / deterministic collapse risk).
    """
    def __init__(self, window: int = 20):
        self.window              = window
        self._div_history: List[float] = []
        self.current_diversity   = 0.0
        self.current_focus       = 0.0
        self.last_probs = None  # V17: son mask'lı softmax cache — grounding bunu reuse eder

    def compute(self, w: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
                ) -> Tuple[float, float, float]:
        """
        Returns: (mean_entropy, head_diversity, focus_score)

        w shape: (batch, heads, seq_q, seq_k)
        """
        with torch.no_grad():
            wf = w.float()
            if attention_mask is not None:
                wf = wf + attention_mask.float()
            p = torch.softmax(wf, dim=-1)  # (B, H, Sq, Sk)
            self.last_probs = p  # Cache masked softmax for subsequent grounding calculations

            # Per-head entropy: sum over keys, mean over batch+seq_q
            h_ent      = -(p * torch.log(p + 1e-9)).sum(-1)   # (B, H, Sq)
            per_head   = h_ent.mean(dim=(0, 2))                 # (H,)

            mean_ent   = float(per_head.mean().clamp(0.0, 5.0))
            diversity  = float(per_head.std().clamp(0.0, 2.0)) \
                         if per_head.numel() > 1 else 0.0

            # Focus: Mean of the maximum attention weight per sequence element
            focus      = float(p.max(-1).values.mean().clamp(0.0, 1.0))

        self.current_diversity = diversity
        self.current_focus     = focus
        self._div_history.append(diversity)
        if len(self._div_history) > self.window:
            self._div_history = self._div_history[-self.window:]

        return mean_ent, diversity, focus

    def rolling_diversity(self) -> float:
        if not self._div_history:
            return 0.0
        return sum(self._div_history) / len(self._div_history)

    def is_zombie(self, entropy_thr: float, focus_thr: float) -> bool:
        """
        Detects if the model is entering a deterministic 'zombie' state, characterized 
        by extreme certainty or hyper-focus on a single token manifold.
        """
        return self.current_focus > focus_thr or \
               (self.current_diversity < 0.05 and self.current_focus > 0.50)


# ═══════════════════════════════════════════════════════════════════
# PHASE SPACE TRACKER
# ═══════════════════════════════════════════════════════════════════

class PhaseSpaceTracker:
    """
    Tracks orbital trajectories of the generation state within the (∇t, x_t) phase space.

    - curvature(): Measures sharp state transitions. High curvature amplifies Diamond intervention.
    - velocity(): Euclidean rate of state change across generation steps.
    """
    def __init__(self, window: int = 12):
        self.window   = window
        self._history: List[Tuple[float, float]] = []

    def update(self, nabla_t: float, x_t: float):
        self._history.append((nabla_t, x_t))
        if len(self._history) > self.window:
            self._history = self._history[-self.window:]

    def curvature(self) -> float:
        """
        Computes the normalized intersection curvature of the last three phase points.
        Returns a value in [0, 1] mapped via hyperbolic tangent.
        """
        if len(self._history) < 3:
            return 0.0
        p1, p2, p3 = self._history[-3], self._history[-2], self._history[-1]
        v1 = (p2[0]-p1[0], p2[1]-p1[1])
        v2 = (p3[0]-p2[0], p3[1]-p2[1])
        cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
        mag   = math.sqrt(v1[0]**2+v1[1]**2) * math.sqrt(v2[0]**2+v2[1]**2)
        return math.tanh(cross / (mag + 1e-9))

    def curvature_raw(self) -> Tuple[float, float]:
        """
        Returns raw cross product and magnitude of the phase curvature.
        Used for telemetry and debugging saturation issues in the phase space scaling.
        """
        if len(self._history) < 3:
            return 0.0, 0.0
        p1, p2, p3 = self._history[-3], self._history[-2], self._history[-1]
        v1 = (p2[0]-p1[0], p2[1]-p1[1])
        v2 = (p3[0]-p2[0], p3[1]-p2[1])
        cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
        mag   = math.sqrt(v1[0]**2+v1[1]**2) * math.sqrt(v2[0]**2+v2[1]**2)
        return cross, mag

    def velocity(self) -> float:
        """Computes the Euclidean distance between the two most recent phase points."""
        if len(self._history) < 2:
            return 0.0
        p1, p2 = self._history[-2], self._history[-1]
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

    def state_str(self) -> str:
        if not self._history:
            return "(–, –)"
        n, x = self._history[-1]
        return (f"∇={n:.3f}  x={x:.3f}  "
                f"curv={self.curvature():.3f}  vel={self.velocity():.3f}")


# ═══════════════════════════════════════════════════════════════════
# SPIKE PROCESSOR (PROMPT COMPLEXITY ANALYSIS)
# ═══════════════════════════════════════════════════════════════════

class SpikeProcessor:
    """
    Computes a multi-dimensional complexity scalar ("spike") from the input prompt.
    Features an adaptive exponential decay (tau) modulated by topic shift.
    """
    def __init__(self, tau: float = 8.0):
        self.tau_base         = tau
        self.tau              = tau
        self.current_spike    = 0.0
        self.decay_counter    = 0.0
        self._history_spike   = 0.0

    def adaptive_tau(self, topic_shift: float) -> float:
        """
        Modulates the spike decay rate based on conversation coherence.
        Topic shifts accelerate decay to highlight the new stimulus, while 
        topic continuity slows decay to allow cumulative spike accumulation.
        topic_shift: 0 = identical topic, 1 = completely orthogonal topic.
        """
        self.tau = self.tau_base * (1.0 - topic_shift * 0.45)
        return self.tau

    def compute_spike(self, text: str, history_depth: int = 0) -> float:
        if not text.strip():
            self.current_spike = 0.0
            return 0.0

        words = text.split()
        n     = max(len(words), 1)

        # 1. Character Diversity
        char_ratio = len(set(text.lower())) / max(len(text), 1)

        # 2. Word Length Variance
        lengths  = [len(w) for w in words]
        mean_l   = sum(lengths) / n
        var_l    = sum((l - mean_l)**2 for l in lengths) / n
        word_var = math.sqrt(var_l) / (mean_l + 1e-9)

        # 3. Bigram Diversity
        bigrams = {(words[i].lower(), words[i+1].lower())
                   for i in range(len(words)-1)}
        bigram_ratio = len(bigrams) / max(n - 1, 1)

        # 4. Punctuation Density
        punct_count   = sum(1 for c in text if c in '.,!?;:')
        punct_density = punct_count / max(len(text), 1)

        # 5. Question Detection (Cross-Lingual)
        _TR_Q = {"ne","neden","nasıl","nerede","kim","kaç","hangi","ne zaman"}
        _EN_Q = {"what","why","how","where","who","which","when"}
        is_question = (
            text.strip().endswith('?') or
            any(w.lower() in _TR_Q | _EN_Q for w in words[:4])
        )
        question_boost = 0.25 if is_question else 0.0

        # 6. Orthographic Emphasis (Capitalization)
        upper_ratio    = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        emphasis_boost = math.tanh(upper_ratio * 10.0) * 0.2

        # 7. Sequence Length Factor
        length_factor = math.tanh(n / 25.0)

        # 8. Conversation Depth Accumulation
        depth_factor = math.tanh(history_depth / 10.0) * 0.15

        raw = (
            char_ratio    * 2.0 +
            word_var      * 1.5 +
            bigram_ratio  * 1.0 +
            punct_density * 3.0 +
            question_boost      +
            emphasis_boost      +
            length_factor * 0.5 +
            depth_factor
        )

        self.current_spike  = math.tanh(raw / 7.0)
        self.decay_counter  = 0.0
        self._history_spike = self._history_spike * 0.8 + self.current_spike * 0.2
        return self.current_spike

    def get_decayed(self) -> float:
        spike = self.current_spike * math.exp(-self.decay_counter / (self.tau + 1e-9))
        self.decay_counter += 1.0
        return spike

    def accumulated(self) -> float:
        return self._history_spike


# ═══════════════════════════════════════════════════════════════════
# SYSTEM HEALTH MONITOR
# ═══════════════════════════════════════════════════════════════════

class HealthMonitor:
    def __init__(self, window: int = 50):
        self.window   = window
        self._scales: List[float] = []
        self._gates:  List[float] = []

    def record(self, scale: float, gate: float):
        self._scales.append(abs(scale))
        self._gates.append(float(gate))
        if len(self._scales) > self.window:
            self._scales = self._scales[-self.window:]
            self._gates  = self._gates[-self.window:]

    @property
    def health_score(self) -> float:
        if not self._scales:
            return 1.0
        mean_s = sum(self._scales) / len(self._scales)
        mean_g = sum(self._gates)  / len(self._gates)
        # Normalize scale health against standard Qwen generation magnitudes
        scale_health = max(0.0, 1.0 - mean_s / 2.5)
        gate_health  = max(0.0, 1.0 - abs(mean_g - 0.5) * 2.0)
        return scale_health * 0.6 + gate_health * 0.4

    @property
    def is_alert(self) -> bool:
        return self.health_score < 0.35

    def status_str(self) -> str:
        h = self.health_score
        if h > 0.70: return f"{C.GREEN}SAĞLIKLI ({h:.0%}){C.RESET}"
        if h > 0.40: return f"{C.YELLOW}ORTA ({h:.0%}){C.RESET}"
        return f"{C.RED}UYARI ({h:.0%}){C.RESET}"


# ═══════════════════════════════════════════════════════════════════
# ADAPTIVE GUARD REGULATOR
# ═══════════════════════════════════════════════════════════════════

class AdaptiveGuard:
    """
    Dynamically resizes protective boundary layers (where injection is disabled).
    Scales aggressively under conditions of poor health or high oscillation 
    to preserve foundational syntax and core semantics.
    """
    def __init__(self, n_layers: int, base_ratio: float = 0.25):
        self.n_layers    = n_layers
        self.base_ratio  = base_ratio
        self._current    = max(4, int(n_layers * base_ratio))

    def adapt(self, health: float, oscillation: float = 0.0):
        ratio = self.base_ratio + (1.0 - health) * 0.12 + oscillation * 0.08
        ratio = min(ratio, 0.46)
        self._current = max(4, int(self.n_layers * ratio))

    def is_protected(self, layer_idx: int) -> bool:
        return layer_idx < self._current or \
               layer_idx >= self.n_layers - self._current

    @property
    def size(self) -> int:
        return self._current


# ═══════════════════════════════════════════════════════════════════
# INTENT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════

class IntentClassifier:
    """
    Classifies the rigidness of the prompt to modulate baseline chaos (Alpha).
    Scores range from 0.10 (Strict Mathematics/Logic) to 1.0 (Casual Conversation).
    """
    _MATH_KEYWORDS = {
        "topla","çarp","böl","çıkar","hesapla","integral","türev","matris",
        "vektör","olasılık","istatistik","limit","türev","fonksiyon","denklem",
        "calculate","solve","integral","derivative","matrix","probability",
        "equation","compute","factorial","modulo","logarithm","sqrt","prime",
    }
    _LOGIC_KEYWORDS = {
        "eğer","ise","değilse","yani","dolayısıyla","ispat","kanıtla","mantık",
        "doğruysa","yanlışsa","hepsi","hiçbiri","bazıları","ise","mıdır","midir",
        "if","then","therefore","because","prove","logic","all","none","implies",
    }
    _MATH_RE = re.compile(
        r"(\d+\s*[\+\-\*\/\^×x]\s*\d+|\d+\^\d+|sqrt|karekök|\d+\s*[xX]\s*\d+)"
    )

    def classify(self, text: str) -> float:
        if not text.strip():
            return 1.0
        tl    = text.lower()
        words = set(tl.split())
        if self._MATH_RE.search(tl):
            return 0.10
        math_hits  = sum(1 for k in self._MATH_KEYWORDS  if k in words)
        logic_hits = sum(1 for k in self._LOGIC_KEYWORDS if k in words)
        if math_hits  >= 2: return 0.12
        if math_hits  == 1: return 0.20
        if logic_hits >= 2: return 0.28
        if logic_hits == 1: return 0.45
        if text.strip().endswith('?') and len(words) <= 8:
            return 0.50
        return 1.0


# ═══════════════════════════════════════════════════════════════════
# ALPHA AUTO-CALIBRATOR
# ═══════════════════════════════════════════════════════════════════

class AlphaAutoCalibrator:
    """
    Autonomously calibrates the baseline chaos multiplier (Alpha) based on 
    the macroscopic health score of the generation manifold.
    """
    def __init__(self, cfg: "DiamondConfig"):
        self.cfg       = cfg
        self._original = cfg.alpha_base

    def calibrate(self, health: float):
        if health < 0.20:
            self.cfg.alpha_base = max(self.cfg.alpha_min * 2,
                                      self.cfg.alpha_base * 0.80)
        elif health < 0.40:
            self.cfg.alpha_base = max(self.cfg.alpha_min * 2,
                                      self.cfg.alpha_base * 0.93)
        elif health > 0.70:
            target = min(self._original, self.cfg.alpha_max * 0.6)
            self.cfg.alpha_base = min(target, self.cfg.alpha_base * 1.03)


# ═══════════════════════════════════════════════════════════════════
# CROSS-TURN COHERENCE TRACKER
# ═══════════════════════════════════════════════════════════════════

class CoherenceTracker:
    """
    Computes cross-turn conversational coherence utilizing TF-IDF pseudo-vectors.
    Detects semantic shifts to reset or amplify exponential decay mechanisms.
    """
    def __init__(self, window: int = 5):
        self.window              = window
        self.coherence_score     = 1.0
        self.topic_shift         = 0.0
        self.conversation_embedding: Dict[str, float] = {}

    def _embed(self, text: str) -> Dict[str, float]:
        _STOP = {"bir","ve","bu","da","de","mi","mu","mı","mü","ki",
                 "ile","ben","sen","o","the","a","is","are","in","on",
                 "için","gibi","kadar","ama","fakat","lakin","veya","ya"}
        words = [w for w in re.findall(r'\w+', text.lower())
                 if w not in _STOP and len(w) > 2]
        vec = {}
        for w in words:
            vec[w] = vec.get(w, 0.0) + 1.0
        norm = math.sqrt(sum(v*v for v in vec.values())) + 1e-9
        return {k: v/norm for k, v in vec.items()}

    def _cosine(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        return sum(v1.get(k, 0.0) * v for k, v in v2.items())

    def update(self, text: str):
        cur = self._embed(text)
        if not self.conversation_embedding:
            self.conversation_embedding = cur
            self.coherence_score = 1.0
            self.topic_shift     = 0.0
            return
        sim = self._cosine(self.conversation_embedding, cur)
        self.coherence_score = sim
        self.topic_shift     = math.exp(-sim * 4.0)
        all_keys = set(self.conversation_embedding) | set(cur)
        new_emb  = {}
        for k in all_keys:
            v = self.conversation_embedding.get(k,0.0)*0.7 + cur.get(k,0.0)*0.3
            if v > 0.02:
                new_emb[k] = v
        norm = math.sqrt(sum(v*v for v in new_emb.values())) + 1e-9
        self.conversation_embedding = {k: v/norm for k, v in new_emb.items()}

    def modulate_spike(self, spike: float) -> float:
        return min(spike * (1.0 + self.topic_shift * 0.60), 1.0)


# ═══════════════════════════════════════════════════════════════════
# DIAMOND ENGINE CORE
# ═══════════════════════════════════════════════════════════════════

class DiamondEngine:
    """
    The central coordinator for state-space observation and perturbation.
    Manages all cognitive trackers and applies the Master Divisor equation 
    during auto-regressive generation.
    """
    def __init__(self, cfg: Optional[DiamondConfig] = None):
        self.cfg   = cfg or DiamondConfig()
        self.alpha = self.cfg.alpha_base

        # Core Modules
        self.entropy   = EntropyTracker()
        self.conflict  = ConflictDetector(self.cfg.n_layers, self.cfg.buffer_ratios, self.cfg.conflict_scale)
        self.spike     = SpikeProcessor(tau=self.cfg.tau_decay)
        self.guard     = AdaptiveGuard(self.cfg.n_layers, self.cfg.guard_ratio)
        self.health    = HealthMonitor(self.cfg.health_window)
        self.intent    = IntentClassifier()
        self.calibr    = AlphaAutoCalibrator(self.cfg)
        self.coherence = CoherenceTracker()

        # High-Fidelity Phase Modules
        self.head_conflict = HeadConflictDetector()
        self.phase         = PhaseSpaceTracker()

        # Intent EMA
        self.intent_multiplier: float = 1.0

        # Exponential Moving Average for Scaling Momentum
        self._w_momentum = 0.0

        # Head Diversity Cache
        self._last_head_div: float = 0.0
        self._last_focus:    float = 0.0

        # State
        self.step        = _load_step()
        self.call_count  = 0
        self.fire_count  = 0
        self.token_turn  = 0
        self.debug_mode  = False

        # Time Gate
        self.t_eq      = float(self.step)
        self.t_model   = float(self.step)
        self.tg_frozen = False
        self.tg_bias   = 0.0

        # Layer Memory
        self.layer_memory: Dict[int, float] = {}
        self._prev_w_scale = 0.0

        # Context Grounding Baseline
        self.prompt_len: Optional[int] = None
        self._last_grounding: float = 1.0

        # Id Pass Telemetry Buffer
        self._id_tele = self._empty_telemetry()

        # Token-Level Aggregation Buffers
        self._tok_nabla: List[float] = []
        self._tok_xt:    List[float] = []
        self._tok_nt:    List[float] = []

    @staticmethod
    def _empty_telemetry() -> dict:
        return {"n": 0, "nabla": 0.0, "conflict": 0.0, "resonance": 0.0,
                "grounding": 0.0, "sr_a": 0.0, "sr_b": 0.0}

    def reset_id_telemetry(self):
        self._id_tele = self._empty_telemetry()

    def record_id_telemetry(self, nabla_t: float, conflict: float,
                             resonance: float, grounding: float,
                             a: float, b: float):
        t = self._id_tele
        t["n"] += 1
        t["nabla"]     += nabla_t
        t["conflict"]  += conflict
        t["resonance"] += resonance
        t["grounding"] += grounding
        t["sr_a"]      += a
        t["sr_b"]      += b

    def get_id_telemetry_avg(self) -> dict:
        t = self._id_tele
        n = max(t["n"], 1)
        sr_ratio = (t["sr_a"] / max(t["sr_b"], 1e-9)) if t["n"] else 1.0
        return {
            "n":         t["n"],
            "nabla":     t["nabla"]     / n,
            "conflict":  t["conflict"]  / n,
            "resonance": t["resonance"] / n,
            "grounding": t["grounding"] / n,
            "sr_ratio":  sr_ratio,
        }

    # ── Dynamic Alpha Modulation ──
    def update_alpha(self, conflict: float):
        ent_var     = self.entropy.rolling_var(self.cfg.entropy_window)
        chaos_brake = 1.0 - min(ent_var * 5.0, 0.30)
        effective   = self.cfg.alpha_base * self.intent_multiplier * chaos_brake
        conf_effect = conflict * (1.5 * self.cfg.conflict_coef / 0.05) * self.intent_multiplier
        self.alpha  = effective + conf_effect
        self.alpha  = max(self.cfg.alpha_min, min(self.alpha, self.cfg.alpha_max))

    # ── Layer-Specific Alpha Distribution ──
    def layer_alpha(self, layer_idx: int) -> float:
        """
        Applies a Bell Curve distribution across network depth:
        Mid-layers (semantic/reasoning) receive 1.5x intensity, while boundary 
        layers (syntax/grammar) are suppressed to 0.6x to maintain grammatical coherence.
        """
        g      = self.guard.size
        active = max(self.cfg.n_layers - 2*g, 1)
        pos    = max(0, layer_idx - g)
        t      = pos / max(active - 1, 1)
        # Bell Curve: peak=1.5x at t=0.5, trough=0.6x at t=0,1
        weight = 0.6 + 3.6 * t * (1.0 - t)
        return self.alpha * max(0.3, weight)

    # ── Harmonic Term ──
    def harmonic_term(self, nabla_t: float, layer_idx: int) -> float:
        L = max(self.cfg.n_layers - 1, 1)
        h = sum(math.sin(k * math.pi * layer_idx / L) / k for k in range(1, 5))
        return math.tanh(h * nabla_t * self.cfg.harmonic_coef)

    # ── Lyapunov Stability ──
    def lyapunov_term(self, w_scale: float) -> float:
        return -self.cfg.lyapunov_coef * w_scale * abs(w_scale)

    # ── Temporal Shift: δ(t) ──
    def delta_t(self, layer_idx: int) -> float:
        T  = max(self.cfg.n_layers, 1)
        t0 = T / 4.0
        omega = 2 * math.pi / T
        raw = 64.0 * math.cos(omega * layer_idx) * (layer_idx / (t0 + 1e-9))
        return math.tanh(raw / (T * 16.0 + 1e-9))

    # ── Pole Scaling: Y(s) ──
    def pole_scale(self, layer_idx: int) -> float:
        t0    = self.cfg.n_layers / 4.0
        omega = 2 * math.pi / self.cfg.n_layers
        disc  = max((32.0/t0)**2 - omega**2, 0)
        s1 = 32.0/t0 + math.sqrt(disc)
        s2 = 32.0/t0 - math.sqrt(disc)
        t  = layer_idx / max(self.cfg.n_layers - 1, 1)
        raw = s1*t + s2*(1-t)
        return math.tanh(raw / (64.0/t0 + 1.0))

    # ── Composite Series: Σ 1/2i² ──
    def composite_inj(self) -> float:
        k   = (self.step % 8) + 1
        raw = sum(1.0/(2*i*i) for i in range(1, k+1))
        return min(raw, _PI_SQ_12)

    # ── Prime Modulo Regulator ──
    def prime_regulator(self) -> float:
        """
        Applies a modulo regulator based on the Sieve of Eratosthenes (62 primes up to 293).
        Formula: reg(n², p) = (80·n²·3) / p²
        Extends the cycle length to prevent mechanical repetition, ensuring organic chaos.
        """
        p      = _PRIMES[self.step % len(_PRIMES)]
        n2     = self.cfg.vocab_n2
        reg    = (80.0 * n2 * 3.0) / (p * p)
        # Normalize to [0, 1]
        reg_norm = math.log1p(reg) / math.log1p(80.0 * 6282.0 * 3.0 / 4.0)
        return 0.85 + 0.30 * reg_norm   # Output Range: [0.85, 1.15]

    # ── Smooth Permutation Symmetry (Cosine Interpolated) ──
    def smooth_self_return(self) -> Tuple:
        first_buf = self.conflict.buffers.get(self.conflict.buffer_layers[0], None)
        if first_buf is not None and first_buf == 0.0:
            return None, None, None, None
        idx1 = self.step % len(_SR_FACTORS)
        idx2 = (self.step + 1) % len(_SR_FACTORS)
        a1, b1 = _SR_FACTORS[idx1]
        a2, b2 = _SR_FACTORS[idx2]
        frac  = (self.step % 10) / 10.0
        blend = 0.5 * (1.0 - math.cos(math.pi * frac))
        a = a1*(1-blend) + a2*blend
        b = b1*(1-blend) + b2*blend
        
        d1 = math.tanh((a - b) / (a + b + 1e-9))
        d2 = math.log1p(b/a) if a > 1e-9 else 0.0
        return d1, d2, a, b

    # ── Cosine Temperature Annealing ──
    def temperature(self) -> float:
        prog  = min(self.step / self.cfg.temp_total, 1.0)
        cos_t = 0.5 * (1.0 + math.cos(math.pi * prog))
        return self.cfg.temp_min + (self.cfg.temp_max - self.cfg.temp_min) * cos_t

    # ── The Master Divisor Equation ──
    def master_equation(self, nabla_t: float, x_t: float,
                        n_t: float, lhs_extra: float = 0.0):
        """
        The fundamental state-space bounding equation. 
        Input variables (nabla_t, x_t) are independently normalized relative to their 
        empirical ceilings to prevent scale mismatches and tanh saturation.
        Computes the Master Gate scalar applied directly to attention distributions.
        """
        x_n     = min(x_t / 30.0, 1.5)
        nabla_n = min(nabla_t / 4.0, 1.5)
        
        E23=2.3; E33=3.3; E4=4.0; E3=3.0
        nab33 = max(nabla_n**E33, 1e-9)
        nab23 = max(nabla_n**E23, 1e-9)
        nab4  = max(nabla_n**E4,  1e-9)
        nab3  = max(nabla_n**E3,  1e-9)
        x33   = max(x_n**E33, 1e-9)
        x23   = max(x_n**E23, 1e-9)
        n25   = max(n_t**2.5, 1e-9)
        n23   = max(n_t**0.23, 1e-9)
        
        reg   = 80.0 - min(1.0/x23, 79.9)
        b_lhs = nab4 * x23 * reg * x33
        b_rhs = nab3 * n25
        
        # Logarithmic domain transition preserves relative ratios against exponential inflation
        log_lhs = math.log1p(b_lhs)
        log_rhs = math.log1p(b_rhs)
        b_norm  = math.tanh((log_lhs - log_rhs) / (abs(log_lhs) + abs(log_rhs) + 1e-9))
        
        lhs   = nab33 - x33 - nab23 - b_norm*10.0 + lhs_extra
        rhs   = (10.0*x33)/n23 - x33/nab3
        raw   = lhs - rhs
        
        # Output bounded mapping
        scalar  = math.tanh(raw / 14.0)
        gate    = (1.0 + scalar) / 2.0
        divisor = 1.0 + 0.2 * scalar
        return gate, divisor, scalar

    # ── Ana Zincir ──
    def diamond_master_divisor(self, nabla_t: float, x_t: float,
                             n_t: float, layer_idx: int):
        # Conflict & resonance
        conflict  = self.conflict.compute_conflict()
        resonance = self.conflict.resonance_score(self.cfg.resonance_thr)
        lhs_extra = self.conflict.lhs_extra()

        # Dynamic alpha + guard adapt
        self.update_alpha(conflict)
        osc = self.entropy.oscillation_score()
        self.guard.adapt(self.health.health_score, osc)

        # Smooth permutation
        perm = self.smooth_self_return()
        if perm[0] is None:
            return None
        d1, d2, a, b = perm

      # ── Stochastic Resonance (Deterministic Anti-Lock) ──
    def apply_stochastic_resonance(self, w: torch.Tensor, nabla_t: float) -> torch.Tensor:
        """
        Injects low-amplitude Gaussian noise into the attention manifold if the 
        model exhibits hyper-focus or extreme certainty (low ∇). 
        This disrupts rigid auto-regressive collapse, forcing exploring alternate latent paths.
        (Inactive if cfg.sr_sigma = 0.0).
        """
        if self.cfg.sr_sigma <= 0.0:
            return w
        
        # Determine zombie state locking using HeadConflict metrics
        is_zombie = self.head_conflict.is_zombie(self.cfg.sr_entropy_thr, self.cfg.sr_focus_thr)
        
        if nabla_t < self.cfg.resonance_thr or is_zombie:
            noise = torch.randn_like(w) * self.cfg.sr_sigma
            # Apply dynamic scaling based on phase velocity (sharper change -> stronger noise)
            vel   = self.phase.velocity()
            scale = 1.0 + math.tanh(vel) * 0.5
            w = w + noise * scale
            self._id_tele["resonance"] += 1.0
        return w

        # Rolling entropy + trend
        rolling_ent = self.entropy.rolling_mean(self.cfg.entropy_window)
        trend       = self.entropy.rolling_trend(self.cfg.entropy_window)
        TARGET      = 580.0 * (1.0 + rolling_ent + trend * 0.1)

        # Master denklem
        gate, divisor, scalar = self.master_equation(nabla_t, x_t, n_t, lhs_extra)

        # Kompozit enjeksiyon
        comp     = self.composite_inj()
        r_fwd    = d1 / (d2 + 1e-9)
        r_inv    = d2 / (d1 + 1e-9)
        x23      = max(x_t**2.3, 1e-9)
        reactive = math.tanh(TARGET / (x23 * 1000.0))

        final_scale = (
            scalar * (1.0 + comp*0.12) +
            math.tanh(r_fwd - r_inv) * 0.06
        )

        # δ(t) + Y(s)
        delta_contrib = self.delta_t(layer_idx)
        pole_contrib  = self.pole_scale(layer_idx)
        inj_w  = self.inject_weight(layer_idx)
        temp   = self.temperature()
        t_gate = 1.0 / (temp + 0.01)

        t_ratio = (layer_idx - self.guard.size) / max(
            self.cfg.n_layers - 2*self.guard.size, 1
        )
        t_ratio = max(0.0, min(t_ratio, 1.0))
        delta_w = 0.15 + 0.10*(1.0-t_ratio)
        pole_w  = 0.15 + 0.10*t_ratio

        # Cross-layer entanglement (6 çift)
        inter_bonus = self.conflict.cross_layer_entanglement()

        # Spike katkıları (çarpan olarak)
        spike_contrib = self.spike.get_decayed() * 0.08
        acc_spike     = self.spike.accumulated() * 0.04

        # Harmonik katkı
        harmonic_contrib = self.harmonic_term(nabla_t, layer_idx)

        # Resonance → damper (V12 felsefesi: bölen olarak kullan)
        resonance_damp = 1.0 + resonance * 0.25

        # V13 YENİ: Phase curvature boost
        # Keskin durum değişimi → daha güçlü injection
        phase_curv    = self.phase.curvature()
        phase_boost   = 1.0 + phase_curv * 0.20

        # V13 YENİ: Head diversity boost
        # Kafalar birbirinden ayrışıyorsa → daha güçlü injection
        head_div_boost = 1.0 + self._last_head_div * 0.15

        # V13 YENİ: Oscillation damping
        # Model salınıyorsa → injection zayıflat
        osc_damp = 1.0 / (1.0 + osc * 0.40)

        final = (
            final_scale * 0.50 * inj_w * gate +
            delta_contrib * delta_w            +
            pole_contrib  * pole_w             +
            reactive      * 0.20 * (1.0-gate) +
            harmonic_contrib
        ) * t_gate

        # Cross-layer inter_bonus: frenleme
        if inter_bonus > 0:
            final = final / (1.0 + inter_bonus)

        # Spike çarpanı
        final = final * (1.0 + spike_contrib + acc_spike)

        # Resonance damping
        final = final / resonance_damp

        # V13: Phase curvature + head diversity boost
        final = final * phase_boost * head_div_boost

        # V13: Oscillation damping
        final = final * osc_damp

        # Lyapunov stabilizasyon
        final += self.lyapunov_term(final)

        # w_scale = 3.0 * tanh(final) — V12 formülü
        w_scale = 3.0 * math.tanh(final)

        # Divisor uygulaması (V12: artık kullanılıyor!)
        w_scale = w_scale / max(divisor, 0.5)

        # Prime Regulator (gelişmiş, 62 asal)
        w_scale = w_scale * self.prime_regulator()

        # ∅₀ none tetikleyici
        if self.should_none() and nabla_t > 1.5:
            w_scale *= 0.5

        # V13 YENİ: W-Scale Momentum (EMA pürüzsüzleştirme)
        self._w_momentum = self._w_momentum * self.cfg.w_momentum_decay + \
                           w_scale * (1.0 - self.cfg.w_momentum_decay)
        w_scale = w_scale * 0.75 + self._w_momentum * 0.25

        self._prev_w_scale = w_scale
        self.health.record(w_scale, gate)

        return w_scale, gate, divisor, scalar, a, b, temp, conflict, resonance

    # ── Stochastic Resonance ──
    def apply_stochastic_resonance(self, w: torch.Tensor,
                                   layer_idx: int) -> torch.Tensor:
        """
        Anti-Zombie Mechanism.
        If the model is hyper-focused or has very low ∇, inject controlled Gaussian noise.
        Only active in mid-layers (guard zones are protected).
        """
        if self.guard.is_protected(layer_idx):
            return w

        entropy_need = max(0.0, self.cfg.sr_entropy_thr - self._last_focus) / \
                       max(self.cfg.sr_entropy_thr, 1e-9)
        focus_excess = max(0.0, self._last_focus - self.cfg.sr_focus_thr) / \
                       max(1.0 - self.cfg.sr_focus_thr, 1e-9)
        need         = max(entropy_need * 0.5, focus_excess)

        if need < 0.05:
            return w

        # Bell weighting: orta katmanlarda güçlü
        t       = layer_idx / max(self.cfg.n_layers - 1, 1)
        layer_w = 4.0 * t * (1.0 - t)
        sigma   = self.cfg.sr_sigma * need * layer_w

        noise = torch.randn_like(w) * sigma
        return w + noise

    # ── Inject Weight ──
    def inject_weight(self, layer_idx: int) -> float:
        P   = max(self.cfg.n_layers // 4, 1)
        env = abs(math.sin(math.pi * layer_idx / P))
        return max(env * random.Random(self.cfg.seed ^ layer_idx).uniform(0.6, 1.4), 0.1)

    # ── SPI  —  V13: Head diversity boost eklendi ──
    def should_inject(self, layer_idx: int, nabla_t: Optional[float] = None) -> bool:
        self.call_count += 1
        if self.guard.is_protected(layer_idx):
            return False

        t    = layer_idx / max(self.cfg.n_layers - 1, 1)
        prob = self.cfg.base_prob * 4.0 * t * (1.0 - t)

        # Logic zone boost
        ls = int(self.cfg.n_layers * 0.30)
        le = int(self.cfg.n_layers * 0.70)
        if ls <= layer_idx < le:
            prob = min(prob * 1.8, 0.85)

        # Layer memory delta
        if nabla_t is not None and layer_idx in self.layer_memory:
            if abs(nabla_t - self.layer_memory[layer_idx]) > 0.1:
                prob = min(prob * 1.5, 0.90)

        # V13 YENİ: Head diversity boost
        # Kafalar birbirine zıt bakıyorsa → daha sık enjekte et
        if self._last_head_div > 0.30:
            prob = min(prob * (1.0 + self._last_head_div * 0.5), 0.95)

        # Time Gate donmuşsa → yarıya indir
        if self.tg_frozen:
            prob *= 0.5

        # BUG FIX (V19): eski seed sadece (cfg.seed, layer_idx, token_turn)'e
        # bağlıydı — cfg.seed sabit (42) ve token_turn her yeni turda 0'a
        # resetlendiği için, N. token'daki L. katmanın inject edilip edilmeme
        # KARARI konuşma içeriğinden tamamen bağımsız olarak HER turda birebir
        # aynıydı (sadece prob eşiği nabla_t/conflict ile oynuyordu, zar hep
        # aynı geliyordu). self.step (küresel, hiç resetlenmeyen sayaç) sokularak
        # gerçek turdan-tura değişkenlik sağlanıyor; reprodüksiyon cfg.seed
        # sabit tutulduğu sürece hâlâ mümkün.
        seed = (self.cfg.seed * 1000003) ^ (layer_idx * 9999991) ^ \
               (self.token_turn * 7919) ^ (self.step * 104729)
        return random.Random(seed).random() < prob

    def should_none(self) -> bool:
        return random.Random(self.cfg.seed ^ self.step).random() < self.cfg.none_prob

    # ── Time Gate (Blur → Sharp) ──
    def time_gate_tick(self, layer_idx: int):
        target = self.cfg.n_layers - self.guard.size - 1
        if layer_idx == target:
            delta        = self.t_model - self.t_eq
            self.tg_bias = -0.08 * math.tanh(delta / 5.0)
            self.t_eq    = self.t_model
            self.tg_frozen = not self.tg_frozen
            self.t_model  += 1.0

    # ── Temporal Ticks ──
    def new_token(self):
        """
        Aggregates layer-by-layer phase and entropy variables into a single 
        unified time-step point for the newly generated token. Ensures phase 
        curvature evaluates temporal progression rather than inter-layer variance.
        """
        if self._tok_nabla:
            m_nabla = sum(self._tok_nabla) / len(self._tok_nabla)
            m_xt    = sum(self._tok_xt)    / len(self._tok_xt)
            m_nt    = sum(self._tok_nt)    / len(self._tok_nt) if self._tok_nt else 0.5
            self.phase.update(m_nabla, m_xt)
            self.entropy.log(self.step, m_nabla, m_xt, m_nt)
            self._tok_nabla.clear()
            self._tok_xt.clear()
            self._tok_nt.clear()
        self.token_turn += 1

    def new_turn_sync(self):
        """
        Converges temporal model clock (t_model) towards equilibrium (t_eq) 
        at the initiation of each conversational turn.
        """
        delta         = self.t_model - self.t_eq
        self.t_eq    += delta * 0.5
        self.token_turn = 0

    def tick(self):
        self.step += 1
        self.fire_count += 1
        _save_step(self.step)
        if self.step % 30 == 0:
            self.calibr.calibrate(self.health.health_score)

    def stats(self) -> dict:
        short, mid, lng = self.entropy.multi_window()
        return {
            "step":       self.step,
            "calls":      self.call_count,
            "fires":      self.fire_count,
            "rate":       round(self.fire_count / max(self.call_count, 1), 4),
            "temp":       round(self.temperature(), 3),
            "alpha":      round(self.alpha, 5),
            "guard":      self.guard.size,
            "health":     round(self.health.health_score, 3),
            "t_model":    self.t_model,
            "tg_frozen":  self.tg_frozen,
            "tg_bias":    round(self.tg_bias, 5),
            "layer_mem":  len(self.layer_memory),
            "df_rows":    len(self.entropy),
            "ent_short":  round(short, 4),
            "ent_mid":    round(mid, 4),
            "ent_long":   round(lng, 4),
            "trend":      round(self.entropy.rolling_trend(), 4),
            "ent_var":    round(self.entropy.rolling_var(), 4),
            "oscillation":round(self.entropy.oscillation_score(), 4),
            "user_spike": round(self.spike.current_spike, 4),
            "acc_spike":  round(self.spike.accumulated(), 4),
            "conflict":   round(self.conflict.compute_conflict(), 4),
            "resonance":  round(self.conflict.resonance_score(), 4),
            "head_div":   round(self.head_conflict.current_diversity, 4),
            "focus":      round(self.head_conflict.current_focus, 4),
            "phase_curv": round(self.phase.curvature(), 4),
            "phase_vel":  round(self.phase.velocity(), 4),
            "w_momentum": round(self._w_momentum, 4),
        }


# ═══════════════════════════════════════════════════════════════════
# GLOBAL ENGINE
# ═══════════════════════════════════════════════════════════════════

DIAMOND_ENGINE: Optional[DiamondEngine] = None

def set_engine(e: DiamondEngine):
    global DIAMOND_ENGINE
    DIAMOND_ENGINE = e


# ═══════════════════════════════════════════════════════════════════
# ROBOTIC & CJK SANITIZER
# ═══════════════════════════════════════════════════════════════════

_ROBOTIC = [
    r"[Bb]en bir yapay zeka[yı]?[mım]*[.,!]?\s*",
    r"[Bb]ir dil modeli olarak[,]?\s*",
    r"[Ss]ize nasıl yardımcı olabilirim[?]?\s*",
    r"^[Mm]erhaba[!,.]?\s*",
    r"^[Tt]abii ki[!,.]?\s*",
    r"^[Ee]lbette[!,.]?\s*",
    r"^[Kk]esinlikle[!,.]?\s*",
    r"[Aa]nlıyorum[,.]?\s*",
    r"[Tt]eşekkür ederim[.,]?\s*",
    r"[Uu]marım bu yardımcı olmuştur[.,]?\s*",
    r"[Bb]aşka bir sorunuz var mı[?,]?\s*",
    r"[Hh]erhangi bir sorunuz olursa[.,]?\s*",
    r"[Yy]ardımcı olmaktan mutluluk duyarım[.,]?\s*",
    r"[Ss]orunuzu daha iyi anlayabilmek için[.,]?\s*",
    r"[Bb]u konuda size yardımcı olmaya çalışacağım[.,]?\s*",
]

_CJK_PATTERN = re.compile(
    r"["
    r"\u4e00-\u9fff"
    r"\u3000-\u303f"
    r"\u3040-\u309f"
    r"\u30a0-\u30ff"
    r"\uac00-\ud7af"
    r"\u0e00-\u0e7f"
    r"\uff00-\uffef"
    r"]+"
)

def _strip_cjk(text: str) -> str:
    lines = text.split("\n")
    clean = []
    for line in lines:
        cjk_c = len(_CJK_PATTERN.findall(line))
        if cjk_c / max(len(line.strip()), 1) < 0.10:
            clean.append(_CJK_PATTERN.sub("", line).strip())
    result = "\n".join(l for l in clean if l).strip()
    return result or "... (dil kayması, tekrar yaz)"

def _filter_robotic(text: str) -> str:
    # We removed _strip_cjk so Ego can actually try to answer without being censored!
    for p in _ROBOTIC:
        text = re.sub(p, "", text, flags=re.IGNORECASE|re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def _word_overlap(a: str, b: str) -> float:
    """
    Computes Jaccard word overlap [0,1].
    Used to empirically measure deterministic collapse mapping onto the text space.
    """
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ═══════════════════════════════════════════════════════════════════
# STATE-SPACE ATTENTION HOOK
# ═══════════════════════════════════════════════════════════════════

def _std_attn(module, query, key, value, attention_mask, scaling, dropout):
    ks = repeat_kv(key,   module.num_key_value_groups)
    vs = repeat_kv(value, module.num_key_value_groups)
    w  = torch.matmul(query, ks.transpose(2,3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask
    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    w = nn.functional.dropout(w, p=dropout, training=module.training)
    return torch.matmul(w, vs).transpose(1,2).contiguous(), w


def diamond_attention(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    global DIAMOND_ENGINE
    layer_idx = getattr(module, "layer_idx", 0)

    if layer_idx == 0 and DIAMOND_ENGINE is not None and getattr(DIAMOND_ENGINE, 'diamond_enabled', True):
        DIAMOND_ENGINE.new_token()

    if DIAMOND_ENGINE is None or not getattr(DIAMOND_ENGINE, 'diamond_enabled', True):
        return _std_attn(module, query, key, value, attention_mask, scaling, dropout)

    ks = repeat_kv(key,   module.num_key_value_groups)
    vs = repeat_kv(value, module.num_key_value_groups)
    w  = torch.matmul(query, ks.transpose(2,3)) * scaling

    # V13: HeadConflictDetector — mask dahil, V12 bug fix
    nabla_t, head_div, focus = DIAMOND_ENGINE.head_conflict.compute(w, attention_mask)
    DIAMOND_ENGINE._last_head_div = head_div
    DIAMOND_ENGINE._last_focus    = focus

    # Extract Phase Variables
    # Normalizing x_t by sqrt(key_len) prevents L2-norm saturation, ensuring 
    # the phase space curvature calculation does not encounter mathematically flat planes.
    x_t = float((w.norm(p=2, dim=-1) / math.sqrt(max(w.size(-1), 1))).mean().clamp(0.001, 30.0))
    n_t = float(torch.sigmoid(w).mean().clamp(1e-6, 0.9999))

    # Context Grounding Constraint
    # Reutilizes the masked softmax cached from HeadConflictDetector to evaluate 
    # prompt-attention mass with zero additional computational overhead.
    pl = DIAMOND_ENGINE.prompt_len
    if pl and w.size(-1) > pl:
        grounding = float(
            DIAMOND_ENGINE.head_conflict.last_probs[..., :pl].sum(dim=-1).mean().clamp(0.0, 1.0)
        )
    else:
        grounding = 1.0   # context henüz prompt sınırını aşmadı → tam bağlı say
    DIAMOND_ENGINE._last_grounding = grounding

    # Temporal Buffering
    # Phase points are aggregated intra-token and processed holistically upon token completion.
    DIAMOND_ENGINE._tok_nabla.append(nabla_t)
    DIAMOND_ENGINE._tok_xt.append(x_t)
    DIAMOND_ENGINE._tok_nt.append(n_t)

    # Null-Gradient Pass
    if nabla_t < 1e-4:
        DIAMOND_ENGINE.conflict.buffers[layer_idx] = 0.0
        return _std_attn(module, query, key, value, attention_mask, scaling, dropout)

    # Conflict buffer güncelle
    DIAMOND_ENGINE.conflict.update(layer_idx, nabla_t)

    # SPI kararı (V13: head_div boost dahil)
    if not DIAMOND_ENGINE.should_inject(layer_idx, nabla_t):
        DIAMOND_ENGINE.layer_memory[layer_idx] = nabla_t
        if attention_mask is not None:
            w = w + attention_mask
        w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        w = nn.functional.dropout(w, p=dropout, training=module.training)
        return torch.matmul(w, vs).transpose(1,2).contiguous(), w

    # Ana zincir
    result = DIAMOND_ENGINE.diamond_master_divisor(nabla_t, x_t, n_t, layer_idx)

    if result is None:
        DIAMOND_ENGINE.layer_memory[layer_idx] = nabla_t
        return _std_attn(module, query, key, value, attention_mask, scaling, dropout)

    w_scale, gate, divisor, scalar, a, b, temp, conflict, resonance = result

    # Context Grounding Gate Formulation
    # Modulates the chaotic scale proportional to the model's focus on the initial prompt.
    # A structural grounding floor ensures chaos does not fully dissolve semantic continuity.
    gf = DIAMOND_ENGINE.cfg.grounding_floor
    grounding_gate = gf + (1.0 - gf) * DIAMOND_ENGINE._last_grounding
    w_scale = w_scale * grounding_gate

    # Telemetry Accumulation (Id Pass)
    DIAMOND_ENGINE.record_id_telemetry(nabla_t, conflict, resonance,
                                     DIAMOND_ENGINE._last_grounding, a, b)

    # V13: Layer-Specific Alpha (bell curve)
    la        = DIAMOND_ENGINE.layer_alpha(layer_idx)
    raw_scale = la * w_scale
    # Tanh darboğazı kırıldı: Faz uzayı, Asal sayılar ve Lyapunov denklemleri 
    # modeli artık 'linear' olarak tam gücüyle bükebilecek.
    actual_scale = 1.0 + (raw_scale * 0.15)

    # Time Gate (Blur → Sharp)
    DIAMOND_ENGINE.time_gate_tick(layer_idx)
    final_scale = actual_scale + DIAMOND_ENGINE.tg_bias
    final_scale = max(final_scale, 0.50)

    # ID PASS: State-Space Perturbation Kernel
    # The gravitational tensor (w_rev) introduces mathematical permutations.
    # The key-dimension flip guarantees scale invariance and operational stability 
    # across both prefill (square) and decode (vector) autoregressive phases.
    w_rev = w.flip(dims=[-1])
    w = (w + (w_rev * 0.15)) / final_scale

    # V13: Stochastic Resonance (anti-zombie)
    w = DIAMOND_ENGINE.apply_stochastic_resonance(w, layer_idx)

    w = torch.nan_to_num(w, nan=0.0, posinf=10.0, neginf=-10.0)

    DIAMOND_ENGINE.layer_memory[layer_idx] = nabla_t
    DIAMOND_ENGINE.tick()

    if DIAMOND_ENGINE.debug_mode:
        c_conf   = C.RED    if conflict > 0.3       else C.GREEN
        c_focus  = C.RED    if focus    > 0.65      else C.WHITE
        c_zombie = C.MAGENTA if DIAMOND_ENGINE.head_conflict.is_zombie(
            DIAMOND_ENGINE.cfg.sr_entropy_thr,
            DIAMOND_ENGINE.cfg.sr_focus_thr
        ) else ""
        _cr, _mg = DIAMOND_ENGINE.phase.curvature_raw()
        print(
            f"  {C.DIM}[L{layer_idx:02d}]{C.RESET} "
            f"α_l={C.YELLOW}{la:.4f}{C.RESET} "
            f"scale={C.CYAN}{actual_scale:.4f}{C.RESET} "
            f"gate={gate:.3f} ∇={nabla_t:.3f} "
            f"conflict={c_conf}{conflict:.3f}{C.RESET} "
            f"hDiv={head_div:.3f} "
            f"focus={c_focus}{focus:.3f}{C.RESET}"
            f"{c_zombie}{'🧟' if c_zombie else ''}{C.RESET} "
            f"curv={DIAMOND_ENGINE.phase.curvature():.3f}(cr={_cr:.2e},mg={_mg:.2e}) "
            f"osc={DIAMOND_ENGINE.entropy.oscillation_score():.2f} "
            f"grnd={C.MAGENTA}{DIAMOND_ENGINE._last_grounding:.2f}{C.RESET}"
        )

    if attention_mask is not None:
        w = w + attention_mask
    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    w = nn.functional.dropout(w, p=dropout, training=module.training)
    return torch.matmul(w, vs).transpose(1,2).contiguous(), w


def install_diamond_hook():
    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register("diamond_v1", diamond_attention)
    try:
        import transformers.masking_utils as mu
        mu.ALL_MASK_ATTENTION_FUNCTIONS.register(
            "diamond_v1", mu.ALL_MASK_ATTENTION_FUNCTIONS["eager"]
        )
    except:
        pass
    cprint("  [HOOK] Diamond V1 registered — Qwen 2.5 14B", C.GREEN, bold=True)


# ═══════════════════════════════════════════════════════════════════
# CLI VIZ FONKSIYONLARI  —  V13: /heads /phase /entropy
# ═══════════════════════════════════════════════════════════════════

def _viz_layers(engine: DiamondEngine):
    n    = engine.cfg.n_layers
    g    = engine.guard.size
    bufs = set(engine.conflict.buffer_layers)
    print(f"\n{C.BOLD}{C.BLUE}  Katman Haritası ({n} katman, guard={g}){C.RESET}")
    row = ""
    for i in range(n):
        if i < g or i >= n - g:
            row += f"{C.DIM}░{C.RESET}"
        elif i in bufs:
            row += f"{C.YELLOW}◆{C.RESET}"
        else:
            v = engine.conflict.buffers.get(i)
            row += f"{C.RED}▪{C.RESET}" if (v and v > 2.0) else f"{C.GREEN}▪{C.RESET}"
    print(f"  [{row}]")
    print(f"  {C.DIM}░=guard  ◆=buffer  ▪=aktif  {C.RED}▪=yüksek∇{C.RESET}\n")

def _show_conflict(engine: DiamondEngine):
    cd = engine.conflict
    c  = cd.compute_conflict()
    r  = cd.resonance_score()
    print(f"\n{C.BOLD}{C.CYAN}  Conflict Durumu{C.RESET}")
    print(f"  Buffer    → {cd.state_str()}")
    bar_c = "█" * int(c * 20)
    bar_r = "█" * int(r * 20)
    c_col = C.RED if c > 0.4 else (C.YELLOW if c > 0.2 else C.GREEN)
    print(f"  Conflict:  [{c_col}{bar_c:<20}{C.RESET}] {c:.3f}")
    print(f"  Resonance: [{C.CYAN}{bar_r:<20}{C.RESET}] {r:.3f}")
    print(f"  Entangle:  {cd.cross_layer_entanglement():.4f}\n")

def _show_health(engine: DiamondEngine):
    h = engine.health
    print(f"\n{C.BOLD}  Sağlık Monitörü{C.RESET}")
    print(f"  Durum:       {h.status_str()}")
    if h._scales:
        print(f"  Ort |scale|: {sum(h._scales)/len(h._scales):.5f}")
        print(f"  Ort  gate:  {sum(h._gates)/len(h._gates):.4f}")
    osc = engine.entropy.oscillation_score()
    c_osc = C.RED if osc > 0.5 else (C.YELLOW if osc > 0.3 else C.GREEN)
    print(f"  Oscillation: {c_osc}{osc:.3f}{C.RESET}")
    print(f"  Guard:       {engine.guard.size} katman ({engine.cfg.guard_ratio:.0%} baz)")
    if h.is_alert:
        cprint("  ⚠️  UYARI! /alpha veya /spi ile ayarla.", C.RED)
    print()

def _show_heads(engine: DiamondEngine):
    hc = engine.head_conflict
    print(f"\n{C.BOLD}{C.MAGENTA}  Head Conflict Durumu (V13){C.RESET}")
    div   = hc.current_diversity
    foc   = hc.current_focus
    r_div = hc.rolling_diversity()
    c_foc = C.RED if foc > 0.65 else (C.YELLOW if foc > 0.5 else C.GREEN)
    c_div = C.RED if div > 0.5  else (C.YELLOW if div > 0.3 else C.GREEN)
    bar_f = "█" * int(foc * 20)
    bar_d = "█" * int(div * 10)
    print(f"  Focus:     [{c_foc}{bar_f:<20}{C.RESET}] {foc:.3f}"
          f"  {'⚠️ zombie risk' if foc > 0.65 else ''}")
    print(f"  Diversity: [{c_div}{bar_d:<10}{C.RESET}] {div:.3f}  "
          f"rolling={r_div:.3f}")
    is_z = hc.is_zombie(engine.cfg.sr_entropy_thr, engine.cfg.sr_focus_thr)
    print(f"  Zombie:    {'🧟 AKTİF — SR devrede' if is_z else '✅ Temiz'}\n")

def _show_phase(engine: DiamondEngine):
    ps = engine.phase
    print(f"\n{C.BOLD}{C.BLUE}  Faz Uzayı Durumu (V13){C.RESET}")
    print(f"  Mevcut: {ps.state_str()}")
    if len(ps._history) >= 3:
        print(f"  Son 5 nokta (∇, x):")
        for i, (n, x) in enumerate(ps._history[-5:]):
            print(f"    [{i}] ∇={n:.3f}  x={x:.3f}")
    print()

def _show_entropy(engine: DiamondEngine):
    short, mid, lng = engine.entropy.multi_window()
    osc   = engine.entropy.oscillation_score()
    trend = engine.entropy.rolling_trend()
    var   = engine.entropy.rolling_var()
    print(f"\n{C.BOLD}{C.CYAN}  Multi-Window Entropy (V13){C.RESET}")
    print(f"  Kısa  (3  adım): {short:.4f}")
    print(f"  Orta  (10 adım): {mid:.4f}")
    print(f"  Uzun  (30 adım): {lng:.4f}")
    trend_sym = "↑" if trend > 0.05 else ("↓" if trend < -0.05 else "→")
    c_osc = C.RED if osc > 0.5 else (C.YELLOW if osc > 0.3 else C.GREEN)
    print(f"  Trend: {trend_sym} {trend:.4f}  |  Var: {var:.4f}")
    print(f"  Oscillation: {c_osc}{osc:.3f}{C.RESET}\n")


# ═══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "Sen Diamond analiz sistemisin. SADECE TÜRKÇE cevap ver. "
    "Kesin olarak emin olmadığın hiçbir bilgiyi söyleme, uydurma."
)


# ═══════════════════════════════════════════════════════════════════
# EĞİTİM
# ═══════════════════════════════════════════════════════════════════

def train(cfg: Optional[DiamondConfig] = None):
    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    cfg = cfg or DiamondConfig()
    print("="*70)
    cprint("  Diamond V1  |  Qwen 2.5 14B Instruct  |  Furkan Elmas 2026", C.CYAN, bold=True)
    cprint("  HeadConflict · PhaseSpace · StoRes · LayerAlpha · Momentum", C.YELLOW)
    print("="*70)

    install_diamond_hook()
    model_id = "huihui-ai/Qwen2.5-14B-Instruct-abliterated"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb,
        attn_implementation="diamond_v1",
        device_map="auto", trust_remote_code=True
    )

    cfg.n_layers = model.config.num_hidden_layers
    engine = DiamondEngine(cfg)
    set_engine(engine)
    cprint(f"  Layers={cfg.n_layers}  Guard={engine.guard.size}  "
           f"AlphaBase={cfg.alpha_base}  Step={engine.step}  "
           f"Primes={len(_PRIMES)}", C.GREEN)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    def fmt(ex):
        u = f"{ex.get('instruction','')} {ex.get('input','')}".strip()
        messages = [
            {"role": "system",    "content": "You are a helpful assistant."},
            {"role": "user",      "content": u},
            {"role": "assistant", "content": ex.get('output','')},
        ]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )}
    dataset = dataset.map(fmt, remove_columns=dataset.column_names)

    sft_cfg = SFTConfig(
        output_dir="diamond_qwen_v1",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=1000,
        learning_rate=2e-4,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        bf16=True,
        report_to="none",
        gradient_checkpointing=False,
        max_length=512,
        dataset_text_field="text"
    )
    trainer = SFTTrainer(
        model=model, train_dataset=dataset,
        peft_config=peft_cfg, args=sft_cfg,
        processing_class=tokenizer
    )
    cprint("\n  EĞİTİM BAŞLIYOR", C.CYAN, bold=True)
    print("-"*70)
    trainer.train()
    trainer.model.save_pretrained("diamond_qwen_v1")
    tokenizer.save_pretrained("diamond_qwen_v1")
    s = engine.stats()
    cprint(f"\n  fires={s['fires']}  rate={s['rate']:.1%}  step={s['step']}", C.GREEN)
    cprint(f"  health={s['health']:.2f}  conflict={s['conflict']:.3f}  "
           f"head_div={s['head_div']:.3f}  osc={s['oscillation']:.3f}", C.YELLOW)
    print("="*70)


# ═══════════════════════════════════════════════════════════════════
# SOHBET
# ═══════════════════════════════════════════════════════════════════

def chat(cfg: Optional[DiamondConfig] = None):
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import PeftModel

    cfg = cfg or DiamondConfig()
    print("="*70)
    cprint("  Diamond V1  |  Qwen 2.5 14B  |  Furkan Elmas 2026", C.CYAN, bold=True)
    cprint("  HeadConflict · PhaseSpace · StoRes · LayerAlpha · Momentum", C.YELLOW)
    cprint(f"  {len(_PRIMES)} asal sayı (2→{_PRIMES[-1]})  ·  62-cycle Prime Regulator", C.DIM)
    print("="*70)

    install_diamond_hook()
    model_id = "huihui-ai/Qwen2.5-14B-Instruct-abliterated"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    turkish_processor = TurkishOnlyLogitsProcessor(tokenizer, device="cuda")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA görünmüyor. 14B 4-bit sohbet için NVIDIA CUDA gerekli.")

    cprint("\n[2] Model yükleniyor (4-bit)...", C.DIM)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    try:
        base = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb,
            attn_implementation="diamond_v1",
            device_map={"": 0},
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA VRAM yetmedi. Bu kod artık device_map='auto' ile CPU/disk offload'a "
            "düşmüyor, çünkü bitsandbytes 4-bit Qwen katmanlarında bu yol "
            "'Cannot copy out of meta tensor' hatasını üretiyor. "
            "14B için VRAM'i boşaltıp tekrar dene; olmazsa 7B/8B veya GGUF/llama.cpp "
            "quant kullanmak gerekiyor."
        ) from e
    base.config.use_cache = True

    meta_params = [name for name, param in base.named_parameters() if param.is_meta]
    if meta_params:
        sample = ", ".join(meta_params[:5])
        raise RuntimeError(
            f"Model eksik yüklendi; meta tensor kaldı: {sample}. "
            "CPU/disk offload kapalı olmalı veya VRAM yetmiyorsa daha küçük/GGUF model kullanılmalı."
        )

    for adapter_dir in ["diamond_qwen_v1", "diamond_qwen_legacy_v12", "diamond_qwen_legacy_v11"]:
        if os.path.isdir(adapter_dir):
            cprint(f"[3] Adapter: {adapter_dir}", C.GREEN)
            model = PeftModel.from_pretrained(base, adapter_dir)
            break
    else:
        cprint("[3] Adapter yok — base model", C.YELLOW)
        model = base
    model.eval()

    infer_device = next(model.parameters()).device

    cfg.n_layers = model.config.num_hidden_layers
    cfg.vocab_n2 = getattr(model.config, "vocab_size", 152064) / 8.0
    engine = DiamondEngine(cfg)
    set_engine(engine)

    cprint(f"\n✅ Diamond V1 Aktif", C.GREEN, bold=True)
    cprint(f"   Layers={cfg.n_layers}  Guard={engine.guard.size}  "
           f"Alpha={cfg.alpha_base}  Step={engine.step}", C.DIM)

    print(f"\n{C.BOLD}  Komutlar:{C.RESET}")
    for cmd in [
        "/alpha <n>              → alpha base",
        "/debug                  → debug aç/kapat",
        "/stats                  → tam istatistik",
        "/df                     → entropy tablosu",
        "/viz                    → katman haritası",
        "/conflict               → conflict / resonance",
        "/health                 → sağlık monitörü",
        "/heads                  → head diversity + zombie durumu  [V13]",
        "/phase                  → faz uzayı yörüngesi           [V13]",
        "/entropy                → multi-window + oscillation     [V13]",
        "/spi <b> <n>            → injection olasılıkları",
        "/sr <sigma>             → stochastic resonance sigma     [V13]",
        "/collapse <thr>         → İD/EGO collapse eşiği (σ+ε≥thr) [V14]",
        "/id-temp <t>            → İd (bilinçaltı) sıcaklığı, öntanımlı 0.90 [V17]",
        "/ego-temp <t>           → Ego (üst bilinç) sıcaklığı, öntanımlı 0.20 [V17]",
        "/config save <f>        → config JSON kaydet",
        "/config load <f>        → config JSON yükle",
        "/reset                  → hafıza temizle",
        "quit                    → çıkış",
    ]:
        print(f"  {C.DIM}{cmd}{C.RESET}")
    print("="*70)

    chat_history = []

    while True:
        try:
            user_input = input(f"\n{C.BOLD}{C.WHITE}Sen:{C.RESET} ").strip()
            if not user_input:
                continue
            low = user_input.lower()

            # ── Komutlar ──
            if low in ["quit","exit","çıkış"]:
                cprint("Görüşmek üzere 👋", C.CYAN); break

            if low in ["/reset","/clear"]:
                chat_history = []
                engine.conflict.buffers.clear()
                engine.layer_memory.clear()
                engine.spike.current_spike  = 0.0
                engine.spike._history_spike = 0.0
                engine.spike.decay_counter  = 0.0
                engine._w_momentum          = 0.0
                engine.phase._history.clear()
                engine._tok_nabla.clear()
                engine._tok_xt.clear()
                engine._tok_nt.clear()
                engine.head_conflict._div_history.clear()
                engine.coherence.conversation_embedding.clear()
                cprint("🧹 Hafıza, buffer ve faz uzayı temizlendi", C.YELLOW)
                continue

            if low == "/debug":
                engine.debug_mode = not engine.debug_mode
                state = f"{C.GREEN}AÇIK" if engine.debug_mode else f"{C.RED}KAPALI"
                print(f"🔬 Debug: {state}{C.RESET}"); continue

            if low == "/viz":      _viz_layers(engine);   continue
            if low == "/conflict": _show_conflict(engine); continue
            if low == "/health":   _show_health(engine);  continue
            if low == "/heads":    _show_heads(engine);   continue
            if low == "/phase":    _show_phase(engine);   continue
            if low == "/entropy":  _show_entropy(engine); continue

            if low == "/stats":
                s = engine.stats()
                print(f"\n{C.BOLD}{C.CYAN}  ─── Diamond V1 İstatistikleri ───{C.RESET}")
                for k, v in s.items():
                    print(f"  {C.DIM}{k:<16}{C.RESET}{C.WHITE}{v}{C.RESET}")
                print(); continue

            if low == "/df":
                tail = engine.entropy.tail(10)
                if _PANDAS_OK and len(engine.entropy) > 0:
                    print(tail.to_string(index=False))
                elif tail:
                    for r in tail: print(r)
                else:
                    cprint("  Veri yok", C.DIM)
                continue

            if low.startswith("/alpha "):
                try:
                    v = float(user_input.split()[1])
                    cfg.alpha_base = v
                    engine.cfg.alpha_base = v
                    engine.alpha = v
                    cprint(f"🔥 AlphaBase → {v}", C.YELLOW)
                    if v > engine.cfg.alpha_max * 0.8:
                        cprint(f"⚠️  alpha_max={engine.cfg.alpha_max}'a yakın, dikkat", C.RED)
                except:
                    cprint("Kullanım: /alpha 0.05", C.RED)
                continue

            if low.startswith("/sr "):
                try:
                    v = float(user_input.split()[1])
                    engine.cfg.sr_sigma = max(0.0, min(v, 0.05))
                    cprint(f"🔊 SR sigma → {engine.cfg.sr_sigma:.4f}", C.MAGENTA)
                except:
                    cprint("Kullanım: /sr 0.012", C.RED)
                continue

            if low.startswith("/collapse "):
                try:
                    v = float(user_input.split()[1])
                    engine.cfg.collapse_threshold = max(0.0, min(v, 2.0))
                    cprint(f"⚡ Collapse threshold → {engine.cfg.collapse_threshold:.3f}", C.RED)
                except:
                    cprint("Kullanım: /collapse 1.00", C.RED)
                continue

            if low.startswith("/id-temp "):
                try:
                    v = float(user_input.split()[1])
                    engine.cfg.id_temperature = max(0.01, min(v, 2.0))
                    cprint(f"🔥 İD sıcaklığı → {engine.cfg.id_temperature:.2f}", C.MAGENTA)
                except:
                    cprint("Kullanım: /id-temp 0.90", C.RED)
                continue

            if low.startswith("/ego-temp "):
                try:
                    v = float(user_input.split()[1])
                    engine.cfg.ego_temperature = max(0.01, min(v, 2.0))
                    cprint(f"🧊 EGO sıcaklığı → {engine.cfg.ego_temperature:.2f}", C.BLUE)
                except:
                    cprint("Kullanım: /ego-temp 0.20", C.RED)
                continue

            if low.startswith("/spi "):
                try:
                    p = user_input.split()
                    engine.cfg.base_prob = float(p[1]) if len(p) > 1 else 0.20
                    engine.cfg.none_prob = float(p[2]) if len(p) > 2 else 0.06
                    cprint(f"🔧 SPI → {engine.cfg.base_prob}/{engine.cfg.none_prob}", C.GREEN)
                except:
                    cprint("Kullanım: /spi 0.20 0.06", C.RED)
                continue

            if low.startswith("/config "):
                parts = user_input.split()
                if len(parts) >= 3:
                    action, fname = parts[1], parts[2]
                    if action == "save":
                        try:
                            cfg.to_json(fname)
                        except Exception as e:
                            cprint(f"Hata: {e}", C.RED)
                    elif action == "load":
                        try:
                            new_cfg = DiamondConfig.from_json(fname)
                            new_cfg.n_layers = cfg.n_layers
                            cfg.__dict__.update(new_cfg.__dict__)
                            engine.cfg.__dict__.update(new_cfg.__dict__)
                            cprint(f"✅ Config yüklendi: {fname}", C.GREEN)
                        except Exception as e:
                            cprint(f"Hata: {e}", C.RED)
                else:
                    cprint("/config save <f>  |  /config load <f>", C.RED)
                continue

            # ── Normal konuşma ──
            engine.new_turn_sync()

            # Intent classification + EMA smoothing
            new_intent = engine.intent.classify(user_input)
            if len(chat_history) == 0:
                engine.intent_multiplier = new_intent
            else:
                engine.intent_multiplier = engine.intent_multiplier * 0.6 + new_intent * 0.4

            # Coherence tracker + adaptive tau
            engine.coherence.update(user_input)
            engine.spike.adaptive_tau(engine.coherence.topic_shift)
            raw_spike = engine.spike.compute_spike(
                user_input, history_depth=len(chat_history)//2
            )
            engine.spike.current_spike = engine.coherence.modulate_spike(raw_spike)

            intent_label = (
                "🔢 MATH" if engine.intent_multiplier < 0.30 else
                ("❓ FACT" if engine.intent_multiplier < 0.60 else "💬 CASUAL")
            )
            if engine.debug_mode:
                cprint(
                    f"  [INTENT] {intent_label}  α×={engine.intent_multiplier:.2f} "
                    f"(raw={new_intent:.2f})",
                    C.CYAN
                )
                cprint(
                    f"  [COHERENCE] score={engine.coherence.coherence_score:.2f}  "
                    f"shift={engine.coherence.topic_shift:.2f}  "
                    f"tau={engine.spike.tau:.1f}  "
                    f"spike={engine.spike.current_spike:.3f}",
                    C.CYAN
                )

            chat_history.append({"role": "user", "content": user_input})
            if len(chat_history) > 14:
                chat_history = chat_history[-14:]

            # --- AŞAMA 1: İD (Bilinçaltı / Diamond Aktif / w = w / final_scale) ---
            engine.diamond_enabled = True
            messages_id = [{"role": "system", "content": _SYSTEM_PROMPT}] + chat_history
            text_id = tokenizer.apply_chat_template(messages_id, tokenize=False, add_generation_prompt=True)
            inputs_id = tokenizer([text_id], return_tensors="pt").to(infer_device)
            in_len_id = inputs_id.input_ids.shape[1]

            # V16: Hook'un context-grounding hesaplayabilmesi için prompt sınırını bildir,
            # ve bu turun İD telemetrisini sıfırla (EGO'ya taşınacak ham sinyaller).
            engine.prompt_len = in_len_id
            engine.reset_id_telemetry()

            t0 = time.time()
            with torch.no_grad():
                out_id = model.generate(
                    **inputs_id, max_new_tokens=256, temperature=engine.cfg.id_temperature, top_p=0.95, top_k=50,
                    logits_processor=LogitsProcessorList([turkish_processor]), repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=True
                )
            dt_id = time.time() - t0
            id_resp = tokenizer.decode(out_id[0][in_len_id:], skip_special_tokens=True).strip()

            # V16: İD'nin bilinçaltı üretimi boyunca biriken Diamond telemetrisi — EGO bunu OKUYUP sorgulayacak
            id_tele = engine.get_id_telemetry_avg()
            # --- PHASE 2: EGO SYNTHESIS (Diamond Deactivated) ---
            # During the Ego pass, the Diamond engine is bypassed to allow the model's 
            # native rational architecture (Pure Qwen) to synthesize the Id's hallucinations.
            engine.diamond_enabled = False

            # V14: Gerçek collapse kararı — σ+ε≥1 analojisi
            # conflict_val: buffer katmanları arası çatışma [0,1) — tanh ile sıkıştırılmış
            # osc_val:      entropi salınım skoru [0,1] — model kararsız mı salınıyor mu
            conflict_val   = engine.conflict.compute_conflict()
            osc_val        = engine.entropy.oscillation_score()
            collapse_score = conflict_val + osc_val
            id_dominant    = collapse_score >= engine.cfg.collapse_threshold

            # Ego Synthesis Telemetry Prompt
            # Projects the internal mathematical state-space constraints into natural language, 
            # allowing the Ego to contextualize and rationalize the Id's stochastic output.
            ego_user_prompt = (
                f"Kullanıcı Sorusu: {user_input}\n\n"
                f"Bilinçaltı (İd) Verisi: {id_resp}\n\n"
                "--- İd'nin bu veriyi üretirken Diamond denklemlerinden ölçülen bilinçaltı telemetrisi ---\n"
                f"• Self-Return oranı (a/b permütasyonu): {id_tele['sr_ratio']:.3f}  "
                "(1.0'a yakınsa simetrik/dengeli döngü, uzaksa asimetrik/çarpık bir permütasyon içinde kalmış)\n"
                f"• Katman-arası Çatışma (conflict): {id_tele['conflict']:.3f} / 1.0  "
                "(yüksekse bilinçaltı kendi İÇİNDE tutarsız konuşmuş — farklı katmanlar birbirine karşı çıkmış)\n"
                f"• Rezonans: {id_tele['resonance']:.3f} / 1.0  "
                "(yüksekse katmanlar senkrona girmiş, tekrar eden bir motif/saplantı olabilir)\n"
                f"• Bağlam Çıpası (grounding): {id_tele['grounding']:.3f} / 1.0  "
                "(düşükse bilinçaltı üretim sırasında orijinal soruya dikkatini kaybetmiş, konudan kopmuş olabilir)\n\n"
                "Görevin: Asistan gibi konuşma. Yukarıdaki 4 sinyale bakarak KENDİN karar ver:\n"
                "- Bağlam Çıpası düşükse (<0.5): bilinçaltı sorudan kopmuş demektir, yeniden yazarken onu "
                "GERÇEK soruya geri çek, konu dışı kısımları törpüle.\n"
                "- Çatışma yüksekse (>0.5): bilinçaltı kendi içinde çelişmiş demektir, çelişen parçaları "
                "ayıklayıp TEK bir tutarlı bakış açısına indirge.\n"
                "- Bağlam Çıpası yüksek VE Çatışma düşükse: bilinçaltı hem soruya sadık hem tutarlı kalmış — "
                "bu durumda onun tuhaf kavramlarını, metaforlarını SİLME, olduğu gibi devrimsel bir dille törpüle.\n"
                "- Rezonans yüksekse: aynı motifi tekrar tekrar döndürüyor olabilir, tekrarı kır ama fikri kaybetme.\n"
                "Bu dört sinyeli birlikte değerlendirip nihai cevabı yaz."
            )

            messages_ego = chat_history[:-1] + [{"role": "user", "content": ego_user_prompt}]
            text_ego = tokenizer.apply_chat_template(messages_ego, tokenize=False, add_generation_prompt=True)
            inputs_ego = tokenizer([text_ego], return_tensors="pt").to(infer_device)
            in_len_ego = inputs_ego.input_ids.shape[1]
            
            t1 = time.time()
            with torch.no_grad():
                out_ego = model.generate(
                    **inputs_ego, max_new_tokens=256, temperature=engine.cfg.ego_temperature, top_p=0.95, top_k=50,
                    logits_processor=LogitsProcessorList([turkish_processor]), repetition_penalty=1.05, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=True
                )
            dt_ego = time.time() - t1
            dt = dt_id + dt_ego
            
            resp = tokenizer.decode(out_ego[0][in_len_ego:], skip_special_tokens=True).strip()
            resp = _filter_robotic(resp)

            # Collapse (Zorunlu Fallback) mekanizması kullanıcı isteğiyle tamamen iptal edildi.
            overlap = _word_overlap(id_resp, resp)

            print(f"\n{C.DIM}[İD HALÜSİNASYONU (Saf Bölme)]: {id_resp}{C.RESET}")
            print(f"  {C.DIM}[Diamond Kaos Skoru={collapse_score:.3f} (conflict={conflict_val:.3f}+osc={osc_val:.3f}) "
                  f"· Ego Sentez Örtüşmesi (overlap)={overlap:.2f}] {C.BLUE}EGO-rewrite (Zorunlu){C.RESET}")
            print(f"  {C.DIM}[İD Telemetri → grnd={id_tele['grounding']:.2f} sr_ratio={id_tele['sr_ratio']:.2f} "
                  f"conflict={id_tele['conflict']:.2f} resonance={id_tele['resonance']:.2f} n={id_tele['n']}]{C.RESET}")
            print(f"  {C.DIM}[sıcaklık: İd={C.MAGENTA}{engine.cfg.id_temperature:.2f}{C.RESET}{C.DIM} · "
                  f"Ego={C.BLUE}{engine.cfg.ego_temperature:.2f}{C.RESET}{C.DIM}]{C.RESET}")
            print(f"{C.BOLD}{C.CYAN}Diamond (Ego):{C.RESET} {resp}")

            # V16: Collapse + telemetri event logu — grounding/sr_ratio dahil, TAM görüntü
            try:
                with open(engine.cfg.collapse_log_file, "a", encoding="utf-8") as _lf:
                    _lf.write(json.dumps({
                        "step":            engine.step,
                        "conflict":        round(conflict_val, 4),
                        "oscillation":     round(osc_val, 4),
                        "collapse_score":  round(collapse_score, 4),
                        "threshold":       engine.cfg.collapse_threshold,
                        "id_dominant":     id_dominant,
                        "overlap":         round(overlap, 4),
                        "forced_fallback": False,
                        "id_grounding":    round(id_tele["grounding"], 4),
                        "id_sr_ratio":     round(id_tele["sr_ratio"], 4),
                        "id_conflict_avg": round(id_tele["conflict"], 4),
                        "id_resonance_avg":round(id_tele["resonance"], 4),
                        "id_telemetry_n":  id_tele["n"],
                        "id_resp_preview": id_resp[:120],
                        "resp_preview":    resp[:120],
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass
            
            engine.diamond_enabled = True

            n_new = out_ego.shape[1] - in_len_ego
            h_str = engine.health.status_str()
            zombie_tag = " 🧟SR" if engine.head_conflict.is_zombie(
                engine.cfg.sr_entropy_thr, engine.cfg.sr_focus_thr
            ) else ""
            cprint(
                f"  [{n_new} tok · {dt:.1f}s · α={engine.alpha:.4f} · "
                f"conflict={engine.conflict.compute_conflict():.3f} · "
                f"hDiv={engine.head_conflict.current_diversity:.3f}{zombie_tag}] {h_str}",
                C.DIM
            )

            if engine.health.is_alert:
                cprint("  ⚠️  Sağlık uyarısı! /health ile kontrol et.", C.RED)

            chat_history.append({"role": "assistant", "content": resp})

            if len(chat_history) % 12 == 0:
                s = engine.stats()
                cprint(
                    f"  [V13] step={s['step']}  α={s['alpha']:.4f}  "
                    f"temp={s['temp']:.2f}  fires={s['fires']}  "
                    f"conflict={s['conflict']:.3f}  head_div={s['head_div']:.3f}  "
                    f"osc={s['oscillation']:.3f}",
                    C.DIM
                )

        except KeyboardInterrupt:
            cprint("\nKapanıyor...", C.DIM); break
        except Exception as e:
            cprint(f"\nHata: {e}", C.RED)
            import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diamond V1 — Qwen 2.5 14B",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("mode",       nargs="?", default="chat",
                        choices=["train","chat"])
    parser.add_argument("--config",   type=str,   default=None)
    parser.add_argument("--alpha",    type=float, default=None)
    parser.add_argument("--prob",     type=float, default=None)
    parser.add_argument("--guard",    type=float, default=None)
    parser.add_argument("--lyapunov", type=float, default=None)
    parser.add_argument("--harmonic", type=float, default=None)
    parser.add_argument("--sr-sigma", type=float, default=None, dest="sr_sigma")
    parser.add_argument("--momentum", type=float, default=None)
    args = parser.parse_args()

    cfg = DiamondConfig()
    if args.config:    cfg = DiamondConfig.from_json(args.config)
    if args.alpha:     cfg.alpha_base      = args.alpha
    if args.prob:      cfg.base_prob       = args.prob
    if args.guard:     cfg.guard_ratio     = args.guard
    if args.lyapunov:  cfg.lyapunov_coef   = args.lyapunov
    if args.harmonic:  cfg.harmonic_coef   = args.harmonic
    if args.sr_sigma:  cfg.sr_sigma        = args.sr_sigma
    if args.momentum:  cfg.w_momentum_decay= args.momentum

    if args.mode == "train":
        train(cfg)
    else:
        chat(cfg)