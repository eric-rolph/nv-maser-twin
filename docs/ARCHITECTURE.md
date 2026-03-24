# Architecture

> NV Maser "Tricorder" Digital Twin — system architecture overview.

## System Diagram

```mermaid
graph TB
    subgraph Config["Configuration Layer"]
        SC[SimConfig<br/>24 Pydantic models]
    end

    subgraph Physics["Physics Engine (57 modules)"]
        direction TB
        Grid[SpatialGrid<br/>64×64, 10 mm]
        BF[BaseField<br/>Halbach B₀]
        Dist[Disturbance<br/>harmonics, mains, drift]
        Coils[ShimCoilArray<br/>gradient harmonics]
        ENV[FieldEnvironment<br/>compositor]
        NV[NV Spin<br/>T₁/T₂/T₂*]
        Pump[Optical Pump<br/>polarization]
        Thermal[Thermal Model<br/>heating → Q/T₂* shift]
        Cavity[Cavity QED<br/>cooperativity, threshold]
        SC_mod[Signal Chain<br/>SNR budget]
        MB[Maxwell-Bloch<br/>ODE solver]
        CL[Closed Loop<br/>time-stepping sim]

        Grid --> ENV
        BF --> ENV
        Dist --> ENV
        Coils --> ENV
        NV --> ENV
        Pump --> ENV
        Thermal --> ENV
        Cavity --> ENV
        SC_mod --> ENV
        MB --> ENV
        ENV --> CL
    end

    subgraph ML["ML Layer"]
        direction TB
        Ctrl[Neural Controller<br/>CNN / MLP / LSTM]
        Loss[Physics Loss<br/>variance + gain + coop]
        Trainer[Supervised Trainer<br/>checkpointing, tracker]
        PPO[PPO Trainer<br/>ActorCritic, GAE]
        ShimEnv[ShimmingEnv<br/>Gymnasium RL env]
        Bridge[Bridge<br/>policy → closed-loop]

        Ctrl --> Trainer
        Loss --> Trainer
        Ctrl --> PPO
        ShimEnv --> PPO
        Bridge --> CL
    end

    subgraph Serving["Serving Layer"]
        API[FastAPI Server<br/>6 endpoints]
        ONNX[ONNX Export]
    end

    subgraph Infra["Infrastructure"]
        DB[SQLite Tracker<br/>experiment history]
        Docker[Docker<br/>multi-stage, non-root]
    end

    SC --> Physics
    SC --> ML
    Physics --> ML
    Ctrl --> API
    ENV --> API
    Ctrl --> ONNX
    Trainer --> DB
    PPO --> DB
    API --> Docker
```

## Layer Dependencies

```
config.py (SimConfig — 24 Pydantic models)
    │
    ├── physics/  ← config only
    │   ├── grid.py, base_field.py, disturbance.py, coils.py
    │   ├── nv_spin.py, optical_pump.py, thermal.py
    │   ├── cavity.py, signal_chain.py, maxwell_bloch.py
    │   ├── environment.py  ← composes all physics
    │   └── closed_loop.py  ← time-stepping simulator
    │
    ├── model/  ← config + physics
    │   ├── controller.py  (CNN / MLP / LSTM factory)
    │   ├── loss.py         (physics-informed: variance + gain_budget + cooperativity)
    │   └── training.py     (supervised loop + checkpointing)
    │
    ├── rl/  ← config + physics + model
    │   ├── env.py    (ShimmingEnv — Gymnasium wrapper over FieldEnvironment)
    │   ├── ppo.py    (ActorCritic, GAE, RolloutBuffer, PPOTrainer)
    │   └── bridge.py (load trained policy → closed-loop validation)
    │
    └── api/  ← config + physics + model
        └── server.py  (FastAPI — /health /shim /metrics /reload /info /ui)
```

## Data Flow

### Supervised Training Path

```mermaid
sequenceDiagram
    participant C as Config
    participant E as FieldEnvironment
    participant T as Trainer
    participant M as Controller
    participant DB as SQLite Tracker

    C->>E: SimConfig
    E->>E: generate_training_data(N)
    E->>T: distorted_fields, disturbances
    T->>M: forward(field) → currents
    T->>E: apply_correction(currents)
    E->>T: corrected_field
    T->>T: physics_loss(corrected)
    T->>M: backward + step
    T->>DB: log(epoch, loss, metrics)
```

### RL Training Path

```mermaid
sequenceDiagram
    participant C as Config
    participant SE as ShimmingEnv
    participant E as FieldEnvironment
    participant PPO as PPOTrainer
    participant AC as ActorCritic

    C->>SE: SimConfig
    SE->>E: reset() → distorted field
    loop Episode steps
        SE->>AC: obs → action
        AC->>SE: coil currents
        SE->>E: apply_correction
        E->>SE: corrected field + reward
        SE->>PPO: store transition
    end
    PPO->>AC: update policy (GAE + clipped objective)
```

### Inference / Serving Path

```mermaid
sequenceDiagram
    participant HW as Hardware / Client
    participant API as FastAPI
    participant M as Controller
    participant E as FieldEnvironment

    HW->>API: POST /shim {field_map}
    API->>M: forward(field) → currents
    API->>E: apply_correction(currents)
    E->>API: corrected_field + metrics
    API->>HW: {currents, variance, snr, cooperativity}
```

## Physics Module Map

| Domain | Modules | Key Outputs |
|--------|---------|-------------|
| **Field Generation** | `base_field`, `halbach`, `planar_gradient`, `depth_profile` | B₀ map (T) |
| **Disturbance** | `disturbance`, `thermal`, `quantum_noise`, `artifact_characterizer` | ΔB perturbations |
| **NV Dynamics** | `nv_spin`, `optical_pump`, `pulsed_pump` | T₂*, pump efficiency |
| **Cavity & Maser** | `cavity`, `maser_gain`, `q_boost`, `gain_lock` | Q, cooperativity, gain budget |
| **Signal Chain** | `signal_chain`, `snr_calculator`, `sensitivity` | SNR (dB), noise temperature |
| **Time-Domain** | `maxwell_bloch`, `spectral_maxwell_bloch` | Output power, photon number |
| **Spectral** | `spectral`, `dipolar`, `spin_squeezing`, `superradiance` | Inversion profiles, coupling |
| **Shimming** | `grid`, `coils`, `surface_coil`, `feedback`, `closed_loop` | Corrected field, loop metrics |
| **Adapters** | `magpylib_adapter`, `sigpy_adapter`, `epg_adapter`, `mrzero_adapter`, `susceptibility_adapter` | External library bridges |
| **Validation** | `phase1_validator`, `phase4_validator`, `phase6_validator`, `phase9_validator` | Pass/fail + diagnostic dicts |

## Configuration Tree

`SimConfig` composes 24 Pydantic sub-models:

```
SimConfig
├── grid: GridConfig              # 64×64, 10 mm, 0.6 active fraction
├── field: FieldConfig            # B₀ = 50 mT, optional gradient
├── halbach: HalbachConfig        # Halbach geometry & tolerances
├── disturbance: DisturbanceConfig # Harmonics, mains, transients, drift
├── coils: CoilConfig             # Shim coil count & max current
├── nv: NVConfig                  # NV density, T₁/T₂/T₂*, pump efficiency
├── maser: MaserConfig            # Cavity Q, frequency, min gain budget
├── cavity: CavityConfig          # Mode volume, mirror reflectivity
├── optical_pump: OpticalPumpConfig # Power, wavelength, pulsed mode
├── signal_chain: SignalChainConfig # ADC/DAC bits, noise, temperature
├── model: ModelConfig            # Architecture, hidden size, layers
├── training: TrainingConfig      # LR, epochs, reward shaping
├── feedback: FeedbackConfig      # Hall sensor count, DAC bits
├── thermal: ThermalConfig        # Initial temp, cooling coefficient
├── viz: VizConfig                # Plot options
├── maxwell_bloch: MaxwellBlochConfig # Enable, duration, tolerances
├── spectral: SpectralConfig      # Enable, linewidth, detuning grid
├── dipolar: DipolarConfig        # Enable, NV density
├── single_sided_magnet: SingleSidedMagnetConfig
├── surface_coil: SurfaceCoilConfig
├── susceptibility: SusceptibilityConfig
└── depth_profile: DepthProfileConfig
```

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Uptime, grid size, coil count, architecture |
| `POST` | `/shim` | Primary inference: field map → coil currents + physics metrics |
| `GET` | `/metrics` | Prometheus-compatible counters & latency |
| `GET` | `/info` | Model metadata & capability summary |
| `POST` | `/reload` | Hot-reload model checkpoint & config |
| `GET` | `/ui` | Interactive web dashboard |

Authentication: optional `X-API-Key` header (set via `NV_MASER_API_KEY` env var).

## Key Design Decisions

See `docs/adr/` for formal Architecture Decision Records.

| Decision | Rationale |
|----------|-----------|
| **Pydantic config** | Single source of truth; JSON-serializable; validation built-in |
| **Frozen dataclasses** for physics results | Immutable, typed, IDE-friendly; `MaserMetrics`, `UniformityReport` |
| **`FieldEnvironment` compositor** | Isolates physics from ML; single call produces all metrics |
| **Gymnasium RL env** | Standard interface; compatible with any RL library |
| **SQLite experiment tracker** | Zero-dependency; local-first; no cloud lock-in |
| **Multi-stage Docker** | Minimal runtime image; non-root; read-only filesystem |
| **ONNX export** | Cross-runtime deployment (C++, WASM, edge devices) |
