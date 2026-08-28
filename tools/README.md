# Tools

Supporting tools remain outside the Python core because they have independent runtimes and release lifecycles.

| Component | Runtime | Responsibility |
| --- | --- | --- |
| [Inertial Positioning Lab](inertial-positioning-lab/) | Android/Kotlin + Python | Sensor capture, `.iplab` archives, canonical conversion, and on-device inference |
| [PaperFlow IO](paperflow-io/) | Codex skills + Python | IO-specific ideation, feasibility review, and experiment planning |

Each tool owns its dependencies, tests, and usage documentation. Shared canonical data behavior belongs in `src/inertial_benchmark/`.
