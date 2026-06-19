# Asset Attribution

SlipZero vendors two robot models from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
so the project runs from a fresh clone without fetching external assets.

Source: `google-deepmind/mujoco_menagerie` @ commit `accb6df40a9a1d1e49eff88157f6818b63a49335`

| Directory | Model | Upstream | License |
|---|---|---|---|
| `franka_emika_panda/` | Franka Emika Panda (7-DOF arm) | Franka Robotics GmbH; MJCF by DeepMind | Apache-2.0 (see `franka_emika_panda/LICENSE`) |
| `leap_hand/` | LEAP Hand (16-DOF dexterous hand) | Ananye Agarwal et al.; MJCF by DeepMind | MIT (see `leap_hand/LICENSE`) |

The vendored MJCF and meshes are unmodified. SlipZero composes the arm and hand at
load time (see `slipzero/env.py`): the LEAP palm is attached to the Panda flange
(`attachment_site`), and the bench/vial/cap scene plus all SlipZero sensors are added
programmatically. The task scene itself lives in `slipzero_bench.xml`.
