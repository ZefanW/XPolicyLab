# JustRUA

JustRUA is an evaluation-only policy for the dual-arm ARX X5 in RoboDojo. A single GPT-6 Astra conversation controls one episode. The actor receives a task instruction and the upstream task recipe when one is available. At each decision it receives:

- Head and wrist RGB images, each paired with a colorized Depth Pro estimate computed from that camera's RGB.
- Controller gripper targets alongside measured finger positions, finger contact and wrist force readings. A closed target can coexist with fingers that have stopped farther apart.
- Measured robot state, camera poses and intrinsics, remaining environment steps and the arrival error for the last target.

The actor can use a local coding runtime to work with the same images and readings it received. Each episode has its own conversation, read-only observation files and writable scratch space. The actor receives depth as an image. Raw metric depth arrays, object poses and task evaluator state stay outside its inputs.

A `move_eef` decision selects absolute world-frame grasp-point dimensions. XPolicyLab's Direct EEF planner converts them to joint controls. RoboDojo executes up to the first 20 controls, discards the rest and reports the measured outcome. The episode may end earlier under its native rules. There is no policy limit on decision count. The actor uses `gpt-6-astra` at medium reasoning effort with a persistent thread across decisions.

## Install

Use a working RoboDojo evaluation checkout with its Isaac environment, robot assets and Cartesian planner, and put this XPolicyLab checkout at `$ROBODOJO_ROOT/XPolicyLab`. The debug client also reads RoboDojo's `env_cfg/arx_x5.yml` from the directory above XPolicyLab. Run the commands below from the XPolicyLab root. Set `EVAL_PY` to the Python for RoboDojo and `SERVER_PY` to the Python for the policy server. They may be the same. Depth Pro runs in the **evaluation** process alongside the simulator. Keep the PyTorch and torchvision builds compatible with your Isaac installation.

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo
export EVAL_PY=/path/to/robodojo-env/bin/python
export SERVER_PY=/path/to/policy-env/bin/python

bash policy/JustRUA/install.sh "$SERVER_PY"
bash policy/JustRUA/install.sh "$EVAL_PY"
"$EVAL_PY" -m pip install 'transformers==4.57.6' safetensors
export JUSTRUA_DEPTHPRO_CHECKPOINT="$HOME/.local/share/justrua/depthpro"
"$EVAL_PY" -m XPolicyLab.policy.JustRUA.download_checkpoint "$JUSTRUA_DEPTHPRO_CHECKPOINT"
```

The downloader fetches the configuration, image processor and weights from a pinned commit and verifies all three SHA-256 hashes. Depth Pro uses its learned field of view. Its checkpoint is distributed by Apple under its own terms. [Depth Pro model card](https://huggingface.co/apple/DepthPro-hf/tree/de816c8ce7168afcb231f96d501d72b869d0beda)

### Coding runtime

On the evaluation machine install the native Codex CLI **0.153.4** and its matching `codex-code-mode-host`. The platform npm package contains both. For example, on Linux or macOS:

```bash
npm install --prefix "$HOME/.local/share/justrua/tools" @openai/codex@0.153.4
mode_host="$(find "$HOME/.local/share/justrua/tools/node_modules" -type f -name codex-code-mode-host -print -quit)"
test -n "$mode_host" && export JUSTRUA_CODEX="$(dirname "$mode_host")/codex"
"$JUSTRUA_CODEX" --version
```

Give the coding tools a separate Python environment outside RoboDojo and XPolicyLab. Install only the generic numerical and image libraries there. If you have a standalone Node binary for native coding, set `JUSTRUA_NODE` to its absolute path.

```bash
python3 -m venv "$HOME/.local/share/justrua/coding-env"
export JUSTRUA_PYTHON="$HOME/.local/share/justrua/coding-env/bin/python"
"$JUSTRUA_PYTHON" -m pip install numpy Pillow
"$EVAL_PY" -m XPolicyLab.policy.JustRUA.actor.preflight
```

Preflight uses the real app-server to check scratch writes, read-only observation access and denial of network and private-file access. It starts zero model turns. A simulator evaluation requires an existing Codex login with access to the selected model. The provider process uses the current `CODEX_HOME/auth.json` (by default `~/.codex/auth.json`), or an existing file selected with `JUSTRUA_AUTH_FILE`. Keep credentials outside the coding environment and the repository.

## Evaluate

```bash
bash policy/JustRUA/eval.sh \
  RoboDojo insert_tubes no-checkpoint arx_x5 joint 0 0 0 "$SERVER_PY" "$EVAL_PY"
```

The last five arguments are seed, policy GPU, environment GPU, policy environment and evaluation environment. The default Depth Pro device is `cuda:0` within the evaluation process. Select a supported device explicitly with `JUSTRUA_DEPTHPRO_DEVICE`. `no-checkpoint` is the harness placeholder for the robot actor. The auxiliary depth checkpoint is selected separately. This policy uses one simulator environment per process. The standard WebSocket endpoint handles the harness lifecycle; the actor and planner live in the environment client so they can use the current robot readings.

For an offline transport check, use the same command with `EVAL_ENV_TYPE=debug`. Set `DEBUG_OBS_ENCODED=1` for compressed camera buffers. The upstream debug environment runs a 20-step joint hold loop without a simulator, provider account or depth weights.

```bash
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash policy/JustRUA/eval.sh \
  RoboDojo insert_tubes no-checkpoint arx_x5 joint 0 0 0 "$SERVER_PY" "$EVAL_PY"

"$EVAL_PY" -m unittest discover -s policy/JustRUA/tests
"$EVAL_PY" -m unittest XPolicyLab.policy.JustRUA.actor.test_client
```

## Data and training

JustRUA performs inference with pretrained models. There is no task-specific training or data preparation stage. The robot measurement code carries its [MIT notice](observations/LICENSE). XPolicyLab supplies the [planner and robot conventions](../GPT_6_Astra_Direct_EEF/README.md).
