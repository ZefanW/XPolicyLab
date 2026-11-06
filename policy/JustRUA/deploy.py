"""RoboDojo entry point. The native Cartesian planner lives in the environment."""
import os
from contextlib import ExitStack
from dataclasses import replace

from XPolicyLab.policy.GPT_6_Astra_Direct_EEF import deploy as eef
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect import deploy as runner
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.docs import eef_docs_from_joint_docs

from .policy import JustRUAPolicy


def eval_one_episode(TASK_ENV, model_client):
    if os.environ.get("EVAL_ENV_TYPE") == "debug":
        # Exercise the actual server/decode/action transport without a model account or physics.
        from XPolicyLab.policy.demo_policy.deploy import eval_one_episode as debug_episode
        return debug_episode(TASK_ENV, model_client)

    from .actor import CodingClient
    from .observations import Observer

    with ExitStack() as resources:
        def make_policy(*, action_spec, env, task_env):
            observer = Observer(task_env)
            resources.callback(observer.close)
            client = resources.enter_context(CodingClient())
            return JustRUAPolicy(
                planner=eef._planner(task_env),
                action_spec=replace(action_spec, docs=eef_docs_from_joint_docs(action_spec.docs)),
                observer=observer,
                client=client,
            )

        return runner.eval_one_episode(TASK_ENV, model_client, policy_factory=make_policy)


def eval_one_episode_batch(TASK_ENV, model_client):
    if int(getattr(TASK_ENV, "num_envs", 1) or 1) != 1:
        raise ValueError("JustRUA expects one RoboDojo environment per conversation.")
    return eval_one_episode(TASK_ENV, model_client)
