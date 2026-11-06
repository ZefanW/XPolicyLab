"""Run RoboDojo's native client, with finger reports and synchronized camera calibration."""
import functools
from contextlib import ExitStack
import os
from pathlib import Path
import runpy
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit('usage: python -m XPolicyLab.policy.JustRUA.observations.launch <RoboDojo>/src/eval_client/main.py [native client options]')
    native_main = Path(args[0]).resolve()
    if not native_main.is_file() or native_main.parts[-3:] != ('src', 'eval_client', 'main.py'):
        raise ValueError('Expected RoboDojo src/eval_client/main.py')
    root = native_main.parents[2]
    from isaaclab.app import AppLauncher
    from .native import install_after_app

    hooks = ExitStack()
    original_init = AppLauncher.__init__
    original_exec = os.execv

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        hooks.enter_context(install_after_app())

    def execv(path, command):
        # A self-restart from the raw entry would skip these sensor hooks and
        # potentially repeat model calls. Leave the native resume state for review.
        if (len(command) >= 2 and Path(path).resolve() == Path(sys.executable).resolve()
                and Path(command[1]).resolve() == native_main):
            raise RuntimeError('RoboDojo requested a process restart. Review the recorded episode before relaunching JustRUA.')
        return original_exec(path, command)

    previous_cwd, previous_args, previous_path = Path.cwd(), sys.argv, list(sys.path)
    try:
        os.chdir(root)
        sys.path.insert(0, str(root))
        sys.argv = [str(native_main), *args[1:]]
        AppLauncher.__init__ = initialize
        os.execv = execv
        runpy.run_path(str(native_main), run_name='__main__')
    finally:
        os.execv = original_exec
        AppLauncher.__init__ = original_init
        sys.argv = previous_args
        sys.path[:] = previous_path
        os.chdir(previous_cwd)
        hooks.close()


if __name__ == '__main__':
    main()
