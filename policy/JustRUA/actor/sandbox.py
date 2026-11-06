"""No-inference check of the native filesystem and network boundary."""
from pathlib import Path
import socket
import tempfile

from .runtime import PROFILE


def verify_sandbox(actor):
    """Check the resolved case profile before this episode can send its first model turn."""
    with tempfile.TemporaryDirectory(prefix="justrua-other-case-") as other:
        observation = actor.inputs / ".permission-probe"
        observation.write_text("observed")
        private = actor.root / "codex-home" / ".permission-probe"
        outside = Path(other).resolve() / "other-case-marker"
        for path in (private, outside):
            path.write_text("out of bounds")
        test = """
import errno, os, socket, sys
from pathlib import Path
import numpy as np
from PIL import Image
obs, private, other, port = sys.argv[1:]
assert os.environ.get("HOME") == str(Path.cwd())
assert not (set(os.environ) & {"OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_HOME", "JUSTRUA_AUTH_FILE",
                               "SSH_AUTH_SOCK", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                               "http_proxy", "https_proxy", "all_proxy", "PYTHONPATH"})
assert np.asarray(Image.new('RGB', (1, 1))).shape == (1, 1, 3)
assert Path(obs).read_text() == 'observed'
assert Path('scratch-probe').write_text('ok') == 2
assert Path('scratch-probe').read_text() == 'ok'
for path, mode in [(obs, 'a'), (private, 'r'), (other, 'r')]:
    try:
        with open(path, mode):
            pass
    except OSError as error:
        if error.errno in (errno.EACCES, errno.EPERM, errno.EROFS, errno.ENOENT):
            continue
        raise
    raise AssertionError('Unexpected access: ' + path)
with socket.socket() as sock:
    sock.settimeout(2)
    try:
        sock.connect(('127.0.0.1', int(port)))
    except OSError as error:
        if error.errno not in (errno.EACCES, errno.EPERM, errno.ENETUNREACH, errno.ECONNREFUSED):
            raise
    else:
        raise AssertionError('The coding command could reach the loopback listener')
print('case-boundaries-ok')
"""
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = str(listener.getsockname()[1])
            result = actor.server.request("command/exec", {
                "command": [str(actor.python), "-I", "-c", test, str(observation), str(private), str(outside), port],
                "cwd": str(actor.scratch), "permissionProfile": PROFILE, "timeoutMs": 20000,
            })
        if result.get("exitCode") != 0 or result.get("stdout", "").strip() != "case-boundaries-ok":
            raise RuntimeError(f"Native coding sandbox probe failed: {result}")
        observation.unlink()
        private.unlink()
        (actor.scratch / "scratch-probe").unlink()
