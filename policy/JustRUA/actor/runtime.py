"""Construct an isolated runtime for this episode's native coding tools."""
import json
import os
from pathlib import Path
import shutil
import subprocess

MODEL = "gpt-6-astra"
EFFORT = "medium"
PROFILE = "justrua_coding"
CLI_VERSION = "0.153.4"


def _toml(value):
    if isinstance(value, dict):
        return "{" + ", ".join(json.dumps(k) + " = " + _toml(v) for k, v in value.items()) + "}"
    return json.dumps(value)


def python_runtime():
    """Permit the interpreter, its standard library and generic numeric/image packages."""
    executable = os.environ.get("JUSTRUA_PYTHON")
    if not executable:
        raise RuntimeError("Set JUSTRUA_PYTHON to a separate coding environment with NumPy and Pillow")
    executable = Path(executable).expanduser().absolute()
    # Probe in isolated mode, exactly as the sandbox preflight will import these libraries.
    script = (
        "import json, pathlib, sys, sysconfig, numpy, PIL; "
        "p=pathlib.Path; "
        "packages=[p(numpy.__file__).parent.resolve(),p(PIL.__file__).parent.resolve()]; "
        "print(json.dumps({'stdlib': [sysconfig.get_path('stdlib'), sysconfig.get_path('platstdlib')], "
        "'packages': [str(v) for v in packages], 'prefix': sys.prefix, "
        "'sites': [v for v in sys.path if p(v).name in ('site-packages','dist-packages')]}))"
    )
    try:
        reply = subprocess.run([str(executable), "-I", "-c", script], check=True,
                               capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Set JUSTRUA_PYTHON to a clean Python with NumPy and Pillow installed") from error
    environment = json.loads(reply.stdout)
    roots = {Path(path).resolve() for path in environment["stdlib"]}
    roots.add(executable.resolve())
    # Generic standalone distributions bundle shared libraries and loader metadata
    # beside bin, including files that the OS reads before Python can import anything.
    prefix = executable.parent.parent.resolve()
    if prefix in (Path("/"), Path("/usr"), Path("/usr/local"), Path.home()):
        raise ValueError("JUSTRUA_PYTHON must use a dedicated environment, not an entire system prefix")
    if prefix != Path(environment["prefix"]).resolve() or any(
            not Path(path).is_relative_to(prefix) for path in environment["packages"]):
        raise ValueError("Install NumPy and Pillow directly in the dedicated JUSTRUA_PYTHON environment")
    roots.add(prefix)
    # The base interpreter's stdlib can also contain a host-wide site-packages.
    # Keep unrelated packages out of the filesystem grant for an external venv.
    candidates = {Path(site).resolve() for site in environment["sites"]}
    for path in roots.copy():
        if path.is_dir():
            candidates.update((path / name).resolve() for name in ("site-packages", "dist-packages"))
    denied_sites = {path for path in candidates if path.is_dir() and not path.is_relative_to(prefix)}
    # A venv interpreter can be a symlink. Python needs its own prefix metadata.
    if (executable.parent.parent / "pyvenv.cfg").is_file():
        roots.add((executable.parent.parent / "pyvenv.cfg").resolve())
    return executable, roots, denied_sites


def auth_source():
    account_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    return Path(os.environ.get("JUSTRUA_AUTH_FILE", account_home / "auth.json")).expanduser().resolve()


def native_command(root, scratch, inputs):
    value = os.environ.get("JUSTRUA_CODEX") or shutil.which("codex")
    if not value:
        raise RuntimeError(f"Install Codex CLI {CLI_VERSION} and its matching code-mode host")
    binary = Path(value).expanduser().resolve()
    host = binary.with_name("codex-code-mode-host")
    if not host.is_file() or not os.access(host, os.X_OK):
        raise RuntimeError("JUSTRUA_CODEX must locate the native codex binary beside codex-code-mode-host")
    version = subprocess.check_output([str(binary), "--version"], text=True, timeout=15).strip()
    if version != f"codex-cli {CLI_VERSION}":
        raise RuntimeError(f"Expected codex-cli {CLI_VERSION}, found {version}")
    python, roots, denied_sites = python_runtime()
    bins = [str(python.parent)]
    node_setting = os.environ.get("JUSTRUA_NODE")
    node = (Path(node_setting).expanduser().absolute() if node_setting
            else python.parent.parent.parent / "node/bin/node")
    if node_setting and (not node.is_file() or not os.access(node, os.X_OK)):
        raise ValueError("JUSTRUA_NODE must point to a working standalone node binary")
    if node.is_file():
        # Only the standalone binary is exposed; do not grant its install prefix.
        roots.add(node.absolute())
        roots.add(node.resolve())
        bins.append(str(node.parent))
    read_roots = {str(path): "read" for path in roots}
    read_roots.update({str(path): "deny" for path in denied_sites})
    for path in roots:
        if (path in (Path("/"), Path.home()) or root.is_relative_to(path) or auth_source().is_relative_to(path)
                or Path(__file__).resolve().is_relative_to(path) or path.is_relative_to(Path(__file__).resolve().parents[3])):
            raise ValueError("Use a separate, generic coding Python outside the case and policy checkout")
    # Do not let shell tools inherit credentials, provider proxies or a simulator PYTHONPATH.
    shell_env = {
        "PATH": ":".join([*bins, "/usr/bin", "/bin", "/usr/sbin", "/sbin"]),
        "HOME": str(scratch), "TMPDIR": str(scratch / ".tmp"),
        "OPENSSL_CONF": "/dev/null", "NODE_REPL_HISTORY": str(scratch / ".node_repl_history"),
        "PYTHONNOUSERSITE": "1",
    }
    config = {
        "model": MODEL, "model_provider": "openai", "model_reasoning_effort": EFFORT,
        "approval_policy": "never", "default_permissions": PROFILE,
        f"permissions.{PROFILE}.filesystem": {
            ":minimal": "read", **read_roots, str(scratch): "write", str(inputs): "read",
        },
        f"permissions.{PROFILE}.network.enabled": False,
        "shell_environment_policy.inherit": "none", "shell_environment_policy.set": shell_env,
        "shell_environment_policy.experimental_use_profile": False,
        "sqlite_home": str(root / "state"), "log_dir": str(root / "logs"),
        "project_doc_max_bytes": 0, "web_search": "disabled",
        "agents.enabled": False, "agents.interrupt_message": False,
    }
    enabled = ["shell_tool", "unified_exec", "view_image", "code_mode.enabled", "code_mode_host",
               "skip_host_skill_discovery"]
    disabled = ["apps", "plugins", "hooks", "codex_hooks", "memories", "chronicle",
                "multi_agent", "image_generation", "sleep_tool", "browser_use",
                "browser_use_external", "browser_use_full_cdp_access", "computer_use",
                "in_app_browser", "remote_plugin", "shell_snapshot", "skill_search",
                "skill_mcp_dependency_install", "workspace_dependencies", "goals",
                "code_mode_only", "responses_websockets", "responses_websockets_v2", "fast_mode"]
    config.update({f"features.{name}": True for name in enabled})
    config.update({f"features.{name}": False for name in disabled})
    argv = [str(binary), "app-server", "--stdio", "--strict-config"]
    for key, setting in config.items():
        argv.extend(["-c", key + "=" + _toml(setting)])
    return python, argv


def server_environment(root, auth):
    home = root / "codex-home"
    home.mkdir(mode=0o700)
    if auth:
        source = auth_source()
        if not source.is_file():
            raise RuntimeError("Sign in to Codex or set JUSTRUA_AUTH_FILE to the authorized auth.json")
        (home / "auth.json").symlink_to(source)
    # Only the parent app-server needs the provider connection. It never loads host config/skills.
    env = {key: os.environ[key] for key in (
        "PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTPS_PROXY", "HTTP_PROXY",
        "ALL_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "all_proxy", "no_proxy",
    ) if key in os.environ}
    env.update(HOME=str(root), CODEX_HOME=str(home), ROLLOUT_ISOLATE_HOST_CONTEXT="1",
               ROLLOUT_CODEX_STATE_DIR=str(root / "state"))
    return env
