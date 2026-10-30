from pathlib import Path

import yaml


CLUSTER_CONFIG = Path(__file__).resolve().parents[1] / "cluster" / "local_gemini.yaml"


def _config():
    return yaml.safe_load(CLUSTER_CONFIG.read_text(encoding="utf-8"))


def test_ray_head_uses_container_safe_temp_directory():
    config = _config()
    commands = config["head_start_ray_commands"]
    assert any("ray stop --force" in command for command in commands)
    assert any("mkdir -p /tmp/ctv-${USER}" in command for command in commands)
    assert any(
        "--head" in command
        and "--temp-dir=/tmp/ctv-${USER}" in command
        for command in commands
    )
    assert any(
        "--include-dashboard=True" in command
        and "--dashboard-host=127.0.0.1" in command
        and "--dashboard-port=8265" in command
        and "--dashboard-agent-listen-port=52365" in command
        for command in commands
    )
    worker_commands = config["worker_start_ray_commands"]
    assert any("--dashboard-agent-listen-port=0" in command for command in worker_commands)


def test_trusted_worker_does_not_edit_host_checkout_metadata():
    config = _config()
    trusted = config["available_node_types"]["trusted_project"]
    setup = trusted["worker_setup_commands"]
    assert not any("pip install -e ${CHIA_PROJECT_ROOT}" in command for command in setup)
    assert any("/tmp/chia-trusted-verification-install" in command for command in setup)
    assert any("pip install --no-cache-dir /tmp/chia-trusted-verification-install" in command for command in setup)


def test_worker_containers_use_writable_python_temp_directory():
    config = _config()
    for worker_name in ("trusted_project", "opencode_vertex"):
        options = config["available_node_types"][worker_name]["docker"]["run_options"]
        assert "-e TMPDIR=/tmp" in options
        assert "-e TMP=/tmp" in options
        assert "-e TEMP=/tmp" in options


def test_opencode_worker_installs_project_package_for_tool_deserialization():
    config = _config()
    opencode = config["available_node_types"]["opencode_vertex"]
    setup = opencode["worker_setup_commands"]
    docker = opencode["docker"]
    options = docker["run_options"]

    assert docker["container_name"] == "chia-ctv-opencode-${USER}"
    assert any("/tmp/chia-trusted-verification-install" in command for command in setup)
    assert any(
        "cp -a /opt/chia-trusted-verification/pyproject.toml" in command
        for command in setup
    )
    assert any(
        "pip install --no-cache-dir /tmp/chia-trusted-verification-install" in command
        for command in setup
    )
    assert "-v ${CHIA_PROJECT_ROOT}:/opt/chia-trusted-verification:ro" in options


def test_all_project_workers_use_fixed_container_mount_path():
    config = _config()
    for worker_name in ("trusted_project", "opencode_vertex"):
        worker = config["available_node_types"][worker_name]
        options = worker["docker"]["run_options"]
        setup = worker["worker_setup_commands"]
        assert "-v ${CHIA_PROJECT_ROOT}:/opt/chia-trusted-verification:ro" in options
        assert any("/opt/chia-trusted-verification/pyproject.toml" in command for command in setup)


def test_remote_workers_use_noneditable_project_install():
    config = _config()
    for worker_name in ("trusted_project", "opencode_vertex"):
        setup = config["available_node_types"][worker_name]["worker_setup_commands"]
        assert not any("pip install -e" in command for command in setup)
        assert any(
            "pip install --no-cache-dir /tmp/chia-trusted-verification-install" in command
            for command in setup
        )


def test_google_adc_is_staged_via_private_docker_volume():
    config = _config()
    init = config["initialization_commands"]
    assert any("docker volume create" in command and "chia-ctv-gcloud-${USER}" in command for command in init)
    assert any('RAY_UID="$(id -u ray)"' in command and 'RAY_GID="$(id -g ray)"' in command for command in init)
    for worker_name in ("trusted_project", "opencode_vertex"):
        options = config["available_node_types"][worker_name]["docker"]["run_options"]
        assert "-v chia-ctv-gcloud-${USER}:/home/ray/.config/gcloud:ro" in options
        assert not any("${HOME}/.config/gcloud:/home/ray/.config/gcloud" in option for option in options)


def test_trusted_worker_preserves_host_paths_and_writable_results_overlay():
    config = _config()
    init = config["initialization_commands"]
    assert any('mkdir -p "${CHIA_PROJECT_ROOT}/runs"' in command for command in init)
    assert any('chmod 0777 "${CHIA_PROJECT_ROOT}/runs"' in command for command in init)
    trusted = config["available_node_types"]["trusted_project"]
    options = trusted["docker"]["run_options"]
    assert "-v ${CHIA_PROJECT_ROOT}:${CHIA_PROJECT_ROOT}:ro" in options
    assert "-v ${CHIA_PROJECT_ROOT}/runs:${CHIA_PROJECT_ROOT}/runs" in options
    assert "-e CHIA_PROJECT_ROOT=${CHIA_PROJECT_ROOT}" in options

