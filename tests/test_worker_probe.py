from pathlib import Path


PROBE = Path(__file__).resolve().parents[1] / "examples" / "check_chia_workers.py"


def test_worker_probe_checks_google_adc_readability():
    text = PROBE.read_text(encoding="utf-8")
    assert 'resource_name in {"opencode_creds", "vertex_creds"}' in text
    assert '"adc_readable": os.access(adc, os.R_OK)' in text
    assert 'Google ADC is not readable' in text


def test_worker_probe_checks_teaal_root_and_results_writeability():
    text = PROBE.read_text(encoding="utf-8")
    assert 'resource_name in {"teaal_evolve", "trusted_verification_tool"}' in text
    assert '"teaal_root_exists"' in text
    assert '"runs_root_writable"' in text
    assert 'TeAAL-Evolve root is not visible' in text
    assert 'Trusted results directory is not writable' in text



def test_default_probe_targets_direct_vertex_credentials():
    text = PROBE.read_text(encoding="utf-8")
    assert 'default=["teaal_evolve", "trusted_verification_tool", "vertex_creds"]' in text
