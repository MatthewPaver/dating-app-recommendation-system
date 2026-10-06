from pathlib import Path


def test_cli_install_does_not_require_the_notebook_stack():
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "requirements-cli.txt").read_text().lower()
    assert all(name in requirements for name in ("numpy", "pandas", "scipy", "scikit-learn", "pytest"))
    assert not any(name in requirements for name in ("jupyter", "matplotlib", "seaborn", "ipykernel"))
    makefile = (root / "Makefile").read_text()
    assert "demo: install-cli" in makefile
    assert "test: install-cli" in makefile
