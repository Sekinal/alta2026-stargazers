"""Attack 2: run the organisers' scoring/evaluate.py under several environments on our files.

Cases: the fake test-phase answer (BOM + CRLF truth), the dev answer vs valid.csv, and every
nasty-text roundtrip file from csv_roundtrip.py (answer == truth, so the score must be 1.0).
Environments are command prefixes; the docker one is the real Codabench image if pulled.

Usage: PYTHONPATH=. uv run python redteam/software/scorer_compat.py [env ...]
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
SCORER = ROOT / "scoring" / "evaluate.py"

ENVS = {
    "modern": ["uv", "run", "--no-project", "--with", "pandas", "--with", "scikit-learn", "python"],
    "py38_pd115_sk024": ["uv", "run", "--no-project", "--python", "3.8", "--with", "pandas==1.1.5",
                         "--with", "scikit-learn==0.24.2", "--with", "numpy<1.20", "python"],
    "py38_pd100_sk022": ["uv", "run", "--no-project", "--python", "3.8", "--with", "pandas==1.0.5",
                         "--with", "scikit-learn==0.22.2.post1", "--with", "numpy<1.20", "python"],
    "docker_py37": ["docker", "run", "--rm", "--network=none", "--user", "1000:1000", "-v", "{tmp}:{tmp}", "-v",
                    f"{SCORER.parent}:{SCORER.parent}:ro", "codalab/codalab-legacy:py37", "python"],
}


def cases():
    yield "e2e_test_answer", ROOT / "submissions/test_rt_sw_e2e_fixed+ukg/answer.csv", HERE / "fake/truth.csv"
    yield "dev_answer", ROOT / "submissions/dev_rt_sw_e2e_fixed+ukg/answer.csv", ROOT / "data/valid.csv"
    for p in sorted((HERE / "out_roundtrip").glob("*.csv")):
        yield f"nasty_{p.stem}", p, HERE / "out_roundtrip/plain.csv"  # clean truth: a misparse cannot cancel out


def score(env: str, answer: Path, truth: Path) -> str:
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        tmp = Path(tmp)
        (tmp / "in/res").mkdir(parents=True)
        (tmp / "in/ref").mkdir(parents=True)
        shutil.copy(answer, tmp / "in/res/answer.csv")
        shutil.copy(truth, tmp / "in/ref/truth.csv")
        cmd = [c.replace("{tmp}", str(tmp)) for c in ENVS[env]] + [str(SCORER), str(tmp / "in"), str(tmp / "out")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            return "CRASH " + r.stderr.strip().splitlines()[-1][:150]
        return (tmp / "out/scores.txt").read_text().splitlines()[-1]


def main():
    envs = sys.argv[1:] or list(ENVS)
    if "--versions" in envs:
        envs.remove("--versions")
        for e in envs:
            v = subprocess.run([c for c in ENVS[e] if "{tmp}" not in c and c != f"{SCORER.parent}:{SCORER.parent}:ro"]
                               + ["-c", "import sys,pandas,sklearn,numpy;print(sys.version.split()[0],pandas.__version__,"
                                        "sklearn.__version__,numpy.__version__)"], capture_output=True, text=True)
            print(e, v.stdout.strip() or v.stderr[-300:])
    rows = {}
    for name, a, t in cases():
        rows[name] = {e: score(e, a, t) for e in envs}
        print(f"{name:22} " + " | ".join(f"{e}: {s}" for e, s in rows[name].items()), flush=True)


if __name__ == "__main__":
    main()
