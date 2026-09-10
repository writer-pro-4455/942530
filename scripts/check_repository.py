from pathlib import Path
import json
import py_compile

ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT / "src").glob("*.py"):
    py_compile.compile(str(path), doraise=True)
    print("OK", path.relative_to(ROOT))

json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
json.loads((ROOT / "configs" / "inference_config.json").read_text(encoding="utf-8"))
json.loads((ROOT / "thingsboard" / "dashboard_config.example.json").read_text(encoding="utf-8"))

print("Repository static checks passed.")
