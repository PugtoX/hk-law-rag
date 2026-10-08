"""Verify the shipped build_index.py parsing logic against a real page.
chromadb / sentence_transformers are stubbed — we only exercise collect()."""
import importlib.util
import shutil
import sys
from pathlib import Path
from unittest import mock

# stub the heavy deps so importing build_index works anywhere
for name in ("chromadb", "sentence_transformers"):
    sys.modules.setdefault(name, mock.MagicMock())

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("bi", ROOT / "build_index.py")
bi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bi)

tmp = ROOT / "_verify_data"
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir()
shutil.copy(ROOT / "samples" / "cap57b.htm", tmp / "cap57b.htm")
bi.DATA = tmp

items = bi.collect()
print(f"\ncollect() -> {len(items)} items")
for it in items:
    print(f"  [{it['topic']}] Q{it['qno']}: {it['question'][:60]}")
    assert it["answer"] and it["question"], "empty field"
    assert "回到問題" not in it["answer"], "back-link not stripped"

print("\nlast answer tail:", repr(items[-1]["answer"][-60:]))
print("\nALL ASSERTIONS PASSED")
shutil.rmtree(tmp, ignore_errors=True)
