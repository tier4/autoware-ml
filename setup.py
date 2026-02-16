"""Setup entrypoint for Autoware-ML with compiled extensions."""

from setuptools import setup

# #region agent log
def _log(msg: str, hypothesis_id: str, **data: object) -> None:
    import json
    with open("/home/kokseang/autoware-ml/.cursor/debug.log", "a") as f:
        f.write(json.dumps({"location": "setup.py", "message": msg, "hypothesisId": hypothesis_id, "data": data, "timestamp": __import__("time").time_ns() // 1_000_000}) + "\n")
# #endregion

# #region agent log
_log("setup.py executed during build", "H2")
# #endregion
try:
    # #region agent log
    _log("before import autoware_ml.ops.build", "H1")
    # #endregion
    from autoware_ml.ops.build import get_cmdclass, get_ext_modules
    # #region agent log
    _log("import succeeded", "H1", has_ext=True)
    # #endregion
    ext_modules = get_ext_modules()
    cmdclass = get_cmdclass()
except ModuleNotFoundError as e:
    # #region agent log
    _log("import skipped (isolated build), using empty ext_modules", "H1", error=str(e), runId="fix")
    # #endregion
    ext_modules = []
    cmdclass = {}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
