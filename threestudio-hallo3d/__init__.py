import threestudio
from packaging.version import Version

if not (
    hasattr(threestudio, "__version__")
    and Version(threestudio.__version__) >= Version("0.2.0")
):
    raise ValueError(
        "threestudio version must be >= 0.2.0, please update threestudio"
    )

from .guidance import hallo3d_sd_guidance
