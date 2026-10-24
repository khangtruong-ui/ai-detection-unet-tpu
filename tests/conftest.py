"""Global pytest fixtures and timeout configuration."""

import os
import pytest

# Ensure tests execute in foreground without accidental detachment
os.environ["SID_FOREGROUND"] = "1"
os.environ["_SID_DETACHED"] = "1"
