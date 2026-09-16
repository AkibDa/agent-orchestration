# config.py
import os

IMD_ENABLED = os.environ.get("IMD_ENABLED", "false").lower() in ("true", "1", "yes")
IMD_DISABLED_REASON = "IMD_DISABLED_PENDING_INSTITUTIONAL_APPROVAL"
