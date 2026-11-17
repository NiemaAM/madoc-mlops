"""python -m mlops.web   # http://0.0.0.0:7870 (MLOPS_UI_PORT), login from MLOPS_AUTH ("name:password,...")"""

import logging
import os

import uvicorn

from mlops.web.api import create_app


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # one process: the monitor's samples live in it; the access log is off (one line per poll, nothing useful)
    uvicorn.run(create_app(), host=os.environ.get("MLOPS_UI_HOST", "0.0.0.0"),
                port=int(os.environ.get("MLOPS_UI_PORT", 7870)), workers=1, access_log=False, log_level="info")


if __name__ == "__main__":
    main()
