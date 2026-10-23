#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GoPro + Manus 手套 + Vive Tracker 同步采集入口。"""

import asyncio

from record_ego_app import main


def run():
    try:
        asyncio.run(main("manus"))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
