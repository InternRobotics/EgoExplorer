#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GoPro + Wuji 手套（含触觉）+ Vive Tracker 同步采集入口。"""

import asyncio

from record_ego_app import main


def run():
    try:
        asyncio.run(main("wuji"))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
