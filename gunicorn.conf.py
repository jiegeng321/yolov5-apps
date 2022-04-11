#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
from prometheus_client import multiprocess
timeout = 5000

def child_exit(server, worker):
    multiprocess.mark_process_dead(worker.pid)
