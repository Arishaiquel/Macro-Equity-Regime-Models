from __future__ import annotations

import os

from redis import Redis
from rq import Connection, Queue, Worker


listen = ["quant-jobs"]
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def main() -> None:
    conn = Redis.from_url(redis_url)
    with Connection(conn):
        worker = Worker([Queue(name) for name in listen])
        worker.work()


if __name__ == "__main__":
    main()
