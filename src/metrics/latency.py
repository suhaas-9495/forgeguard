from dataclasses import dataclass
from time import perf_counter

@dataclass(slots = True)
class LatencyResult:
    """
    Stores latency information for a model response.
    """

    started_at: float
    finished_at: float
    latency_ms: float
    
class LatencyTracker:
    """"
    just keeping a track on inference latency
    
    """
    def __init__(self):
        self._start = None

    def start(self):

        self._start = perf_counter()

    def stop(self) -> LatencyResult:

        end = perf_counter()

        latency_ms = (
            end - self._start
        ) * 1000

        return LatencyResult(
            started_at=self._start,
            finished_at=end,
            latency_ms=round(
                latency_ms,
                2,
            ),
        )