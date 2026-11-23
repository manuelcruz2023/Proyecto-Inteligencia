import threading
import time

from src.inference import StreamAnalyzer


class SessionManager:
    """Un StreamAnalyzer por cámara (session_id). Las sesiones inactivas caducan (TTL)."""

    def __init__(self, predictor, ttl=120):
        self.predictor = predictor
        self.ttl = ttl
        self._sessions = {}
        self._lock = threading.Lock()

    def get(self, sid):
        now = time.time()
        with self._lock:
            for k in [k for k, (_, ts) in self._sessions.items() if now - ts > self.ttl]:
                del self._sessions[k]
            if sid not in self._sessions:
                self._sessions[sid] = [StreamAnalyzer(self.predictor), now]
            entry = self._sessions[sid]
            entry[1] = now
            return entry[0]

    def reset(self, sid):
        with self._lock:
            self._sessions.pop(sid, None)