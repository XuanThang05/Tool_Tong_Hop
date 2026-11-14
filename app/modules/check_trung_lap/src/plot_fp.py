from __future__ import annotations

import numpy as np

PROBE_VERSION = 2

PROBES = [
    "Người yêu hoặc vợ chồng rời khỏi nhà, căn phòng còn đó nhưng trống trải.",
    "Thế giới bên ngoài vẫn vận hành, nắng mưa xe cộ ngày thường không đổi.",
    "Người ở lại cảm thấy thế giới riêng trở nên trống rỗng sau khi một người biến mất.",
    "Thói quen cũ như nhìn ghế cửa sổ, chờ người đã không còn.",
    "Con cái xa quê nhớ mẹ, mong được đoàn tụ gia đình.",
    "Sống nơi đất khách nhiều năm, lòng thương mẹ không nguôi.",
    "Hôn nhân gượng ép để trả thù, trong lòng vẫn yêu người khác.",
    "Viết thư tình mạo danh người khác, bí mật bị phơi bày.",
    "Chọn trâm định tình trước mặt nhiều quý nữ, gây sốc cho đích tỷ.",
    "Thiên tai bão lũ, sơ tán dân, cứu trợ lương thực.",
    "Thành phố tương lai mất điện, kỹ sư xử lý dữ liệu ký ức.",
    "Săn bắn, ám sát, cứu người bằng cung tên.",
    "Mẹ chồng hiền từ, phu quân bận công vụ, vợ chồng ngủ chung nhưng chưa động phòng.",
    "Tiệc cung đình, bị chị họ chế giễu vì chưa viên phòng.",
    "Giấy nợ hoặc giấy bảo đảm để giữ mạng khi sự thật bị lộ.",
]


def fingerprint(chunk_embs: np.ndarray, probe_embs: np.ndarray, top_k: int = 5) -> np.ndarray:
    if chunk_embs.size == 0:
        return np.zeros(probe_embs.shape[0], dtype=np.float32)
    doc = np.mean(chunk_embs, axis=0)
    n = float(np.linalg.norm(doc))
    if n < 1e-8:
        return np.zeros(probe_embs.shape[0], dtype=np.float32)
    doc = doc / n
    raw = np.clip(doc @ probe_embs.T, 0.0, None)
    out = np.zeros_like(raw, dtype=np.float32)
    k = min(top_k, raw.size)
    idx = np.argpartition(-raw, kth=k - 1)[:k]
    out[idx] = raw[idx]
    n2 = float(np.linalg.norm(out))
    if n2 < 1e-8:
        return out
    return (out / n2).astype(np.float32)


def fingerprint_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    return float(np.dot(a, b))


def dumps(fp: np.ndarray) -> bytes:
    return np.asarray(fp, dtype=np.float32).tobytes()


def loads(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    return np.array(arr, dtype=np.float32, copy=True)


class PlotEncoder:
    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self._probe_embs: np.ndarray | None = None

    def probe_embs(self) -> np.ndarray:
        if self._probe_embs is None:
            self._probe_embs = self.embedder.encode_passages(list(PROBES))
        return self._probe_embs

    def encode(self, chunk_embs: np.ndarray) -> np.ndarray:
        return fingerprint(chunk_embs, self.probe_embs())
