from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("TRUNGLAP_DATA", str(Path(tempfile.mkdtemp(prefix="trunglap-"))))

from src.services import AppService  # noqa: E402


ORIGINAL = (
    "Cô gái bước vào quán cà phê cũ trên phố Nguyễn Huệ và ngồi xuống chiếc ghế gỗ gần cửa sổ. "
    "Cô gọi một ly đen đá, mở cuốn sổ tay đã sờn góc, rồi viết tiếp chương truyện về người lính trở về nhà sau mười năm. "
    "Bên ngoài trời mưa phùn, chiếc xe buýt số tám dừng lại, một người đàn ông cầm bó cúc họa mi bước xuống."
)

REWRITE = (
    "Cô gái đi vào tiệm cà phê cũ ở đường Nguyễn Huệ rồi ngồi ở ghế gỗ bên cửa sổ. "
    "Cô gọi ly đen đá, mở cuốn sổ tay sờn góc và viết tiếp chương về người lính về nhà sau mười năm. "
    "Bên ngoài trời mưa phùn, xe buýt số tám dừng, một đàn ông cầm bó cúc họa mi bước xuống."
)

UNRELATED = (
    "Bão số ba đổ bộ miền Trung trong đêm, hàng nghìn hộ dân được sơ tán lên vùng cao. "
    "Chính quyền địa phương phát lương khô, nước uống và dựng nhà bạt tại nhà văn hóa xã."
)


def main() -> None:
    svc = AppService()
    t0 = time.perf_counter()
    svc.warmup(lambda msg, frac: print(f"  [{frac:.0%}] {msg}"))
    print(f"device={svc.device_label()} load={time.perf_counter() - t0:.1f}s")

    nb1 = svc.create_notebook("Sổ A")
    nb2 = svc.create_notebook("Sổ B")
    svc.add_text(nb1, "Ban goc", ORIGINAL)

    t1 = time.perf_counter()
    r_same = svc.check_text(ORIGINAL, "copy", notebook_id=nb1, warn=70, block=90)
    r_rw = svc.check_text(REWRITE, "xao", notebook_id=nb1, warn=70, block=90)
    r_un = svc.check_text(UNRELATED, "khac", notebook_id=nb1, warn=70, block=90)
    r_cross = svc.check_text(REWRITE, "cheo", notebook_id=nb2, warn=70, block=90)
    r_all = svc.check_text(REWRITE, "all", notebook_id=None, warn=70, block=90)
    print(f"checks={time.perf_counter() - t1:.2f}s")
    for name, r in [
        ("nguyen-ban", r_same),
        ("viet-lai", r_rw),
        ("khong-lien-quan", r_un),
        ("trong-so-B-trong", r_cross),
        ("cheo-so", r_all),
    ]:
        top = r.matches[0] if r.matches else None
        extra = f"{top.score}% {top.kind} <- {top.title}" if top else "no-match"
        print(f"{name}: verdict={r.verdict} overall={r.overall_score} {extra} ({r.elapsed:.2f}s)")

    para = (
        "Nắng chiều rơi trên mái ngói cũ. Người đàn bà ngồi bên hiên, chải tóc, nhớ đứa con trai ra thành phố. "
        "Trong bếp nồi canh chua sôi lăn tăn. Con mèo vàng nhảy lên bậu cửa."
    )
    long = "\n".join(f"Đoạn {i}. {para}" for i in range(1, 280))
    print("long_chars", len(long))
    t2 = time.perf_counter()
    svc.add_text(nb1, "Truyen dai", long)
    print(f"add_50k={time.perf_counter() - t2:.2f}s")
    long_rw = long.replace("Người đàn bà", "Người phụ nữ").replace("con trai", "đứa con").replace("mèo vàng", "mèo vàng nhỏ")
    t3 = time.perf_counter()
    r_long = svc.check_text(long_rw, "dai-xao", notebook_id=nb1, warn=70, block=90)
    print(
        f"check_50k={time.perf_counter() - t3:.2f}s score={r_long.overall_score} "
        f"verdict={r_long.verdict} kind={r_long.matches[0].kind if r_long.matches else '-'}"
    )
    svc.close()
    assert r_same.overall_score >= 90
    assert r_rw.overall_score >= 70
    assert r_un.overall_score < 55
    assert r_cross.overall_score == 0
    assert r_all.overall_score >= 70
    print("SMOKE OK")


if __name__ == "__main__":
    main()
