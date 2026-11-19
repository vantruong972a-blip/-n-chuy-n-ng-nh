"""Tải ảnh MINH HỌA dòng xe từ Wikimedia Commons (chỉ ảnh giấy phép tự do), ghi tác giả và giấy phép.

Chạy trên máy CÓ Internet (không chạy được trong môi trường bị chặn mạng):
    python scripts/fetch_car_images.py --top 120          # 120 dòng xe nhiều tin nhất
    python scripts/fetch_car_images.py --only toyota/vios # chỉ một dòng
    python scripts/fetch_car_images.py --dry-run          # chỉ in truy vấn, không tải
    python scripts/fetch_car_images.py --drop toyota/vios # xóa ảnh/dòng ghi nguồn bị chọn sai

Thứ tự ưu tiên nguồn ảnh: (1) Wikimedia Commons giấy phép tự do; (2) chỉ khi không có, mới dùng link ảnh tự chọn:
    python scripts/fetch_car_images.py --add kia/rondo --url https://.../anh.jpg --author "..." --license "..."
        (thử Wikimedia trước; thêm --force-url để dùng thẳng link)
    hoặc ghi sẵn nhiều link vào Data/images/manual_urls.csv (cột model,url,author,license); lần chạy thường sẽ chỉ
    dùng các link này cho dòng xe mà Wikimedia không có ảnh.

Kết quả:
    Data/images/<hãng>/<dòng>.jpg|png     ảnh 640 px
    Data/images/CREDITS.csv               hãng, dòng, file, tiêu đề, tác giả, giấy phép, link giấy phép, link trang ảnh
    Data/images/review.html               trang xem lại toàn bộ ảnh + nguồn để kiểm tra bằng mắt

Lưu ý quan trọng:
- Đây là ảnh minh họa của DÒNG xe, không phải ảnh của chiếc xe trong tin rao, và có thể khác đời/phiên bản.
  Giao diện luôn ghi rõ điều này kèm tác giả, giấy phép, link nguồn.
- Chương trình chọn ảnh bằng bộ lọc (giấy phép, tiêu đề, kích thước) chứ không "nhìn" ảnh, nên CÓ THỂ chọn sai.
  Hãy mở review.html, thấy ảnh nào sai thì chạy --drop hãng/dòng.
- Tuân thủ chính sách của Wikimedia: User-Agent mô tả, giới hạn tốc độ, không tải hàng loạt song song.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "CarPriceMLOpsStudentProject/1.0 (educational project; image illustrations with attribution)"
THUMB_WIDTH = 640
MIN_WIDTH = 800                      # ảnh gốc tối thiểu, để thumbnail 640 px không bị phóng to
RELAXED = False                      # --relaxed: nới lỏng tên liền kề, mô tả ảnh, tỷ lệ, số kết quả (giấy phép và từ khóa tiêu đề vẫn giữ)
ASPECT_RANGE = (1.2, 2.3)            # ảnh ngang (ngoại thất xe), loại ảnh dọc/vuông
CREDIT_FIELDS = ["brand", "model", "file", "title", "author", "license", "license_url", "source_url"]

# Giấy phép tự do cho phép dùng kèm ghi công. Loại NC (phi thương mại) và ND (không chỉnh sửa).
LICENSE_OK = re.compile(r"^(CC[ -]BY(?:[ -]SA)?[ -]\d(?:\.\d)?(?:[ -][A-Za-z]{2,3})?|CC0(?:[ -]1\.0)?|Public domain|PD[- ].*)$", re.I)
LICENSE_BAD = re.compile(r"\b(NC|ND)\b", re.I)
BAD_WORDS = re.compile(
    r"\b(interior|engine|dashboard|cockpit|logo|badge|emblem|wheel|rim|headlight|taillight|tail light|detail|seat|steering|trunk|boot|"
    r"sketch|drawing|illustration|render|crash|accident|wreck|burnt|toy|model car|scale model|diecast|lego|poster|brochure|advertis|"
    r"sign|plate|key|remote|gearbox|speedometer|odometer|chassis|undercarriage|taxi|police|ambulance|fire|"
    r"handle|door handle|frost|frosty|frozen|ice|icy|snow flake|mirror|grille|grill|bumper|lamp|hood|bonnet|antenna|fuel cap|"
    r"window|windscreen|windshield|roof rack|sticker|close-up|closeup|macro|part|parts|spoiler|exhaust|tyre|tire|pedal|lever|knob|button|display|screen|badge|"
    r"motorcycle|motorbike|bike|rally|racing|race|dtm|wrc|concept|prototype|limousine)\b", re.I)

BAD_DESC = re.compile(
    r"\b(interior|engine|dashboard|cockpit|wheel|rim|headlight|taillight|tail light|seat|steering|handle|frost|frozen|"
    r"motorcycle|motorbike|rally|racing|race car|dtm|wrc|concept car|prototype|close-up|closeup|macro)\b", re.I)


def slug(s: str) -> str:
    t = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")


def tokens(s: str) -> List[str]:
    t = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return [x for x in re.split(r"[^a-z0-9]+", t) if x]


def display_name(brand: str, model: str) -> str:
    """'mercedes-benz','c' -> 'Mercedes-Benz C'; dùng cho truy vấn tìm kiếm."""
    b = "-".join(p.capitalize() for p in brand.split("-"))
    return f"{b} {model.upper() if len(model) <= 3 else model.title()}"


def build_query(brand: str, model: str) -> str:
    return f"{display_name(brand, model)} filetype:bitmap"


def strip_html(v: Any) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", str(v or "")))).strip()


def license_ok(short: str) -> bool:
    short = (short or "").strip()
    return bool(short) and bool(LICENSE_OK.match(short)) and not LICENSE_BAD.search(short)


def meta(info: Dict[str, Any], key: str) -> str:
    return strip_html((info.get("extmetadata") or {}).get(key, {}).get("value", ""))


def candidate_ok(page: Dict[str, Any], brand: str, model: str) -> Optional[Dict[str, Any]]:
    """Trả thông tin ảnh nếu đạt mọi bộ lọc, ngược lại None."""
    infos = page.get("imageinfo") or []
    if not infos:
        return None
    info = infos[0]
    if info.get("mime") not in ("image/jpeg", "image/png"):
        return None
    w, h = int(info.get("width") or 0), int(info.get("height") or 0)
    if w < MIN_WIDTH or h <= 0 or not ((1.0 if RELAXED else ASPECT_RANGE[0]) <= w / h <= (2.8 if RELAXED else ASPECT_RANGE[1])):
        return None
    lic = meta(info, "LicenseShortName")
    if not license_ok(lic):
        return None
    title = str(page.get("title", ""))
    cats = meta(info, "Categories")
    desc = meta(info, "ImageDescription")
    # Tên hãng và dòng phải đứng liền nhau (vd "mazda 323", "mazda2"), không khớp rời rạc ở chỗ khác trong tiêu đề.
    sep = r"[\s\-]*"
    pat = r"\b" + sep.join(re.escape(t) for t in tokens(brand) + tokens(model)) + r"\b"
    norm = " ".join(tokens(title)) + " | " + " ".join(tokens(cats))
    if RELAXED:   # chỉ cần có đủ các từ của hãng và dòng trong tiêu đề hoặc danh mục
        if not all(t in set(tokens(title + " " + cats)) or t in norm.replace(" ", "") for t in tokens(brand) + tokens(model)):
            return None
    elif not re.search(pat, norm):
        return None
    if BAD_WORDS.search(title + " " + cats):
        return None
    if not RELAXED and BAD_DESC.search(desc):   # mô tả ảnh cũng cho biết ảnh chụp chi tiết, xe đua hay xe trưng bày ý tưởng
        return None
    thumb = info.get("thumburl") or info.get("url")
    if not thumb:
        return None
    return {
        "title": title, "thumb_url": thumb, "mime": info["mime"], "author": meta(info, "Artist") or "Không rõ tác giả",
        "license": lic, "license_url": meta(info, "LicenseUrl"), "source_url": info.get("descriptionurl", ""),
        "rank": int(page.get("index", 9999)), "desc": desc,
    }


def pick_candidate(api_json: Dict[str, Any], brand: str, model: str) -> Optional[Dict[str, Any]]:
    pages = list(((api_json or {}).get("query") or {}).get("pages", {}).values())
    pages.sort(key=lambda p: int(p.get("index", 9999)))
    for p in pages:
        c = candidate_ok(p, brand, model)
        if c:
            return c
    return None


# ---------------- mạng (tách riêng để kiểm thử bằng phản hồi giả) ----------------
def fetch(url: str, retries: int = 3, timeout: int = 40) -> bytes:
    last: Optional[Exception] = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in (429, 500, 502, 503, 504):
                break
        except Exception as e:  # lỗi mạng tạm thời
            last = e
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"Không tải được {url}: {last}")


def _search_one(query: str) -> Dict[str, Any]:
    q = {"action": "query", "format": "json", "generator": "search", "gsrnamespace": "6", "gsrsearch": query,
         "gsrlimit": "50" if RELAXED else "20", "prop": "imageinfo", "iiprop": "url|mime|size|extmetadata", "iiurlwidth": str(THUMB_WIDTH),
         "iiextmetadatafilter": "Artist|LicenseShortName|LicenseUrl|ImageDescription|Categories"}
    return json.loads(fetch(API + "?" + urllib.parse.urlencode(q)).decode("utf-8"))


def search(brand: str, model: str) -> Dict[str, Any]:
    res = _search_one(build_query(brand, model))
    if len(model) <= 3 or model.isdigit():
        # Tên dòng ngắn thường được viết liền trên Wikimedia ("Mazda2", "BMW X5"), tìm thêm cách viết liền.
        time.sleep(1.0)
        extra = _search_one(f"{display_name(brand, model).replace(' ', '')} filetype:bitmap")
        pages = ((res.get("query") or {}).setdefault("pages", {}))
        for k, pg in (((extra or {}).get("query") or {}).get("pages", {})).items():
            pg = dict(pg)
            pg["index"] = int(pg.get("index", 0)) + 100
            pages.setdefault(f"x{k}", pg)
    return res


# ---------------- dữ liệu ra ----------------
def read_credits(path: Path) -> Dict[tuple, Dict[str, str]]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {(r["brand"], r["model"]): r for r in csv.DictReader(f)}


def write_credits(path: Path, rows: Dict[tuple, Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CREDIT_FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow({c: rows[k].get(c, "") for c in CREDIT_FIELDS})


def write_review(out_dir: Path, rows: Dict[tuple, Dict[str, str]]) -> None:
    cards = []
    for (b, m), r in sorted(rows.items()):
        if not (out_dir / r["file"]).exists():
            continue
        cards.append(
            f'<figure><img src="{html.escape(r["file"])}" loading="lazy" alt=""><figcaption><b>{html.escape(b)} / {html.escape(m)}</b><br>'
            f'{html.escape(r["title"])}<br>Tác giả: {html.escape(r["author"])} · {html.escape(r["license"])}<br>'
            f'<a href="{html.escape(r["source_url"])}" target="_blank" rel="noreferrer">Trang ảnh</a> · '
            f'<code>python scripts/fetch_car_images.py --drop {html.escape(b)}/{html.escape(m)}</code></figcaption></figure>')
    (out_dir / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Duyệt ảnh minh họa</title>'
        "<style>body{font:14px system-ui;margin:24px;background:#f6f7f9}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}"
        "figure{margin:0;background:#fff;border:1px solid #dde;border-radius:8px;overflow:hidden}img{width:100%;aspect-ratio:3/2;object-fit:cover;display:block}"
        "figcaption{padding:10px;line-height:1.5}code{font-size:12px;word-break:break-all}</style>"
        "<h1>Duyệt ảnh minh họa dòng xe</h1><p>Ảnh nào <b>không đúng dòng xe</b> thì chạy lệnh --drop bên dưới ảnh đó.</p>"
        f"<main>{''.join(cards)}</main>", encoding="utf-8")


def top_models(clean_csv: Path, top: int, min_listings: int) -> List[tuple]:
    import pandas as pd
    d = pd.read_csv(clean_csv, usecols=["brand", "model"], low_memory=False).dropna()
    d["brand"] = d["brand"].astype(str).str.lower().str.strip()
    d["model"] = d["model"].astype(str).str.lower().str.strip()
    c = d.value_counts().reset_index(name="n")
    c = c[(c["n"] >= min_listings) & (~c["model"].isin({"other", "khac", "khác"}))].head(top)
    return [(r.brand, r.model) for r in c.itertuples()]


def add_manual(args: argparse.Namespace, rows: Dict[tuple, Dict[str, str]], out: Path, cred_path: Path, fetch_fn,
               target: Optional[str] = None, url: Optional[str] = None, author: Optional[str] = None,
               license_: Optional[str] = None) -> Dict[str, int]:
    """Thêm ảnh do người dùng tự chọn bằng đường dẫn ảnh (không qua Wikimedia). Giấy phép do người thêm tự chịu trách nhiệm."""
    brand, _, model = (target or args.add).lower().partition("/")
    url = (url or args.url or "").strip()
    author = author if author is not None else getattr(args, "author", None)
    license_ = license_ if license_ is not None else getattr(args, "license", None)
    if not brand or not model or not re.match(r"^https://", url, re.I):
        print("Cần --add hãng/dòng và --url bắt đầu bằng https://")
        return {"error": 1}
    data = fetch_fn(url)
    if not data or len(data) > args.max_bytes * 4:
        print(f"Ảnh rỗng hoặc quá lớn ({len(data) if data else 0} byte, tối đa {args.max_bytes * 4})")
        return {"error": 1}
    if data[:3] == b"\xff\xd8\xff":
        ext = ".jpg"
    elif data[:8] == b"\x89PNG\r\n\x1a\n":
        ext = ".png"
    else:
        print("Đường dẫn này không phải file ảnh JPEG hoặc PNG (có thể là trang web, hãy dùng 'Sao chép địa chỉ hình ảnh').")
        return {"error": 1}
    rel = f"{slug(brand)}/{slug(model)}{ext}"
    (out / slug(brand)).mkdir(parents=True, exist_ok=True)
    (out / rel).write_bytes(data)
    rows[(brand, model)] = {"brand": brand, "model": model, "file": rel, "title": urllib.parse.unquote(url.rsplit("/", 1)[-1])[:120],
                            "author": author or "", "license": license_ or "Chưa xác minh (ảnh do nhóm tự thêm)",
                            "license_url": "", "source_url": url}
    write_credits(cred_path, rows)
    write_review(out, rows)
    print(f"  + {brand}/{model}: đã thêm từ {url}")
    return {"ok": 1}


def read_manual_urls(path: Path) -> Dict[tuple, Dict[str, str]]:
    """Đọc danh sách link ảnh dự phòng: cột model (hãng/dòng), url, author, license. Chỉ dùng khi Wikimedia không có ảnh."""
    if not path.exists():
        return {}
    out: Dict[tuple, Dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            b, _, m = (r.get("model") or "").strip().lower().partition("/")
            if b and m and (r.get("url") or "").strip():
                out[(b, m)] = {k: (v or "").strip() for k, v in r.items()}
    return out


def try_wikimedia(brand: str, model: str, args, rows, out: Path, cred_path: Path, fetch_fn, search_fn) -> bool:
    """Tìm ảnh giấy phép tự do trên Wikimedia. Trả True nếu đã lưu được ảnh."""
    cand = pick_candidate(search_fn(brand, model), brand, model)
    if not cand:
        return False
    data = fetch_fn(cand["thumb_url"])
    if not data or len(data) > args.max_bytes:
        raise RuntimeError(f"ảnh rỗng hoặc quá lớn ({len(data) if data else 0} byte)")
    ext = ".png" if cand["mime"] == "image/png" else ".jpg"
    rel = f"{slug(brand)}/{slug(model)}{ext}"
    (out / slug(brand)).mkdir(parents=True, exist_ok=True)
    (out / rel).write_bytes(data)
    rows[(brand, model)] = {"brand": brand, "model": model, "file": rel, "title": cand["title"], "author": cand["author"],
                            "license": cand["license"], "license_url": cand["license_url"], "source_url": cand["source_url"]}
    write_credits(cred_path, rows)          # ghi dần để chạy lại được nếu bị ngắt
    print(f"  + {brand}/{model}: {cand['title']} ({cand['license']}) [Wikimedia]")
    return True


def run(args: argparse.Namespace, fetch_fn=None, search_fn=None, sleep=time.sleep) -> Dict[str, int]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cred_path = out / "CREDITS.csv"
    rows = read_credits(cred_path)
    fetch_fn = fetch_fn or fetch
    search_fn = search_fn or search

    if args.drop:
        b, _, m = args.drop.partition("/")
        r = rows.pop((b.lower(), m.lower()), None)
        if r and (out / r["file"]).exists():
            (out / r["file"]).unlink()
        write_credits(cred_path, rows)
        write_review(out, rows)
        print("Đã xóa" if r else "Không có dòng này trong CREDITS.csv", f"{b}/{m}")
        return {"dropped": int(bool(r))}

    if getattr(args, "add", None):
        b, _, m = args.add.lower().partition("/")
        if not getattr(args, "force_url", False):
            try:
                if try_wikimedia(b, m, args, rows, out, cred_path, fetch_fn, search_fn):
                    write_review(out, rows)
                    return {"ok": 1}
            except Exception as e:
                print(f"  ! Wikimedia lỗi ({e}), chuyển sang link ảnh")
            print(f"  Wikimedia không có ảnh đạt bộ lọc cho {b}/{m} → dùng link ảnh bạn đưa")
        return add_manual(args, rows, out, cred_path, fetch_fn)

    pairs = [tuple(x.lower().split("/", 1)) for x in args.only] if args.only else top_models(Path(args.clean), args.top, args.min_listings)
    stats = {"ok": 0, "skipped": 0, "not_found": 0, "error": 0}
    manual = read_manual_urls(Path(args.manual_file) if getattr(args, "manual_file", None) else out / "manual_urls.csv")
    for brand, model in pairs:
        key = (brand, model)
        if key in rows and (out / rows[key]["file"]).exists() and not args.refresh:
            stats["skipped"] += 1
            continue
        if args.dry_run:
            print(build_query(brand, model))
            continue
        try:
            if try_wikimedia(brand, model, args, rows, out, cred_path, fetch_fn, search_fn):
                stats["ok"] += 1
            elif key in manual:                      # chỉ khi Wikimedia không có ảnh đạt bộ lọc
                mu = manual[key]
                r = add_manual(args, rows, out, cred_path, fetch_fn, target=f"{brand}/{model}", url=mu["url"],
                               author=mu.get("author"), license_=mu.get("license"))
                stats["ok" if r.get("ok") else "error"] += 1
            else:
                print(f"  - {brand}/{model}: không có ảnh đạt bộ lọc")
                stats["not_found"] += 1
        except Exception as e:
            print(f"  ! {brand}/{model}: {e}")
            stats["error"] += 1
        sleep(args.delay)
    if args.dry_run:
        print("Chạy thử (--dry-run): chưa tải gì.")
    else:
        write_review(out, rows)
        print(f"Xong: {stats}. Mở {out / 'review.html'} để kiểm tra bằng mắt.")
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default=str(ROOT / "Data" / "processed" / "clean.csv"))
    ap.add_argument("--out", default=str(ROOT / "Data" / "images"))
    ap.add_argument("--top", type=int, default=120, help="số dòng xe nhiều tin nhất")
    ap.add_argument("--min-listings", type=int, default=30)
    ap.add_argument("--only", nargs="*", help="chỉ các dòng dạng hãng/dòng, ví dụ toyota/vios")
    ap.add_argument("--delay", type=float, default=1.0, help="giây nghỉ giữa các yêu cầu")
    ap.add_argument("--max-bytes", type=int, default=800_000)
    ap.add_argument("--relaxed", action="store_true", help="nới lỏng bộ lọc (tên liền kề, mô tả, tỷ lệ, 50 kết quả); PHẢI xem lại review.html")
    ap.add_argument("--add", help="thêm ảnh tự chọn cho một dòng xe: hãng/dòng (dùng cùng --url)")
    ap.add_argument("--url", help="địa chỉ https của file ảnh (jpg/png) cho --add")
    ap.add_argument("--force-url", action="store_true", help="với --add: bỏ qua Wikimedia, dùng thẳng --url")
    ap.add_argument("--manual-file", help="CSV link ảnh dự phòng (cột model,url,author,license); mặc định Data/images/manual_urls.csv")
    ap.add_argument("--author", help="tác giả hoặc nguồn ảnh, ghi vào CREDITS.csv")
    ap.add_argument("--license", help="giấy phép ảnh, ghi vào CREDITS.csv")
    ap.add_argument("--refresh", action="store_true", help="tải lại cả dòng đã có ảnh")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--drop", help="xóa ảnh đã chọn sai, dạng hãng/dòng")
    args = ap.parse_args(argv)
    global RELAXED
    RELAXED = bool(args.relaxed)
    return 0 if run(args) is not None else 1


if __name__ == "__main__":
    sys.exit(main())
