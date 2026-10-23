import hashlib
import io
import zipfile
from pathlib import Path

from PIL import Image

EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def adjust_image(image):
    w, h = image.size
    size = (1024, int(h / w * 1024)) if w > h else (int(w / h * 1024), 1024)
    image = image.resize(size, Image.Resampling.LANCZOS)
    cw, ch = (v // 16 * 16 for v in size)
    left, top = (size[0] - cw) // 2, (size[1] - ch) // 2
    return image.crop((left, top, left + cw, top + ch)).convert("RGB")


def zoom_crop(image, scale, x_frac, y_frac):

    w, h = image.size
    cw, ch = max(16, int(w * scale) // 16 * 16), max(16, int(h * scale) // 16 * 16)
    left = int((w - cw) * x_frac)
    top = int((h - ch) * y_frac)
    return image.crop((left, top, left + cw, top + ch)).resize((w, h), Image.Resampling.LANCZOS)


def read_rows(source):
    source = Path(source)
    rows = []
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                cap = path.with_suffix(".txt")
                rows.append((str(path.relative_to(source)), path.read_bytes(), cap.read_text() if cap.exists() else None))
    else:
        with zipfile.ZipFile(source) as z:
            names = z.namelist()
            for name in sorted(names):
                if name.lower().endswith(EXTENSIONS):
                    cap = str(Path(name).with_suffix(".txt"))
                    rows.append((name, z.read(name), z.read(cap).decode("utf-8") if cap in names else None))
    if not rows:
        raise ValueError("Empty image dataset")
    return rows


def load_dataset(source, trigger_word=None):
    items = []
    for name, raw, caption in read_rows(source):
        if not caption or not caption.strip():
            caption = trigger_word or "image"
        image = adjust_image(Image.open(io.BytesIO(raw)))
        digest = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
        items.append({"name": name, "image": image, "caption": caption, "sha256": digest})
    return items
