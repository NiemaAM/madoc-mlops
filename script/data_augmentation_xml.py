"""Data augmentation of PageXML pages: appearance (noise, lighting, shadow, blur, scribbles) then geometry (rotation,
perspective), with the coordinates of the PageXML transformed by the same matrix as the image.

    python script/data_augmentation_xml.py --input pages/ --output augmented/ --copies 2 --seed 42

--input   a folder with the PageXML files and their images side by side (as in <data>/pages)
--output  the augmented pages are written there, also side by side: <name>_aug<k>.<ext> and <name>_aug<k>.xml,
          the XML pointing to its new image (imageFilename) with its new size
--copies  augmented versions of every page (default 1)
--seed    the same seed gives the same augmented pages (the pipeline can rebuild them exactly)

The augmentations and their ranges are the ones of the previous version of this script (CONFIGURATION below).
Every PAGE version is read (the namespace is taken from each file).
"""

import argparse
import hashlib
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from augraphy import Scribbles, ShadowCast

# ============================================================
# CONFIGURATION
# ============================================================

USE_NOISE = True
USE_BLUR = True
USE_LIGHTING = True
USE_SHADOW = True
USE_ROTATION = True
USE_PERSPECTIVE = True
USE_SCRIBBLES = True

NOISE_MIN, NOISE_MAX = 1, 50                    # Gaussian noise (standard deviation)
BLUR_MIN, BLUR_MAX = 0, 10                      # 0-100 scale
ROTATION_MIN, ROTATION_MAX = -5, 5              # degrees
PERSPECTIVE_MIN, PERSPECTIVE_MAX = -5, 5        # percent displacement of the corners
LIGHTING_MIN, LIGHTING_MAX = 0, 70              # 0-100 scale
SHADOW_MIN, SHADOW_MAX = 0, 70                  # 0-100 scale
SCRIBBLES_MIN, SCRIBBLES_MAX = 0, 5             # number of scribbles

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"


# ============================================================
# XML POINTS
# ============================================================

def parse_points(points_string):
    """"623,0 613,0 603,0" -> [(623.0, 0.0), (613.0, 0.0), (603.0, 0.0)]"""
    points = []
    for point in (points_string or "").strip().split():
        try:
            x, y = point.split(",")
            points.append((float(x), float(y)))
        except ValueError:
            continue
    return points


def points_to_string(points):
    return " ".join(f"{int(round(x))},{int(round(y))}" for x, y in points)


def transform_points(points, matrix, width, height):
    """A 3x3 matrix applied to the points, clipped to the new image."""
    if not points:
        return []
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
    return [(max(0, min(width - 1, float(x))), max(0, min(height - 1, float(y)))) for x, y in transformed]


# ============================================================
# PAGE XML
# ============================================================

def namespace_of(root):
    return root.tag[1:].split("}", 1)[0] if root.tag.startswith("{") else ""


def transform_page_xml(input_xml, output_xml, matrix, new_width, new_height, image_name):
    """The PageXML with all its Coords and Baselines moved by `matrix`, its new size and its new image."""
    tree = ET.parse(input_xml)
    root = tree.getroot()
    namespace = namespace_of(root)
    ET.register_namespace("", namespace)      # written as the default namespace (no "ns0:")
    ET.register_namespace("xsi", XSI_NAMESPACE)
    ns = {"pc": namespace} if namespace else {}
    prefix = "pc:" if namespace else ""
    page = root.find(f".//{prefix}Page", ns)
    if page is None:
        raise ValueError(f"no <Page> element in {input_xml}")
    page.set("imageFilename", image_name)
    page.set("imageWidth", str(new_width))
    page.set("imageHeight", str(new_height))
    for tag in ("Coords", "Baseline"):
        for element in root.findall(f".//{prefix}{tag}", ns):
            points = parse_points(element.get("points"))
            if points:
                element.set("points", points_to_string(transform_points(points, matrix, new_width, new_height)))
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)


# ============================================================
# APPEARANCE
# ============================================================

def add_noise(image, strength):
    """Gaussian noise."""
    if strength <= 0:
        return image
    noise = np.random.normal(0, strength, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_blur(image, strength):
    """Gaussian blur, strength 0-100 -> kernel 1-31 (odd)."""
    strength = max(0, min(100, int(strength)))
    if strength == 0:
        return image
    kernel_size = 1 + int(strength * 30 / 100)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def add_lighting_gradient(image, strength):
    """A smooth light spot at a random place."""
    if strength <= 0:
        return image
    height, width = image.shape[:2]
    xx, yy = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    cx, cy = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
    gradient = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2)) / 0.5)
    change = gradient * 80 * (strength / 100.0)
    result = image.astype(np.float32) + change[:, :, np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)


def add_shadow(image, strength):
    """A random shadow (Augraphy ShadowCast)."""
    if strength <= 0:
        return image
    opacity_min = max(0.0, (strength / 100.0) * 0.3)
    opacity_max = max(opacity_min, strength / 100.0)
    shadow = ShadowCast(shadow_side="random", shadow_vertices_range=(1, 20), shadow_width_range=(0.3, 0.8),
                        shadow_height_range=(0.3, 0.8), shadow_color=(0, 0, 0),
                        shadow_opacity_range=(opacity_min, opacity_max), shadow_iterations_range=(1, 2),
                        shadow_blur_kernel_range=(101, 301), p=1.0)
    return shadow(image)


def add_scribbles(image, count):
    """Random scribbles (Augraphy Scribbles)."""
    if count <= 0:
        return image
    # not "random": Augraphy's "highlighter" ink always fails with Python >= 3.12 (it passes a float to
    # random.randint); the other inks are drawn here instead
    ink = random.choice(["pencil", "pen", "marker"])
    scribbles = Scribbles(scribbles_type="lines", scribbles_ink=ink, scribbles_location="random",
                          scribbles_size_range=(250, 600), scribbles_count_range=(count, count),
                          scribbles_thickness_range=(1, 3), scribbles_brightness_change=[32, 64, 128],
                          scribbles_skeletonize=0, scribbles_skeletonize_iterations=(2, 3),
                          scribbles_color="random", scribbles_text="random", scribbles_text_font="random",
                          scribbles_text_rotate_range=(0, 360), scribbles_lines_stroke_count_range=(1, 6), p=1.0)
    return scribbles(image)


# ============================================================
# GEOMETRY
# ============================================================

def rotate_image(image, angle):
    """-> rotated image (canvas enlarged, white borders), 3x3 matrix."""
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix_2x3 = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix_2x3[0, 0]), abs(matrix_2x3[0, 1])
    new_width, new_height = int(height * sin + width * cos), int(height * cos + width * sin)
    matrix_2x3[0, 2] += new_width / 2 - center[0]
    matrix_2x3[1, 2] += new_height / 2 - center[1]
    rotated = cv2.warpAffine(image, matrix_2x3, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    return rotated, np.vstack([matrix_2x3, [0, 0, 1]])


def perspective_image(image, top_left, top_right, bottom_left, bottom_right):
    """-> image with its corners moved (percent of the size), 3x3 matrix."""
    height, width = image.shape[:2]
    src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    dst = np.float32([
        [width * top_left[0] / 100, height * top_left[1] / 100],
        [width - 1 + width * top_right[0] / 100, height * top_right[1] / 100],
        [width - 1 + width * bottom_right[0] / 100, height - 1 + height * bottom_right[1] / 100],
        [width * bottom_left[0] / 100, height - 1 + height * bottom_left[1] / 100],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    transformed = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(255, 255, 255))
    return transformed, matrix


# ============================================================
# ONE IMAGE
# ============================================================

def augment_image(image):
    """Appearance first, then geometry -> (augmented image, combined 3x3 matrix of the geometry)."""
    result = image.copy()
    if USE_NOISE:
        result = add_noise(result, random.randint(NOISE_MIN, NOISE_MAX))
    if USE_LIGHTING:
        result = add_lighting_gradient(result, random.randint(LIGHTING_MIN, LIGHTING_MAX))
    if USE_SHADOW:
        result = add_shadow(result, random.randint(SHADOW_MIN, SHADOW_MAX))
    if USE_BLUR:
        result = add_blur(result, random.randint(BLUR_MIN, BLUR_MAX))
    if USE_SCRIBBLES:
        result = add_scribbles(result, random.randint(SCRIBBLES_MIN, SCRIBBLES_MAX))
    combined = np.eye(3, dtype=np.float64)
    if USE_ROTATION:
        result, rotation = rotate_image(result, random.uniform(ROTATION_MIN, ROTATION_MAX))
        combined = rotation @ combined
    if USE_PERSPECTIVE:
        corner = lambda: (random.uniform(PERSPECTIVE_MIN, PERSPECTIVE_MAX),   # noqa: E731
                          random.uniform(PERSPECTIVE_MIN, PERSPECTIVE_MAX))
        result, perspective = perspective_image(result, corner(), corner(), corner(), corner())
        combined = perspective @ combined
    return result, combined


def seed_for(name, copy, seed):
    """The same page, copy and seed always give the same random numbers."""
    return int(hashlib.sha256(f"{seed}:{copy}:{name}".encode()).hexdigest()[:8], 16)


def augment_with_retries(image, name, copy, seed, attempts=3):
    """augment_image with the seed of the page; Augraphy sometimes fails on a random draw (e.g. its ink generator
    with Python >= 3.12): the next draws are tried before giving up on the copy."""
    for attempt in range(attempts):
        value = seed_for(name, copy, seed if attempt == 0 else f"{seed}-retry{attempt}")
        random.seed(value)
        np.random.seed(value)
        try:
            return augment_image(image)
        except Exception as e:   # noqa: BLE001 - a bug of one augmentation must not stop the dataset
            error = e
    raise RuntimeError(f"augmentation failed {attempts} times ({type(error).__name__}: {error})")


def image_of(xml_path):
    """The image of a PageXML: its imageFilename next to it, else a file with the same stem."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    page = next((el for el in root.iter() if el.tag.rsplit("}", 1)[-1] == "Page"), None)
    name = Path((page.get("imageFilename") if page is not None else "") or "").name
    if name and (xml_path.parent / name).is_file():
        return xml_path.parent / name
    return next((xml_path.with_suffix(ext) for ext in IMAGE_EXTENSIONS if xml_path.with_suffix(ext).is_file()), None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    pages = sorted(args.input.glob("*.xml"))
    print(f"Found {len(pages)} pages.")
    done = failed = 0
    for index, xml_path in enumerate(pages, start=1):
        image_path = image_of(xml_path)
        image = cv2.imread(str(image_path)) if image_path else None
        if image is None:
            print(f"WARNING [{index}/{len(pages)}] no readable image for {xml_path.name}")
            failed += 1
            continue
        for copy in range(1, args.copies + 1):
            stem = f"{xml_path.stem}_aug{copy}"
            out_image = args.output / f"{stem}{image_path.suffix.lower()}"
            try:
                augmented, matrix = augment_with_retries(image, xml_path.stem, copy, args.seed)
                if not cv2.imwrite(str(out_image), augmented):
                    raise OSError(f"could not write {out_image.name}")
                height, width = augmented.shape[:2]
                transform_page_xml(xml_path, args.output / f"{stem}.xml", matrix, width, height, out_image.name)
                done += 1
            except Exception as e:
                print(f"ERROR [{index}/{len(pages)}] {xml_path.name} copy {copy}: {e}")
                out_image.unlink(missing_ok=True)
                failed += 1
        print(f"[{index}/{len(pages)}] {xml_path.name}")
    print(f"DONE: {done} augmented pages in {args.output}, {failed} failed")
    return 0 if done or not pages else 1


if __name__ == "__main__":
    sys.exit(main())
