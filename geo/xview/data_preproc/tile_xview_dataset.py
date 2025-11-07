import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm


def tile_image_with_boxes(image_path, boxes, tile_size=768, overlap=0.33):
    """Tile a single image and adjust bounding boxes.

    Ensures all tiles are full-sized by snapping edge tiles back to create
    full coverage rather than having partially black tiles.

    Args:
        image_path: Path to source image
        boxes: List of bounding boxes in [x1, y1, x2, y2, class_id] format
        tile_size: Size of output tiles
        overlap: Overlap ratio (0.33 = 33%)

    Returns:
        List of (tile_image, tile_boxes, tile_offset) tuples
    """
    img = Image.open(image_path)
    width, height = img.size

    stride = int(tile_size * (1 - overlap))
    tiles = []

    # Calculate tile positions ensuring full coverage with no partial tiles
    x_positions = []
    x = 0
    while x < width:
        x_positions.append(x)
        x += stride
        # If we'd overshoot, snap the last tile to the right edge
        if x + tile_size > width and x < width:
            x_positions.append(width - tile_size)
            break

    y_positions = []
    y = 0
    while y < height:
        y_positions.append(y)
        y += stride
        # If we'd overshoot, snap the last tile to the bottom edge
        if y + tile_size > height and y < height:
            y_positions.append(height - tile_size)
            break

    # Remove duplicates while preserving order
    x_positions = list(dict.fromkeys(x_positions))
    y_positions = list(dict.fromkeys(y_positions))

    for y in y_positions:
        for x in x_positions:
            # Ensure we don't go out of bounds
            x_start = max(0, min(x, width - tile_size))
            y_start = max(0, min(y, height - tile_size))
            x_end = x_start + tile_size
            y_end = y_start + tile_size

            # Crop tile (always full tile_size x tile_size)
            tile = img.crop((x_start, y_start, x_end, y_end))

            # Adjust boxes for this tile
            tile_boxes = []
            for box in boxes:
                bx1, by1, bx2, by2, class_id = box

                # Check if box intersects with tile
                if bx2 <= x_start or bx1 >= x_end or by2 <= y_start or by1 >= y_end:
                    continue

                # Clip box to tile boundaries and adjust to tile coordinates
                new_x1 = max(0, bx1 - x_start)
                new_y1 = max(0, by1 - y_start)
                new_x2 = min(tile_size, bx2 - x_start)
                new_y2 = min(tile_size, by2 - y_start)

                # Only include if box is substantial (>10% area remains)
                orig_area = (bx2 - bx1) * (by2 - by1)
                new_area = (new_x2 - new_x1) * (new_y2 - new_y1)
                if new_area > 0.1 * orig_area:
                    tile_boxes.append([new_x1, new_y1, new_x2, new_y2, class_id])

            tiles.append((tile, tile_boxes, (x_start, y_start)))

    return tiles


def process_image(args):
    """Process a single image - for parallel execution."""
    image_path, boxes, output_dir, tile_size, overlap = args

    tiles = tile_image_with_boxes(image_path, boxes, tile_size, overlap)

    image_name = image_path.stem
    saved_tiles = []

    for idx, (tile, tile_boxes, offset) in enumerate(tiles):
        # Save tile
        tile_name = f"{image_name}_tile_{idx:04d}.png"
        tile_path = output_dir / "images" / tile_name
        tile.save(tile_path)

        # Convert boxes to YOLO format (normalized xywh)
        yolo_boxes = []
        tile_w, tile_h = tile.size
        for box in tile_boxes:
            x1, y1, x2, y2, class_id = box
            x_center = ((x1 + x2) / 2) / tile_w
            y_center = ((y1 + y2) / 2) / tile_h
            w = (x2 - x1) / tile_w
            h = (y2 - y1) / tile_h
            yolo_boxes.append(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # Save labels
        label_path = output_dir / "labels" / f"{image_name}_tile_{idx:04d}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_boxes))

        saved_tiles.append(tile_name)

    return image_name, len(saved_tiles)


def tile_xview_dataset(
    images_dir,
    geojson_file,
    output_dir,
    tile_size=768,
    overlap=0.33,
    max_workers=4
):
    """Tile XView dataset into smaller chips.

    Args:
        images_dir: Directory containing source images (.tif files)
        geojson_file: Path to xView GeoJSON file (e.g., xView_train.geojson)
        output_dir: Output directory for tiled dataset
        tile_size: Size of output tiles
        overlap: Overlap ratio
        max_workers: Number of parallel workers
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    geojson_file = Path(geojson_file)

    # Create output directories
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Load GeoJSON labels
    print(f"Loading GeoJSON from {geojson_file}...")
    with open(geojson_file, 'r') as f:
        geojson_data = json.load(f)

    # Group boxes by image_id
    image_boxes = {}
    for feature in geojson_data['features']:
        props = feature['properties']
        image_id = props['image_id'].replace('.tif', '')  # Remove .tif extension
        bounds_str = props['bounds_imcoords']  # "xmin,ymin,xmax,ymax"
        type_id = props['type_id']

        # Parse bounds from string "xmin,ymin,xmax,ymax"
        bounds = [float(x) for x in bounds_str.split(',')]

        if image_id not in image_boxes:
            image_boxes[image_id] = []

        # Store as [x1, y1, x2, y2, type_id]
        # Keep original type_id (11-94) - remapping happens at model level
        image_boxes[image_id].append([
            bounds[0], bounds[1], bounds[2], bounds[3], type_id
        ])

    print(f"Loaded labels for {len(image_boxes)} images")
    print(f"Loading images from {images_dir}...")

    tasks = []
    for img_path in images_dir.glob("*.tif"):
        image_id = img_path.stem  # e.g., "100" from "100.tif"

        if image_id not in image_boxes:
            print(f"Warning: No labels found for {img_path.name}, skipping")
            continue

        boxes = image_boxes[image_id]
        tasks.append((img_path, boxes, output_dir, tile_size, overlap))

    print(f"Processing {len(tasks)} images with {max_workers} workers...")

    total_tiles = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures)):
            image_name, num_tiles = future.result()
            total_tiles += num_tiles

    print(f"Created {total_tiles} tiles from {len(tasks)} images")
    print(f"Output saved to {output_dir}")
