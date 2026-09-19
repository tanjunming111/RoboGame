import cv2
import numpy as np


# HSV thresholds calibrated for the two block colors.
LOWER_ORANGE = np.array([9, 0, 110])
UPPER_ORANGE = np.array([35, 170, 255])
LOWER_PURPLE = np.array([101, 16, 57])
UPPER_PURPLE = np.array([140, 154, 255])

# Detection and filtering parameters.
MORPH_KERNEL_SIZE = 5
MIN_BLOCK_AREA = 300
MIN_BLOCK_WIDTH = 20
MIN_BLOCK_HEIGHT = 10
BLUR_SIZE = 5


def _get_color_range(color):
    """Return the HSV range for a supported block color."""
    if color == "orange":
        return LOWER_ORANGE, UPPER_ORANGE
    if color == "purple":
        return LOWER_PURPLE, UPPER_PURPLE
    raise ValueError(f"Unsupported block color: {color}")


def _make_mask(frame, color):
    """Create a denoised binary mask for the requested block color."""
    blurred = cv2.GaussianBlur(frame, (BLUR_SIZE, BLUR_SIZE), 1)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower, upper = _get_color_range(color)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def detect_block(frame, color, min_area=MIN_BLOCK_AREA, show=False):
    """Detect a colored block and estimate its image-space center.

    The horizontal center is computed from the left and right boundaries of
    the visible colored region. Therefore, the upper part may be hidden as
    long as both side boundaries of the lower part remain visible.
    """
    if frame is None or frame.size == 0:
        return None

    mask = _make_mask(frame, color)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        if (area >= min_area and width >= MIN_BLOCK_WIDTH
                and height >= MIN_BLOCK_HEIGHT):
            candidates.append((area, contour, x, y, width, height))

    if not candidates:
        return None

    area, contour, x, y, width, height = max(
        candidates,
        key=lambda candidate: candidate[0],
    )

    # Extract the two visible side edges from the selected color component.
    component_mask = np.zeros_like(mask)
    cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)
    occupied_columns = np.where(np.any(component_mask > 0, axis=0))[0]
    if occupied_columns.size == 0:
        return None

    left_edge_x = int(occupied_columns[0])
    right_edge_x = int(occupied_columns[-1])
    center_x = (left_edge_x + right_edge_x) / 2.0

    moments = cv2.moments(contour)
    if moments["m00"]:
        center_y = moments["m01"] / moments["m00"]
    else:
        center_y = y + height / 2.0

    result = {
        "detected": True,
        "center_x": float(center_x),
        "center_y": float(center_y),
        "left_edge_x": left_edge_x,
        "right_edge_x": right_edge_x,
        "area": float(area),
        "bbox": (x, y, width, height),
        "mask": mask,
    }

    if show:
        display = frame.copy()
        cv2.rectangle(display, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.line(display, (left_edge_x, y),
                 (left_edge_x, y + height), (255, 0, 0), 2)
        cv2.line(display, (right_edge_x, y),
                 (right_edge_x, y + height), (255, 0, 0), 2)
        cv2.circle(display, (round(center_x), round(center_y)), 5, (0, 0, 255), -1)
        cv2.imshow("block_detection", display)

    return result


def pd_down(frame, crs, al=0):
    """Backward-compatible boolean API used by existing control code."""
    return detect_block(frame, crs) is not None


def main():
    """Run a standalone visual test for orange-block detection."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera 0")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detection = detect_block(frame, "orange", show=True)
        if detection is not None:
            print(
                f"center=({detection['center_x']:.1f}, "
                f"{detection['center_y']:.1f}), "
                f"edges=({detection['left_edge_x']}, "
                f"{detection['right_edge_x']})"
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
