import argparse

import cv2

from box_detector import solve_lr, solve_the_frame


VERTICAL_HALF_HEIGHT = 30


def draw_result(frame, distances):
    height, width = frame.shape[:2]
    cx = int(width * 0.43)
    cy = int(height * 0.11)
    y_start = max(0, cy - VERTICAL_HALF_HEIGHT)
    y_end = min(height, cy + VERTICAL_HALF_HEIGHT + 1)
    left_distance, right_distance = distances

    result = frame.copy()
    cv2.line(result, (cx, 0), (cx, height - 1), (255, 0, 0), 1)
    cv2.circle(result, (cx, cy), 4, (0, 0, 255), -1)

    label = f"left: {left_distance}px  right: {right_distance}px"
    cv2.putText(result, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(result, "Press q to quit", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    return result


def draw_mask_result(mask, distances):
    height, width = mask.shape[:2]
    cx = int(width * 0.43)
    cy = int(height * 0.11)
    y_start = max(0, cy - VERTICAL_HALF_HEIGHT)
    y_end = min(height, cy + VERTICAL_HALF_HEIGHT + 1)
    left_distance, right_distance = distances

    result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(result, (0, y_start), (width - 1, y_end - 1),
                  (255, 200, 0), 1)
    cv2.circle(result, (cx, cy), 4, (0, 0, 255), -1)

    if left_distance > 0:
        cv2.rectangle(result, (cx - left_distance, y_start),
                      (cx - 1, y_end - 1), (0, 255, 0), 2)
    if right_distance > 0:
        cv2.rectangle(result, (cx + 1, y_start),
                      (cx + right_distance, y_end - 1), (0, 255, 0), 2)

    label = f"left: {left_distance}px  right: {right_distance}px"
    cv2.putText(result, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Display left/right black-area distances from a camera."
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="camera index, default: 0")
    parser.add_argument("--color", choices=("orange", "purple"),
                        default="orange", help="detector color")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read frame")
                break

            mask = solve_the_frame(frame, args.color)
            distances = solve_lr(mask)
            print(f"left={distances[0]} px, right={distances[1]} px",
                  end="\r", flush=True)

            cv2.imshow("left-right distance", draw_result(frame, distances))
            cv2.imshow("binary mask", draw_mask_result(mask, distances))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()
