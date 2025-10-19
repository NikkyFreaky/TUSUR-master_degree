from pathlib import Path

import cv2 as cv
import numpy as np


def process_video(
    input_path: str,
    output_path: str,
    lower_rgb: tuple,
    upper_rgb: tuple,
    min_area: int,
    kernel_size: int = 5,
):
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Не удалось открыть {input_path}")
        return

    fps = cap.get(cv.CAP_PROP_FPS)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # ядро свёртки для размытия (равномерное)
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mask = cv.inRange(rgb, np.array(lower_rgb), np.array(upper_rgb))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask)

        region_mask = np.zeros((h, w), dtype=np.uint8)

        for i in range(1, n_labels):
            x, y, w_box, h_box, area = stats[i]
            if area >= min_area:
                cv.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv.rectangle(region_mask, (x, y), (x + w_box, y + h_box), 255, -1)

        # выполняем размытие СВЁРТКОЙ
        blurred = cv.filter2D(frame, -1, kernel)

        # превращаем маску в 3 канала
        region_mask_3ch = cv.merge([region_mask, region_mask, region_mask])

        # комбинируем: внутри рамок — оригинал, вне — свёртка
        result = np.where(region_mask_3ch == 255, frame, blurred)

        out.write(result)

    cap.release()
    out.release()
    print(f"Сохранено в {output_path}")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    process_video(
        input_path="input/orange_1080p_120s.mp4",
        output_path="output/orange_1080p_120s_blur.mp4",
        lower_rgb=(214, 117, 13),
        upper_rgb=(255, 197, 93),
        min_area=500,
        kernel_size=5,  # можно менять 3, 5, 7, 9 — сильнее размытие
    )
