import cv2

def main():
    # Create a trivial gray image via numpy
    import numpy as np
    img = np.full((64, 64, 3), 127, dtype=np.uint8)
    # BGR->GRAY
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_NEAREST)
    # Write (optional): cv2.imwrite("_smoke.png", small)
    assert small.shape == (16, 16)
    print("ok", small.dtype, small.shape)

if __name__ == "__main__":
    main()

