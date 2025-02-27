import cv2 as cv
import numpy as np
import numpy.typing as npt
import random


def scale(img: npt.NDArray) -> npt.NDArray:
    rows, cols, _ = img.shape
    M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 0, 1.25)
    return cv.warpAffine(img, M, (cols, rows))


def rotate(img: npt.NDArray) -> npt.NDArray:
    rows, cols, _ = img.shape

    degree = random.randint(20, 340)

    M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), degree, 1)
    return cv.warpAffine(img, M, (cols, rows))


def blur(img: npt.NDArray) -> npt.NDArray:
    blur = random.randint(5, 15)

    return cv.blur(img, (blur, blur))


def contrast(img: npt.NDArray) -> npt.NDArray:
    return cv.addWeighted(img, 2, img, 0, -100)


def exposure(img: npt.NDArray) -> npt.NDArray:
    return cv.addWeighted(img, .5, img, 0, 100)


def random_perspective_shift(points, scale) -> npt.NDArray:
    pts = points.copy()

    for i in range(0, len(pts)):
        # (rand - .5) * 2 is calculated to get random signed values
        pts[i:] += (np.random.rand(2) - .5) * 2 * scale

    return pts.astype(np.float32)


def project(img: npt.NDArray, scale) -> npt.NDArray:
    rows, cols, _ = img.shape

    row5 = rows / 5
    col5 = cols / 5
    pts1 = np.array([[row5, col5],
                     [row5, col5 + col5],
                     [row5 + row5, col5]],
                    dtype=np.float32)
    pts2 = random_perspective_shift(pts1, scale)

    M = cv.getAffineTransform(pts1, pts2)
    return cv.warpAffine(img, M, (cols, rows))


def augment(img: npt.NDArray) -> dict[str, npt.NDArray]:
    return {
        "scaled": scale(img),
        "rotated": rotate(img),
        "blurred": blur(img),
        "contrast": contrast(img),
        "exposed": exposure(img),
        "projected": project(img, 10)
    }


def _get_files(path, recursive=False) -> set[str]:
    """Retrieve file paths from a directory"""
    import os

    file_paths = set()

    if recursive:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                file_paths.add(os.path.join(dirpath, f))
    else:
        for entry in os.scandir(path):
            if entry.is_file():
                file_paths.add(entry.path)

    return file_paths


def _save_augments(filename, augments, output_dir=None):
    import os

    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    root, ext = os.path.splitext(basename)

    target_dir = dirname if output_dir is None else output_dir

    print("Writing", filename, "augmentation to", target_dir)

    for label, data in augments.items():
        output = os.path.join(target_dir, root + '_' + label + ext)
        # print("output", output)  # DEBUG
        cv.imwrite(output, data)


def _parse_cmd_arguments():
    from argparse import ArgumentParser
    import os

    parser = ArgumentParser(
        prog="Augmentor",
        description="Augment image(s)"
    )

    parser.add_argument("path", nargs='+')
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("-r", "--recursive", action="store_true")

    args = parser.parse_args()

    # Check if output_dir is valid
    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        if not os.path.isdir(args.output_dir):
            raise AssertionError(
                "specified output " + args.output_dir + " is not a directory.")

    # Retrieve path of all files
    files = set()
    for path in args.path:
        if os.path.isdir(path):
            files |= _get_files(path, args.recursive)
        else:
            files.add(path)

    return args, files


def _main():
    args, files = _parse_cmd_arguments()

    for f in files:
        img = cv.imread(f)

        if img is None:
            print("Skipping file", f)
            continue

        _save_augments(f, augment(img), args.output_dir)


if __name__ == "__main__":
    _main()
