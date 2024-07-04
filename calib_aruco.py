"""
This script extracts camera extrinsics and intrinsics using COLMAP.

The directory structure of the root directory should be as follows:
root_dir
|   calib
|   |   *cam*
|   |   |   image_*.jpg

Run the script using:
    python calib_aruco.py -r <root_dir>

The parameters will be stored in <root_dir>/calib/params.txt.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from argparse import ArgumentParser
from glob import glob

import numpy as np
from numpy.lib import recfunctions as rf

from python_utils.feature_utils import add_features, exhaustive_match
from python_utils.file_utils import create_dir
from python_utils.image_utils import get_images
from python_utils.point_utils import points3D_to_ply

logging.getLogger().setLevel(logging.INFO)


debug = False

# Find temporary directory cross-platform
tmp_dir = os.path.join(tempfile.gettempdir(), "brics_calib")


def main(args):
    db_path = os.path.join(tmp_dir, "db.db")
    if args.separate_calib:
        input_image_path = os.path.join(args.root_dir, "image", "synced")
    elif args.no_subdir:
        input_image_path = args.root_dir
    else:
        input_image_path = os.path.join(args.root_dir, "calib")

    image_path = os.path.join(tmp_dir, "images")
    create_dir(image_path)

    image_dirs = list(sorted(glob(f"{input_image_path}/*cam*")))
    for image_dir in image_dirs:
        to_copy = os.path.basename(sorted(glob(f"{image_dir}/*.jpg"))[0])
        os.makedirs(os.path.join(image_path, os.path.basename(image_dir)))
        shutil.copyfile(
            os.path.join(image_dir, to_copy),
            os.path.join(image_path, os.path.basename(image_dir), to_copy),
        )

    if os.path.exists(db_path):
        logging.warning("Previous database found, deleting.")
        os.remove(db_path)
        try:
            os.remove(os.path.join(tmp_dir, "db.db-wal"))
            os.remove(os.path.join(tmp_dir, "db.db-shm"))
        except FileNotFoundError:
            pass

    logging.info("Importing images in database")
    subprocess.run(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            db_path,
            "--image_path",
            image_path,
            "--ImageReader.single_camera_per_folder",
            "1",
            "--ImageReader.camera_model",
            "OPENCV",
        ]
    )

    image_ids, image_paths = get_images(args, db_path, input_image_path)
    image_id_desc = add_features(image_ids, image_paths, db_path, debug)
    exhaustive_match(
        image_ids, image_paths, db_path, args, image_id_desc, input_image_path, tmp_dir
    )

    logging.info("Performing geometric verification")
    subprocess.run(
        [
            "colmap",
            "matches_importer",
            "--database_path",
            db_path,
            "--match_list_path",
            f"{os.path.join(tmp_dir, 'match_list.txt')}",
            "--match_type",
            "pairs",
            "--TwoViewGeometry.min_num_inliers",
            "1",  # SiftMatching.min_num_inliers
        ]
    )

    logging.info("Reconstructing")
    recon_path = os.path.join(tmp_dir, "reconstruction")
    create_dir(recon_path)
    subprocess.run(
        [
            "colmap",
            "mapper",
            "--database_path",
            db_path,
            "--image_path",
            image_path,
            "--output_path",
            recon_path,
        ]
    )

    logging.info("Performing global bundle adjustment")
    subprocess.run(
        [
            "colmap",
            "bundle_adjuster",
            "--input_path",
            os.path.join(recon_path, "0"),
            "--output_path",
            os.path.join(recon_path, "0"),
            "--BundleAdjustment.refine_principal_point",
            "1",
            "--BundleAdjustment.max_num_iterations",
            "1000",
        ]
    )

    output_dir = os.path.join(tmp_dir, "output")
    create_dir(output_dir)
    subprocess.run(
        [
            "colmap",
            "model_converter",
            "--input_path",
            os.path.join(recon_path, "0"),
            "--output_path",
            output_dir,
            "--output_type",
            "TXT",
        ]
    )
    # Read images
    image_params = []
    with open(os.path.join(output_dir, "images.txt")) as f:
        skip_next = False
        for line in f.readlines():
            if skip_next:
                skip_next = False
                continue
            if line.startswith("#"):
                continue
            data = line.split()
            param = []
            param.append(int(data[8]))
            param.append(data[9].split("/")[0])
            param += [float(datum) for datum in data[1:8]]
            image_params.append(tuple(param))
            skip_next = True

    images = np.array(
        image_params,
        dtype=[
            ("cam_id", int),
            ("cam_name", "<U22"),
            ("qvecw", float),
            ("qvecx", float),
            ("qvecy", float),
            ("qvecz", float),
            ("tvecx", float),
            ("tvecy", float),
            ("tvecz", float),
        ],
    )

    # Read cameras
    cam_params = []
    with open(os.path.join(output_dir, "cameras.txt")) as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            data = line.split()
            param = []
            param.append(int(data[0]))
            param.append(int(data[2]))
            param.append(int(data[3]))
            param += [float(datum) for datum in data[4:]]
            cam_params.append(tuple(param))
    cameras = np.array(
        cam_params,
        dtype=[
            ("cam_id", int),
            ("width", int),
            ("height", int),
            ("fx", float),
            ("fy", float),
            ("cx", float),
            ("cy", float),
            ("k1", float),
            ("k2", float),
            ("p1", float),
            ("p2", float),
        ],
    )

    img_cams = rf.join_by("cam_id", cameras, images)
    print("Number of cameras detected:" + str(len(cameras)))
    if args.separate_calib:
        create_dir(os.path.join(args.root_dir, "calib"))
    save_path = os.path.join(args.root_dir, "calib", "params.txt")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.savetxt(save_path, img_cams, fmt="%s", header=" ".join(img_cams.dtype.fields))

    logging.warning(
        f"Stored the parameters at {os.path.join(args.root_dir, 'calib', 'params.txt')}"
    )

    points3D_to_ply(
        os.path.join(output_dir, "points3D.txt"),
        os.path.join(args.root_dir, "calib", "point_cloud.ply"),
    )

    sparse_path = os.path.join(recon_path, "0")
    dense_path = os.path.join(tmp_dir, "dense")
    logging.info("Undistorting images")
    subprocess.run(
        [
            "colmap",
            "image_undistorter",
            "--image_path",
            image_path,
            "--input_path",
            sparse_path,
            "--output_path",
            dense_path,
            "--output_type",
            "COLMAP",
            "--max_image_size",
            "2000",
        ]
    )

    if os.path.exists("build"):
        shutil.rmtree("build")
    logging.info("Performing stereo matching")
    subprocess.run(
        [
            "colmap",
            "patch_match_stereo",
            "--workspace_path",
            dense_path,
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true",
        ]
    )

    logging.info("Fusing depth maps")
    subprocess.run(
        [
            "colmap",
            "stereo_fusion",
            "--workspace_path",
            dense_path,
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            os.path.join(dense_path, "fused.ply"),
        ]
    )

    logging.info("Creating a mesh from the dense point cloud using poisson")
    subprocess.run(
        [
            "colmap",
            "poisson_mesher",
            "--input_path",
            os.path.join(dense_path, "fused.ply"),
            "--output_path",
            os.path.join(dense_path, "meshed-poisson.ply"),
        ]
    )

    logging.info("Creating a mesh from the dense point cloud using delaunay")
    subprocess.run(
        [
            "colmap",
            "delaunay_mesher",
            "--input_path",
            os.path.join(dense_path, "fused.ply"),
            "--output_path",
            os.path.join(dense_path, "meshed-delaunay.ply"),
        ]
    )

    # copy ply results
    fused_path = os.path.join(args.root_dir, "calib", "dense.ply")
    shutil.copyfile(
        os.path.join(dense_path, "fused.ply"),
        fused_path,
    )
    print(f"Saved to {fused_path}")
    poisson_path = os.path.join(args.root_dir, "calib", "meshed-poisson.ply")
    shutil.copyfile(
        os.path.join(dense_path, "meshed-poisson.ply"),
        poisson_path
    )
    print(f"Saved to {poisson_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--root_dir",
        default="/dev/hdd/BRICS/brics-studio/2024-06-30_session_snapshot",
        help="Base directory",
    )  # never put / at the end!!!
    parser.add_argument("--separate_calib", action="store_true", default=False)
    parser.add_argument("--no-subdir", action="store_true", default=True)
    parser.add_argument(
        "-o",
        "--calib-files-path",
        help="Path to save the raw calibration files",
        default=None,
        type=str,
    )

    args = parser.parse_args()

    if os.path.exists(tmp_dir):
        logging.warning("Previous temporary directory found, deleting.")
        shutil.rmtree(tmp_dir)

    if args.calib_files_path is not None:
        os.makedirs(args.calib_files_path, exist_ok=True)
        tmp_dir = args.calib_files_path

    main(args)
