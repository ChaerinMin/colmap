import os
import struct


def points3D_to_ply(points3D_path, ply_path):
    points3D = {}
    if os.path.splitext(points3D_path)[1] == ".bin":
        with open(points3D_path, "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_points):
                point_id = struct.unpack("<Q", f.read(8))[0]
                xyz = struct.unpack("<ddd", f.read(24))
                rgb = struct.unpack("<BBB", f.read(3))
                error = struct.unpack("<d", f.read(8))[0]
                track_length = struct.unpack("<Q", f.read(8))[0]
                track = []
                for _ in range(track_length):
                    image_id, point2D_idx = struct.unpack("<ii", f.read(8))
                    track.append((image_id, point2D_idx))
                points3D[point_id] = {
                    "xyz": xyz,
                    "rgb": rgb,
                    "error": error,
                    "track": track,
                }
    elif os.path.splitext(points3D_path)[1] == ".txt":
        with open(points3D_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("#"):
                    continue  # Skip comments
                elements = line.split()
                point_id = int(elements[0])
                xyz = tuple(map(float, elements[1:4]))
                rgb = tuple(map(int, elements[4:7]))
                error = float(elements[7])
                track = [
                    (int(elements[i]), int(elements[i + 1]))
                    for i in range(8, len(elements), 2)
                ]
                points3D[point_id] = {
                    "xyz": xyz,
                    "rgb": rgb,
                    "error": error,
                    "track": track,
                }
    else:
        raise ValueError(f"Unknown file extension for {points3D_path}")

    with open(ply_path, "w") as f:
        # Write the PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(points3D)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write the points
        for point_id, point in points3D.items():
            xyz = point["xyz"]
            rgb = point["rgb"]
            f.write(
                "{} {} {} {} {} {}\n".format(
                    xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2]
                )
            )
    print(f"Points written to {ply_path}")
