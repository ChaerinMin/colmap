import cv2
from cv2 import aruco
from tqdm import tqdm
import logging
import numpy as np
import os
import sqlite3

MARKER_AREA_THRESHOLD = 500  # 1000 # Parameter

MAX_IMAGE_ID = 2**31 - 1


def array_to_blob(array):
    return array.tobytes()

def blob_to_array(blob, dtype, shape=(-1,)):
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)

def add_features(image_ids, image_paths, db_path, debug):
    # IMPORTANT: Change this if dictionary is changed
    aruco_dict = aruco.getPredefinedDictionary(
        aruco.DICT_4X4_250
    )  # Using few bits per marker for better detection
    # parameters = aruco.DetectorParameters()
    # detector = aruco.ArucoDetector(aruco_dict)#, parameters)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.0001,
    )  # Sub-pixel detection

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("DELETE FROM keypoints;")
    cursor.execute("DELETE FROM descriptors;")
    cursor.execute("DELETE FROM matches;")
    connection.commit()

    logging.info("Extracting features from ChArUco")
    image_id_desc = {}
    for image_id, image_path in tqdm(list(zip(image_ids, image_paths))):
        frame = cv2.imread(image_path)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # orig_corners, orig_ids, _ = detector.detectMarkers(gray)
        orig_corners, orig_ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict)

        if len(orig_corners) <= 0:
            logging.warning("No markers found in {image_path}")
            continue
        else:
            corners = []
            ids = []
            for corner, id in zip(orig_corners, orig_ids):
                area = cv2.contourArea(corner)
                if area >= MARKER_AREA_THRESHOLD:  # PARAM
                    ids.append(id)
                    corners.append(corner)

            if len(orig_corners) - len(corners) > 0:
                logging.warning(
                    f"Ignoring {len(orig_corners) - len(corners)} sliver markers."
                )

            ids = np.asarray(ids).flatten()
            # ipdb.set_trace()
            uniq, cnt = np.unique(ids, return_counts=True)
            if debug:
                frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
                cv2.imshow("markers", frame_markers)
                cv2.waitKey(0)
            if np.any(cnt > 1):
                logging.warning(
                    f"{np.count_nonzero(cnt > 1)} duplicate IDs found, ignoring"
                )
                raise
                # if ch == 27:
                #     exit()
            if len(corners) < 4:
                logging.warning(f"Not enough markers found in {image_path}")

            non_uniq = dict(zip(uniq, cnt))

            uniq_corners = []
            uniq_ids = []
            for i in range(len(corners)):
                # Skip if ID is non-unique
                if non_uniq[ids[i]] > 1:
                    continue

                if np.all(corners[i] >= 0):
                    cv2.cornerSubPix(
                        gray,
                        corners[i],
                        winSize=(3, 3),
                        zeroZone=(-1, -1),
                        criteria=criteria,
                    )
                    corners[i] = corners[i].squeeze()
                    uniq_ids.append(ids[i])
                    uniq_corners.append(corners[i])
                else:
                    raise NotImplementedError

            ids = np.asarray(uniq_ids)
            uniq, cnt = np.unique(ids, return_counts=True)

            # Insert keypoints
            keypoints = np.concatenate(uniq_corners)
            cursor.execute(
                "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
                (image_id,)
                + keypoints.shape
                + (array_to_blob(keypoints.astype(np.float32)),),
            )

            ids = np.repeat(ids, 4)
            for i in range(4):
                # TODO: Find a more elegant solution for this
                ids[i::4] += i * 1000
            image_id_desc[image_id] = ids

        connection.commit()

    cursor.close()
    connection.close()
    return image_id_desc


def exhaustive_match(image_ids, image_paths, db_path, args, image_id_desc, image_dir, tmp_dir):
    def image_ids_to_pair_id(image_id1, image_id2):
        if image_id1 > image_id2:
            image_id1, image_id2 = image_id2, image_id1
        return image_id1 * MAX_IMAGE_ID + image_id2
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("DELETE FROM matches;")
    connection.commit()

    logging.info("Matching features")
    image_pairs = []
    for i in tqdm(range(len(image_ids))):
        for j in range(len(image_ids)):
            if image_ids[i] >= image_ids[j]:
                continue

            desc1 = image_id_desc[image_ids[i]]
            desc2 = image_id_desc[image_ids[j]]

            # Find matches
            matches = []
            for k in range(desc1.shape[0]):
                for l in range(desc2.shape[0]):
                    if desc1[k] == desc2[l]:
                        matches.append([k, l])

            # Insert into database
            pair_id = image_ids_to_pair_id(image_ids[i], image_ids[j])
            if not matches:
                continue

            image_pairs.append([image_paths[i], image_paths[j]])
            matches = np.asarray(matches, np.uint32)

            cursor.execute(
                "INSERT INTO matches VALUES (?, ?, ?, ?)",
                (pair_id,) + matches.shape + (array_to_blob(matches),),
            )

            connection.commit()

    cursor.close()
    connection.close()

    match_list_path = os.path.join(tmp_dir, "match_list.txt")
    logging.info(f"Writing image pairs at {match_list_path}")
    with open(os.path.join(match_list_path), "w") as f:
        for pair in image_pairs:
            f.write((" ".join(pair)).replace(image_dir + "/", "") + "\n")