import os
import sqlite3


def get_images(args, db_path, image_dir):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("SELECT image_id,name FROM images;")
    image_ids, image_paths = zip(*cursor.fetchall())
    image_ids = list(image_ids)
    image_paths = [os.path.join(image_dir, image_path) for image_path in image_paths]
    cursor.close()
    connection.close()
    return image_ids, image_paths