import time
import os
import shutil
import pickle
from PIL import Image

import cv2
import gym
from flask import Flask, render_template, jsonify, request, send_from_directory


app = Flask(__name__)

server_data = None
data_file = os.path.join(app.static_folder, 'asterix', 'data.pickle')


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/frame')
def get_frame():
    """Get single frame"""
    frame_id = request.args.get('id')
    if frame_id is None:
        frame_id = 0

    fname = str(frame_id) + '.png'
    return send_from_directory(server_data["img_dir"], fname)


@app.route('/frame/info')
def get_frame_info():
    """Get single frame"""
    frame_id = request.args.get('id')
    if frame_id is None:
        frame_id = 0

    frame_info = {
        "num_frames": server_data["num_frames"],
        "frame_id": frame_id,
        "rgb_obs_height": server_data["rgb_obs_shape"][0],
        "rgb_obs_width": server_data["rgb_obs_shape"][1]
    }
    return jsonify(frame_info)


def read_data():
    """Load data on server"""
    print('[INFO] Loading server resources ...')
    with open(data_file, 'rb') as file:
        data = pickle.load(file)

    data["rgb_obs_shape"] = data["rgb_frames"][0].shape
    data["num_frames"] = len(data["rgb_frames"])

    # convert rgb array to png files (this makes future life easier) and save in temporary directory
    img_dir = os.path.join(app.static_folder, "tmp")
    data["img_dir"] = img_dir
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    else:
        # if tmp directory exists, empty it first
        shutil.rmtree(img_dir)
        os.mkdir(img_dir)
    for idx in range(data["num_frames"]):
        img = Image.fromarray(data["rgb_frames"][idx])
        img.save(os.path.join(img_dir, str(idx) + '.png'))

    return data


def main():
    global server_data
    server_data = read_data()
    app.run()


if __name__ == '__main__':
    main()
