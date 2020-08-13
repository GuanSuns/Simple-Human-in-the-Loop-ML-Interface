import time
import os
import shutil
import pickle
from PIL import Image

import cv2
import gym
from flask import Flask, render_template, jsonify, request, send_from_directory

from examples.object_tracking.tracker.tracker import ObjectTracker

app = Flask(__name__)
trackers = ObjectTracker()


server_data = None
# server data format
# {
# 'rgb_frames': [],
# 'actions': [],
# 'rgb_obs_shape':[],
# 'num_frames': int,
# 'current_frame_id': int,
# 'frame_info': [{'bounding_boxes': [], 'human_feedback': int, 'is_evaluated': int, 'tracking_boxes': []}]
# }

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
    frame_id = int(frame_id)

    fname = str(frame_id) + '.png'
    return send_from_directory(server_data['img_dir'], fname)


@app.route('/frame/info', methods=['GET'])
def get_frame_info():
    """Get single frame"""
    frame_id = request.args.get('id')
    if frame_id is None:
        frame_id = 0
    frame_id = int(frame_id)

    global server_data
    if frame_id != server_data['current_frame_id']:
        prev_frame_id = server_data['current_frame_id']
        current_bgr_obs = cv2.cvtColor(server_data['rgb_frames'][frame_id], cv2.COLOR_RGB2BGR)

        # check whether need to perform object tracking
        if frame_id == prev_frame_id + 1 and frame_id != 0:

            # only do tracking when previous frame has tracking boxes information
            # and the user hasn't provided any bounding boxes for current frame
            if len(server_data['frame_info'][frame_id]['bounding_boxes']) == 0 and \
                    len(server_data['frame_info'][prev_frame_id]['tracking_boxes']) > 0:
                # do tracking
                (success, boxes) = trackers.update(current_bgr_obs)
                # loop over the bounding boxes and save them in numpy coordinate
                server_data['frame_info'][frame_id]['tracking_boxes'] = []
                for box in boxes:
                    (x, y, h, w) = [int(v) for v in box]
                    server_data['frame_info'][frame_id]['tracking_boxes'].append([y, y + w, x, x + h])
        # reset tracking if human bounding boxes are provided
        if len(server_data['frame_info'][frame_id]['bounding_boxes']) > 0:
            trackers.reset_trackers()
            for box in server_data['frame_info'][frame_id]['bounding_boxes']:
                trackers.add_new_tracker(current_bgr_obs, (box[2], box[0], box[3]-box[2], box[1]-box[0]))

        # update frame visit information
        server_data['current_frame_id'] = frame_id

    frame_info = server_data['frame_info'][frame_id]
    frame_info.update({
        'num_frames': server_data['num_frames'],
        'frame_id': frame_id,
        'rgb_obs_height': server_data['rgb_obs_shape'][0],
        'rgb_obs_width': server_data['rgb_obs_shape'][1],
        'action': server_data['actions'][frame_id]
    })
    return jsonify(frame_info)


@app.route('/frame/info', methods=['POST'])
def update_frame_info():
    new_info = request.get_json()
    frame_id = request.args.get('id')

    if frame_id is not None and new_info is not None:
        frame_id = int(frame_id)
        global server_data
        # update tracking information
        if frame_id == server_data['current_frame_id']:
            # only update tracking when current bounding boxes has been reset
            if len(server_data['frame_info'][frame_id]['bounding_boxes']) == 0 \
                    and ('bounding_boxes' in new_info) and len(new_info['bounding_boxes']) > 0:
                server_data['frame_info'][frame_id]['tracking_boxes'] = list(new_info['bounding_boxes'])
                # update tracker
                current_bgr_obs = cv2.cvtColor(server_data['rgb_frames'][frame_id], cv2.COLOR_RGB2BGR)
                trackers.reset_trackers()
                for box in new_info['bounding_boxes']:
                    trackers.add_new_tracker(current_bgr_obs, (box[2], box[0], box[3]-box[2], box[1]-box[0]))
        # save new info
        server_data['frame_info'][frame_id].update(new_info)
    else:
        return jsonify({'success': False}), 400, {'ContentType': 'application/json'}

    return jsonify({'success': True}), 200, {'ContentType': 'application/json'}


# noinspection PyBroadException
@app.route('/save', methods=['POST'])
def save_human_data():
    try:
        save_dir = os.path.join(app.root_path, "human_study_data")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_file = os.path.join(save_dir, 'human_data.pickle')
        with open(save_file, 'wb') as file:
            pickle.dump(server_data, file, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('[ERROR] Fail to save server data:', str(e))
        return jsonify({'success': False}), 400, {'ContentType': 'application/json'}
    return jsonify({'success': True}), 200, {'ContentType': 'application/json'}


def read_data():
    """Load data on server"""
    print('[INFO] Loading server resources ...')
    with open(data_file, 'rb') as file:
        data = pickle.load(file)

    data['current_frame_id'] = -1
    data['rgb_obs_shape'] = data['rgb_frames'][0].shape
    data['num_frames'] = len(data['rgb_frames'])
    data['frame_info'] = [{'bounding_boxes': [], 'human_feedback': 0, 'is_evaluated': 0, 'tracking_boxes': []} for _ in
                          range(data['num_frames'])]

    # convert rgb array to png files (this makes future life easier) and save in temporary directory
    img_dir = os.path.join(app.static_folder, "tmp")
    data['img_dir'] = img_dir
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    else:
        # if tmp directory exists, empty it first
        shutil.rmtree(img_dir)
        os.mkdir(img_dir)
    for idx in range(data['num_frames']):
        img = Image.fromarray(data['rgb_frames'][idx])
        img.save(os.path.join(img_dir, str(idx) + '.png'))

    return data


def main():
    global server_data
    server_data = read_data()
    # init tracker
    global trackers
    trackers.reset_trackers()
    # run the server program
    app.run()


if __name__ == '__main__':
    main()
