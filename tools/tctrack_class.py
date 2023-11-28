from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
sys.path.append('../')

import argparse
import cv2
import torch
from glob import glob

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pysot.core.config import cfg
from pysot.models.utile_tctrack.model_builder import ModelBuilder_tctrack
from pysot.tracker.tctrack_tracker import TCTrackTracker
from pysot.utils.model_load import load_pretrain

torch.set_num_threads(1)

class TCTrack:
    fps = 0
    cap = cv2.VideoCapture(0)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    if not os.path.exists("recorded_videos"):
        os.mkdir("recorded_videos")
        print("folder recorded_videos is created")
    result = cv2.VideoWriter('./recorded_videos/{t}_webcam.mp4'.format(t=time.time()),
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             frame_rate, size)

    # config_file = "./experiments/TCTrack/config.yaml"
    # snapshot = "./tools/snapshot/checkpoint00_e84.pth"
    # video_name = ""

    def get_frames(self, video_name):
        if not video_name:
            # global cap

            # warmup
            for i in range(5):
                self.cap.read()

            #frame_width = int(cap.get(3))
            #frame_height = int(cap.get(4))
            #size = (frame_width, frame_height)
            #result = cv2.VideoWriter('{t}_webcam.avi'.format(t=time.time()),
            #                 cv2.VideoWriter_fourcc(*'MJPG'),
            #                 10, size)

            #num_frames = 0
            #seconds = 0
            #start = time.time()
            global fps
            new_frame_time = 0
            prev_frame_time = 0
            while True:
                #if seconds > 1:
                    #global fps
                    #fps = num_frames
                    #start = time.time()
                    #num_frames = 0
                    #print("fps:", fps)

                ret, frame = self.cap.read()

                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                print("fps:", fps)

                #result.write(frame)

                #num_frames = num_frames + 1
                #end = time.time()
                #seconds = end - start

                #fps = num_frames / seconds
                #print("fps:", fps), "num_frames:", num_frames, "seconds:", seconds)
                if ret:
                    yield frame
                else:
                    break
        elif video_name.endswith('avi') or \
            video_name.endswith('mp4'):
            print("video_name:", video_name)
            print("self.video_name_f:", self.video_name_f) 
            cap = cv2.VideoCapture(self.video_name_f)
            while True:
                ret, frame = cap.read()
                if ret:
                    yield frame
                else:
                    break
        else:
            images = sorted(glob(os.path.join(video_name, 'img', '*.jp*')))
            for img in images:
                frame = cv2.imread(img)
                yield frame


    def __init__(self, config_file, snapshot, coordinates, video_name_f=""):
        self.config_file = config_file
        self.snapshot = snapshot
        self.coordinates = coordinates
        self.video_name_f = video_name_f

    def run(self):
        # load config
        cfg.merge_from_file(self.config_file)
        cfg.CUDA = torch.cuda.is_available()
        print("cuda:", cfg.CUDA)
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        # create model
        model = ModelBuilder_tctrack('test')

        # load model
        model = load_pretrain(model, self.snapshot).eval().to(device)

        print("is model on cuda:", next(model.parameters()).is_cuda)

        # build tracker
        tracker = TCTrackTracker(model)
        hp=[cfg.TRACK.PENALTY_K,cfg.TRACK.WINDOW_INFLUENCE,cfg.TRACK.LR] #cfg.TRACK.PENALTY_K,cfg.TRACK.WINDOW_INFLUENCE,cfg.TRACK.LR

        first_frame = True
        if self.video_name_f:
            video_name = video_name_f.split('/')[-1].split('.')[0]
        else:
            video_name = 'webcam'

        cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)


        for frame in self.get_frames(self.video_name_f):
            #print(frame)
            if first_frame:
                try:
                    # TO DO: change this part to use [(x,y), (x,y)] coordinats
                    # init_rect = cv2.selectROI(video_name, frame, False, False)
                    init_rect = tuple(coord for couple in self.coordinates for coord in couple)
                    print(init_rect)
                except:
                    print("asdf")
                    exit()
                tracker.init(frame, init_rect)
                first_frame = False
            else:
                outputs = tracker.track(frame,hp)
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 3)
                cv2.putText(frame, str(int(fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)

                self.result.write(frame)

                cv2.imshow(video_name, frame)
                #cv2.waitKey(40)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
