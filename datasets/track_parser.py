import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import numpy as np
import copy
import plotly.graph_objects as go
import plotly.express as px

from yolox.tracking_utils.io import read_results, unzip_objs
import bisect
from PIL import Image
from icecream import ic

import matplotlib, random
import matplotlib.pyplot as plt

hex_colors_dic = {}
rgb_colors_dic = {}
hex_colors_only = []
for name, hex in matplotlib.colors.cnames.items():
    hex_colors_only.append(hex)
    hex_colors_dic[name] = hex
    rgb_colors_dic[name] = matplotlib.colors.to_rgb(hex)


class Track:
    __slots__ = ['data', 'termination', 'label', 'frame_ids']
    def __init__(self, frame_id, tlwh, label, termination=None):
        self.data = {frame_id: tlwh}
        self.termination = termination
        self.label = label
        self.frame_ids = [frame_id]

    @staticmethod
    def array_stack(tracks):
        pass

    def add(self, frame_id, tlwh):
        self.data[frame_id] = tlwh
        bisect.insort(self.frame_ids, frame_id)

    @property
    def num_frames(self):
        return len(self.data)

    @property
    def length(self):
        return self.frame_ids[-1] - self.frame_ids[0]

    def get_nth_frame_id(self, n):
        # track: {frame_id: tlwh}
        if self.length < n:
            return self.frame_ids[-1]
        else:
            i = bisect.bisect_left(self.frame_ids, self.frame_ids[0] + n)
            return self.frame_ids[i]

    def plot(self, name, H):
        x, y, w, h = zip(*[self.data[frame_id] for frame_id in self.frame_ids])
        ind = hash((x[-1], y[-1]))
        x = np.array(x)
        y = np.array(y)
        w = np.array(w)
        h = np.array(h)
        color = hex_colors_only[ind % len(hex_colors_only)]
        track_line = go.Scatter(x=x, y=H-np.array(y), text=self.frame_ids, mode='lines', line=dict(color=color))
        track_label = go.Scatter(x=x[0:1], y=H-np.array(y[0:1]), text=f"{name}:{self.termination}. Len: {self.length}", mode='text', textfont=dict(color='white'))
        rect = go.Scatter(
            x=np.array([x[0], x[0] + w[0], x[0] + w[0], x[0], x[0]]),
            y=H - np.array([y[0], y[0], y[0] + h[0], y[0] + h[0], y[0]]),
            # x=np.array([x[0], w[0], w[0], x[0], x[0]]),
            # y=H - np.array([y[0], y[0], h[0], h[0], y[0]]),
            mode='lines', line=dict(color=color))
        return [track_line, track_label, rect]
            

def threshold_tracks(tracks, thresholds=[10, 20, 50]):
    end_points = {}
    for gt_id, track in tracks.items():
        if track.length < thresholds[0]:
            continue
        end_points[gt_id] = [track.get_nth_frame_id(threshold) for threshold in thresholds]
    return end_points

class TrackParser(object):

    def __init__(self, data_root, seq_name, data_type, max_missing_frames):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type
        self.max_missing_frames = max_missing_frames

    def load_annotations(self):
        assert self.data_type == 'mot'

        gt_filename = str(self.data_root / self.seq_name / 'gt' / 'gt.txt')
        # format: dict that maps frame_id to list of (tlwh, target_id, score)
        self.gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True)
        self.gt_ignore_frame_dict = read_results(gt_filename, self.data_type, is_ignore=True)
        self.frames = sorted(list(set(self.gt_frame_dict.keys()))) 
        return self        

    def load_image(self, frame_id):
        im_path = os.path.join(self.data_root, self.seq_name, 'img1', f'{frame_id:06d}.jpg')
        img = Image.open(im_path)
        return img

    def is_stationary(self, hue_threshold=10, background_percentage=.6):
        # load images
        imgs = []
        for frame_id in self.frames[::60]:
            img = self.load_image(frame_id)
            # img = img.convert('HSV')
            imgs.append(np.asarray(img))
        imgs = np.stack(imgs, axis=0)
        # first, find the background image
        bimg = np.median(imgs, axis=0, keepdims=True)
        hue_diff = np.abs((imgs - bimg)[..., 0])
        mean_hue_diff = np.where(hue_diff > hue_threshold, 1, 0).mean()
        return mean_hue_diff < background_percentage
        
    def accumulate_tracks(self, frame_id):
        # frame ids are ints
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids, _, gt_labs = unzip_objs(gt_objs, ret_labels=True)
        gt_ids_set = set(gt_ids)
        missing_count = {gt_id: 0 for gt_id in gt_ids}
        tracks = {gt_id: Track(frame_id, gt_tlwh, gt_lab) for gt_id, gt_tlwh, gt_lab in zip(gt_ids, gt_tlwhs, gt_labs)}
        for oi in range(self.frames.index(frame_id), self.frames[-1]):
            oframe_id = self.frames[oi]
            ogt_objs = self.gt_frame_dict.get(oframe_id, [])
            ogt_tlwhs, ogt_ids = unzip_objs(ogt_objs)[:2]
            ogt_ids_set = set(ogt_ids)
            intersect = gt_ids_set.intersection(ogt_ids_set)
            # end process if all tracks are lost
            if len(intersect) == 0:
                break
            for gt_id in gt_ids_set.difference(ogt_ids_set):
                missing_count[gt_id] += 1
                # we don't terminate tracks here because we don't know if they will show back up

            # extend tracks
            for gt_id in intersect:
                missing_count[gt_id] = 0
                tracks[gt_id].add(oframe_id,  ogt_tlwhs[ogt_ids.index(gt_id)])
        
        for gt_id in gt_ids_set:
            count = missing_count[gt_id]
            if count <= self.max_missing_frames:
                # these are the tracks that are not terminated        
                tracks[gt_id].termination = -1
            else:
                # these are tracks that would have been terminated but we didn't know if they would show back up
                tracks[gt_id].termination = self.frames[-1] - count

        return tracks

    def plot(self, frame_id, scale_factor=0.8):
        tracks = self.accumulate_tracks(frame_id)
        img = self.load_image(frame_id)
        objs = []
        for gt_id, track in tracks.items():
            objs.extend(track.plot(gt_id, img.height))

        fig = go.Figure(data=objs)
        # fig = go.Figure()
        fig.update_layout(
            xaxis=dict(
                range=[0, img.width],
                showgrid=False,
            ),
            yaxis=dict(
                range=[0, img.height],
                showgrid=False,
            ),
            width=img.width*scale_factor,
            height=img.height*scale_factor,
        )
        fig.add_layout_image(
            x=0,
            sizex=img.width,
            y=img.height,
            sizey=img.height,
            xref="x",
            yref="y",
            sizing='stretch',
            opacity=1.0,
            layer="below",
            source=img,
            # xref="x domain",
            # yref="x domain",
            # xanchor="left",
            # yanchor="top",
            # layer="below",
            # sizing="stretch",
            # sizex=1.0,
            # sizey=1.0
        )
        return fig

if __name__ == "__main__":
    data_root = Path("/data/MOT17/train")
    seq_names = [seq_path.name for seq_path in data_root.iterdir()]

    # detect stationary sequences
    parsers = {seq_name:
        TrackParser(data_root, seq_name, "mot", max_missing_frames=2).load_annotations()
        for seq_name in seq_names}
    for seq_name in seq_names:
        print(f"{seq_name}: {parsers[seq_name].is_stationary()}")