import os
import time
import utils
import numpy as np

class Evaluator(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.img_saver = utils.ImageSaver()
        self.sdm = utils.DAVISLabels()

    def evaluate_video(self, model, video_name, video_parts, output_path):
        imgs = video_parts['imgs'].cuda()
        flows = video_parts['flows'].cuda()
        gts = video_parts['gts']
        print(gts.shape)
        files = video_parts['files']

        # inference
        t0 = time.time()
        vos_out = model(imgs, flows)
        t1 = time.time()

        for idx in range(len(files)):
            fpath = os.path.join(output_path, video_name, files[idx])
            gt = gts[idx]
            data = ((vos_out['masks'][0, idx, 0, :, :].cpu().byte().numpy(), fpath), self.sdm)
            print(data[0][0].shape, data[0][0].dtype, np.unique(data[0][0]))
            self.img_saver.enqueue(data)
        return t1 - t0, imgs.size(1)

    def evaluate(self, model, output_path):
        model.cuda()
        total_time, total_frames = 0, 0
        video_name, video_parts = self.dataset.get_video()
        os.makedirs(os.path.join(output_path, video_name), exist_ok=True)
        time_elapsed, frames = self.evaluate_video(model, video_name, video_parts, output_path)
        total_time = total_time + time_elapsed
        total_frames = total_frames + frames
        print('{} done, {:.1f} fps'.format(video_name, frames / time_elapsed))
        print('total fps: {:.1f}\n'.format(total_frames / total_time))
        self.img_saver.kill()

