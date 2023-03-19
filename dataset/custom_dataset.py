import os
import imghdr
import shutil
from .transforms import *
from torch.utils.data import Dataset
import torchvision as tv
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image

class TestCustomDataset(Dataset):
    def __init__(self, root, dataset_name):
        self.root = root
        self.dataset_name = dataset_name
        self.format_intermediate_dir_structure()
    
    def __del__(self):
        self.reformat_dir_structure()

    def format_intermediate_dir_structure(self):
        source_dir = os.path.join(self.root, self.dataset_name)
        destination_dir = os.path.join(source_dir, "Images")
        os.mkdir(destination_dir)

        for directory_item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, directory_item)
            #import pdb;pdb.set_trace()
            if os.path.isdir(source_path) or not imghdr.what(source_path):
                continue
            destination_path = os.path.join(destination_dir, directory_item)
            shutil.move(source_path, destination_path)

    def reformat_dir_structure(self):
        destination_dir = os.path.join(self.root, self.dataset_name)
        source_dir = os.path.join(destination_dir, "Images")
        flow_dir = os.path.join(destination_dir, "Flows")

        for directory_item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, directory_item)
            destination_path = os.path.join(destination_dir, directory_item)
            shutil.move(source_path, destination_path)
        shutil.rmtree(source_dir)
        shutil.rmtree(flow_dir)


    def read_img(self, path):
        pic = Image.open(path)
        transform = tv.transforms.ToTensor()
        return transform(pic)

    def read_mask(self, path):
        pic = Image.open(path)
        transform = LabelToLongTensor()
        return transform(pic)

    def create_flows(self, video_name):
        img_directory_path = os.path.join(self.root, video_name, 'Images')
        flow_directory_path = os.path.join(self.root, video_name, 'Flows')
        if not os.path.exists(flow_directory_path):
            os.makedirs(flow_directory_path)
        file_paths = os.listdir(img_directory_path)
        sorted_file_paths = sorted(file_paths, key=lambda x:int(x.split(".")[0]))

        images1 = []
        images2 = []
        frameids = []
        for index in range(0, len(sorted_file_paths), 1):
            try:
                second = os.path.join(img_directory_path, sorted_file_paths[index + 1])
                first = os.path.join(img_directory_path, sorted_file_paths[index])
            except IndexError:
                first = os.path.join(img_directory_path, sorted_file_paths[index])
                second = os.path.join(img_directory_path, sorted_file_paths[index - 1])
            images1.append(first)
            images2.append(second)
            frameids.append(sorted_file_paths[index])

        batch_count = 0
        batch_size = 3
        while images1[batch_count * batch_size: (batch_count + 1) * batch_size]:
            img1_batch = torch.stack(
                [
                    self.read_img(i) 
                    for i in 
                    images1[batch_count * batch_size: (batch_count + 1) * batch_size]
                ]
            )
            img2_batch = torch.stack(
                [
                    self.read_img(i) 
                    for i in 
                    images2[batch_count * batch_size: (batch_count + 1) * batch_size]
                ]
            )
            frameid_batch = frameids[batch_count * batch_size: (batch_count + 1) * batch_size]
            device = "cuda" if torch.cuda.is_available() else "cpu"

            model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
            model = model.eval()

            list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
            predicted_flows = list_of_flows[-1]
            predicted_flows = flow_to_image(predicted_flows)
            for i in range(predicted_flows.size(0)):
                image_tensor = predicted_flows[i]
                frameid = frameid_batch[i]
                file_path = os.path.join(flow_directory_path, frameid)
                image_pil = F.to_pil_image(image_tensor.to("cpu"))
                image_pil.save(file_path)

            batch_count += 1

    def get_video(self):
        image_sequence_path = os.path.join(self.root, self.dataset_name, 'Images')
        flow_sequence_path = os.path.join(self.root, self.dataset_name, 'Flows')
        image_name_list = os.listdir(image_sequence_path) 
        image_name_list = sorted(image_name_list, key= lambda x: int(x.split(".")[0]))
        self.create_flows(self.dataset_name)
        imgs = torch.stack([
            self.read_img(
                os.path.join(image_sequence_path, image_name)
            ) for image_name 
            in image_name_list]).unsqueeze(0)
        flows = torch.stack([
            self.read_img(
                os.path.join(flow_sequence_path, image_name )
            ) for image_name 
            in image_name_list]).unsqueeze(0)
        return self.dataset_name, {'imgs': imgs, 'flows': flows, 'files': image_name_list}
