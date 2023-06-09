from torch.utils.data import Dataset
import os
class HMEDataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./data/HME100K/test_images",
        ann_path= "./data/HME100K/test_labels.txt",
    ):
        file = open(ann_path, "r")
        self.lines = file.readlines()
        self.image_dir_path = image_dir_path
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, idx):
        image_id = self.lines[idx].split("\t")[0]
        img_path = os.path.join(self.image_dir_path, image_id)
        answers = self.lines[idx].split("\t")[1]
        return {
            "image_path": img_path,
            "gt_answers": answers}
if __name__ == "__main__":
    data = HMEDataset("/home/zhangli/GPT4/MutimodelOCR/data/HME100K/test_images","/home/zhangli/GPT4/MutimodelOCR/data/HME100K/test_labels.txt")
    data = iter(data)
    batch = next(data)