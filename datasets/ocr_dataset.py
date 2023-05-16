from torch.utils.data import Dataset
import os
class ocrDataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./data/ocr",
        dataset_name = "ct80"
    ):
        self.image_dir_path = image_dir_path
        self.dataset_name = dataset_name
        file_path = os.path.join(image_dir_path, f'{dataset_name}/test_label.txt')
        file = open(file_path, "r")
        self.lines = file.readlines()
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, idx):
        image_id = self.lines[idx].split()[0]
        img_path = os.path.join(self.image_dir_path,f'{self.dataset_name}/{image_id}')
        answers = self.lines[idx].split()[1]
        return {
            "image_path": img_path,
            "gt_answers": answers}