from dataset import *
import evaluation
from tmo import TMO
import warnings
import os
import random
warnings.filterwarnings('ignore')

def test_custom_dataset(model):
    dataset_path = os.environ["DATASET_PATH"]
    output_path = os.environ["OUTPUT_PATH"]
    os.makedirs(output_path, exist_ok=True)

    #import pdb;pdb.set_trace()
    for dataset_name in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, dataset_name)):
            dataset = TestCustomDataset(dataset_path, dataset_name)
            print(f"---- Processing dataset {dataset_name} ----")
            evaluator = evaluation.Evaluator(dataset)        
            evaluator.evaluate(model, output_path)

def main():
    # seed to reproduce results
    torch.cuda.set_device(0)
    seed = 19971007
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # define model
    model = TMO().eval()
    model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load('trained_model/TMO.pth', map_location='cuda:0'))
    with torch.no_grad():
        test_custom_dataset(model)

if __name__ == '__main__':
    main()
