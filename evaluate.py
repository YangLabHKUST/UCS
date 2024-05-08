import warnings
warnings.filterwarnings("ignore")
import numpy as np
from src.experiment_manager import ExperimentManager
from src.dataset import UCSDataset
from src.model import UCSModel

# Only evaluate the data, not train the model
MODEL_PATH = "/import/home2/yhchenmath/Code/ucs/log/xenium_breast_cancer/checkpoint/model.pth"

if __name__ == "__main__":
    manager = ExperimentManager()
    opt = manager.get_opt()

    print("Initializing pred dataset with shift 0 and shift patch size//2")
    pred_dataset_0 = UCSDataset(manager, 0)
    pred_dataset_1 = UCSDataset(manager, opt.patch_size//2)
    gene_map_shape = pred_dataset_0.gene_map.shape

    model = UCSModel(manager, gene_map_shape)
    model.load(MODEL_PATH)
    model.eval()
    pred_dataset_0.set_train(False)
    pred_dataset_1.set_train(False)
    model.predict_whole(pred_dataset_0, pred_dataset_1)
    model.postprocess()


