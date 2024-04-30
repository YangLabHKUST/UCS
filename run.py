import warnings
warnings.filterwarnings("ignore")
import numpy as np
from src.experiment_manager import ExperimentManager
from src.dataset import UCSDataset
from src.model import UCSModel

if __name__ == "__main__":
    manager = ExperimentManager()
    opt = manager.get_opt()

    # Initializing the Dataset
    print("Initializing train dataset")
    train_dataset = UCSDataset(manager)
    train_dataset.set_train(True)
    gene_map_shape = train_dataset.gene_map.shape

    # Initializing the model
    model = UCSModel(manager, gene_map_shape)
    model.train_model(train_dataset)

    print("Initializing pred dataset with shift 0 and shift patch size//2")
    pred_dataset_0 = UCSDataset(manager, 0)
    pred_dataset_1 = UCSDataset(manager, opt.patch_size//2)
    pred_dataset_0.set_train(False)
    pred_dataset_1.set_train(False)
    model.predict_whole(pred_dataset_0, pred_dataset_1)
    model.postprocess()


