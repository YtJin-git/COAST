import copy
import json
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from test import (
    DIR_PATH,
    CompositionDataset,
    Evaluator,
    get_model,
    load_args,
    parser,
    predict_logits,
    test,
    threshold_with_feasibility,
)
from models.coast import (
    predict_logits_text_first_with_coast,
)

try:
    import wandb
except ImportError:
    wandb = None


cudnn.benchmark = True


def _select_prediction_function(config):
    if not getattr(config, "text_first", False):
        return predict_logits

    supported_functions = {
        "predict_logits_text_first_with_coast": predict_logits_text_first_with_coast,
    }
    tta_function = getattr(config, "tta_function", "predict_logits_text_first_with_coast")
    if tta_function not in supported_functions:
        print(f"TTA function '{tta_function}' is not included in this COAST-only release; using COAST default.")
        tta_function = "predict_logits_text_first_with_coast"

    print(f"Using TTA function: {tta_function}")
    return supported_functions[tta_function]


if __name__ == "__main__":
    config = parser.parse_args()
    if config.yml_path:
        load_args(config.yml_path, config)
    config.eval_batch_size = 1

    if config.use_wandb:
        if wandb is None:
            raise ImportError("wandb is required when config.use_wandb=True.")
        wandb.init(project=f"TTA_CZSL_{config.dataset}")
        for key, value in wandb.config.items():
            if hasattr(config, key):
                setattr(config, key, value)
                print(f"Sweep override: {key} = {value}")

    print(f"Config: {config}")
    device = f"cuda:{config.cuda_device}" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"load_model: {config.load_model}")

    print("----")
    test_type = "OPEN WORLD" if config.open_world else "CLOSED WORLD"
    print(f"{test_type} evaluation details")
    print("----")
    print(f"dataset: {config.dataset}")

    print("loading test dataset")
    test_dataset = CompositionDataset(
        config.dataset_path,
        phase="test",
        split="compositional-split-natural",
        open_world=config.open_world,
    )

    allattrs = test_dataset.attrs
    allobj = test_dataset.objs
    classes = [obj.replace(".", " ").lower() for obj in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset, dset=test_dataset).to(device)
    if config.load_model:
        model.load_state_dict(torch.load(config.load_model, map_location=device, weights_only=True))

    predict_logits_func = _select_prediction_function(config)

    print("evaluating on the test set")
    evaluator = Evaluator(test_dataset, model=None)
    best_th = None

    if config.open_world and config.threshold is None:
        feasibility_path = os.path.join(DIR_PATH, f"data/feasibility_{config.dataset}.pt")
        unseen_scores = torch.load(feasibility_path, map_location="cpu")["feasibility"]
        seen_mask = test_dataset.seen_mask.to("cpu")
        min_feasibility = (unseen_scores + seen_mask * 10.0).min()
        max_feasibility = (unseen_scores - seen_mask * 10.0).max()
        thresholds = np.linspace(min_feasibility, max_feasibility, num=config.threshold_trials)

        best_auc = 0.0
        test_stats = None
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits_func(model, test_dataset, config)
        for th in thresholds:
            temp_logits = threshold_with_feasibility(
                all_logits.detach(),
                test_dataset.seen_mask,
                threshold=th,
                feasiblity=unseen_scores,
            )
            results = test(test_dataset, evaluator, temp_logits, all_attr_gt, all_obj_gt, all_pair_gt, config)
            auc = results["AUC"]
            if auc > best_auc:
                best_auc = auc
                best_th = th
                print("New best AUC", best_auc)
                print("Threshold", best_th)
                test_stats = copy.deepcopy(results)
    else:
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits_func(model, test_dataset, config)
        if config.open_world:
            feasibility_path = os.path.join(DIR_PATH, f"data/feasibility_{config.dataset}.pt")
            unseen_scores = torch.load(feasibility_path, map_location="cpu")["feasibility"]
            best_th = config.threshold
            print("using threshold: ", best_th)
            all_logits = threshold_with_feasibility(
                all_logits,
                test_dataset.seen_mask,
                threshold=best_th,
                feasiblity=unseen_scores,
            )

        test_stats = copy.deepcopy(test(test_dataset, evaluator, all_logits, all_attr_gt, all_obj_gt, all_pair_gt, config))
        result = ""
        for key in test_stats:
            result = result + key + "  " + str(round(test_stats[key] * 100, 2)) + " | "
        print(result)

        if config.use_wandb:
            wandb.log({
                "S": test_stats["best_seen"],
                "U": test_stats["best_unseen"],
                "HM": test_stats["best_hm"],
                "AUC": test_stats["AUC"],
                "attr_acc": test_stats["attr_acc"],
                "obj_acc": test_stats["obj_acc"],
            })

    results = {"test": test_stats}
    if config.open_world and best_th is not None:
        results["best_threshold"] = float(best_th)

    if config.load_model:
        os.makedirs(config.load_model_path, exist_ok=True)
        title = config.load_model_path
    else:
        os.makedirs(config.save_path, exist_ok=True)
        title = config.save_path

    result_path = os.path.join(title, "open.calibrated.json" if config.open_world else "closed.json")
    with open(result_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp)

    print("Done!")
