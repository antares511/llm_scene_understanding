from cProfile import label
import os
from tqdm import tqdm
from load_matterport3d_dataset import Matterport3dDataset
from model_utils import get_category_index_map
from perplexity_measure import compute_object_norm_inv_ppl
from extract_labels import create_label_lists
import numpy as np
from sympy.utilities.iterables import multiset_permutations
import pickle

import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from transformers import (
    BertModel,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    GPTNeoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTJModel,
)

from kumaraditya.utils import *
from torchmetrics.classification import MulticlassAveragePrecision


class BaselineRunner:

    def __init__(self, device=None, verbose=False, label_set="mpcat40", use_test=True):

        self.verbose = verbose
        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )

        self.label_set = label_set

        dataset = Matterport3dDataset(
            "./mp_data/" + label_set + "_matterport3d_w_edge_502030_new.pkl"
        )
        labels, pl_labels = create_label_lists(dataset)
        self.building_list, self.room_list, self.object_list = labels
        self.building_list_pl, self.room_list_pl, self.object_list_pl = pl_labels

        if use_test:
            dataset = dataset.get_test_set()

        if self.verbose:
            print("Using device:", self.device)

        # create data loader
        self.dataloader = DataLoader(dataset, batch_size=82)
        room_obj_freqs = (
            np.load(
                "./cooccurrency_matrices/" + label_set + "_gt" + "/room_object.npy",
            )
            + 1
        )
        self.object_norm_inv_perplexity = compute_object_norm_inv_ppl(
            "./cooccurrency_matrices/" + label_set + "_gt" + "/room_object.npy",
            True,
        ).to(self.device)

        excluded_room_indices = np.array(
            [
                self.room_list.index(excluded_room)
                for excluded_room in ["None", "yard", "porch", "balcony"]
            ]
        )
        room_obj_freqs[excluded_room_indices] = 0

        self.room_obj_conditionals = room_obj_freqs / np.sum(
            room_obj_freqs, axis=0, keepdims=True
        )
        """ print(self.room_obj_conditionals) """

    def extract_data(self, max_num_obj):
        """
        Extracts and saves the most interesting objects from each room.

        TODO: Finish docstring
        """
        # self.max_num_obj = max_num_obj

        batch = next(iter(self.dataloader))
        label = (
            batch.y[batch.building_mask],
            batch.y[batch.room_mask],
            batch.y[batch.object_mask],
        )
        y_object = F.one_hot(label[-1], len(self.object_list)).type(torch.LongTensor)
        category_index_map = get_category_index_map(batch)
        object_room_edge_index = batch.object_room_edge_index

        total_count = 0
        correct_count = 0
        correct_count_use_topk = 0

        gt_rooms_list = []
        inferred_rooms_list = []

        metric = MulticlassAveragePrecision(
            num_classes=len(self.room_list), average="weighted", thresholds=None
        )

        for i in range(len(label[1])):  # range(len(label[1])):
            ground_truth_room = label[1][i]

            mask = category_index_map[object_room_edge_index[1]] == i
            neighbor_dists = y_object[
                category_index_map[object_room_edge_index[0][mask]]
            ]
            neighbor_dists = neighbor_dists.to(self.device)
            all_objs = torch.sum(neighbor_dists, dim=0) > 0

            scores = all_objs * self.object_norm_inv_perplexity
            best_objs = (
                torch.topk(scores, max(min((all_objs > 0).sum(), max_num_obj), 1))
                .indices.cpu()
                .numpy()
                .flatten()
            )

            room_label = self.room_list[ground_truth_room]
            if (
                room_label in ["None", "yard", "porch", "balcony"]
                or len(neighbor_dists) == 0
            ):
                continue

            # print("------------------------------------------------")
            # print(self.room_list[ground_truth_room])
            objs_list = all_objs.nonzero().cpu().numpy().flatten()
            inferred_room_dist = self.room_obj_conditionals[:, objs_list].prod(axis=1)
            inferred_room = np.argmax(inferred_room_dist)

            softmax = torch.nn.Softmax(dim=0)
            inferred_room_dist = softmax(torch.tensor(inferred_room_dist)).unsqueeze(0)
            # inferred_room_dist = torch.tensor(inferred_room_dist).unsqueeze(0)

            unsqueezed_ground_truth_room = ground_truth_room.unsqueeze(0)
            metric.update(
                inferred_room_dist,
                unsqueezed_ground_truth_room,
            )

            gt_rooms_list.append(ground_truth_room.item())
            inferred_rooms_list.append(inferred_room)

            inferred_room_dist_use_topk = self.room_obj_conditionals[:, best_objs].prod(
                axis=1
            )
            inferred_room_use_topk = np.argmax(inferred_room_dist_use_topk)
            # print(self.room_list[inferred_room])

            if inferred_room == ground_truth_room:
                correct_count += 1
            if inferred_room_use_topk == ground_truth_room:
                correct_count_use_topk += 1
            total_count += 1

        print("Fraction correct using all objects:", correct_count / total_count)
        print(
            "Fraction correct using top k objects:",
            correct_count_use_topk / total_count,
        )

        save_folder = os.path.join(
            "/scratch/kumaraditya_gupta/roomlabel/lc_baselines/", "statistical"
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        print("MAP:", metric.compute())

        # compute_metrics_by_class(
        #     np.array(gt_rooms_list),
        #     np.array(inferred_rooms_list),
        #     self.room_list,
        #     self.label_set,
        #     save_folder,
        # )


if __name__ == "__main__":
    # for label_set in ["mpcat40", "nyuClass"]:
    #     for use_test in [True, False]:
    #         print(label_set, "use test:", use_test)
    #         bl_runner = BaselineRunner(label_set=label_set, use_test=use_test)
    #         bl_runner.extract_data(3)

    label_set = "nyuClass"
    use_test = True

    bl_runner = BaselineRunner(label_set=label_set, use_test=use_test)
    bl_runner.extract_data(3)
