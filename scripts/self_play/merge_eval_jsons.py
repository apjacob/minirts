# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import sys
import pprint

from set_path import append_sys_path

append_sys_path()

import torch
import random
import json
import glob
from multitask_pop_eval import *
import scipy.stats as st


def parse_args():
    parser = argparse.ArgumentParser(description="Gather eval jsons")
    parser.add_argument("--json_dir", type=str)
    parser.add_argument(
        "--out_path", type=str, default="/home/gridsan/apjacob/minirts/merged_evals/"
    )
    args = parser.parse_args()

    return args


def confidence_interval(samples):
    if len(samples) == 1:
        return "-"
    low, hi = st.t.interval(
        alpha=0.95, df=len(samples) - 1, loc=np.mean(samples), scale=st.sem(samples)
    )
    mid = (low + hi) / 2
    return f"{mid} +/- {hi - mid}"


def welch_test(data1, data2, alpha=0.05, tail=2):

    assert (
        tail == 1 or tail == 2
    ), "tail should be one or two, referring to the one-sided or two-sided t-test."
    data1 = data1.squeeze()
    data2 = data2.squeeze()
    assert alpha < 1 and alpha > 0, "alpha should be between 0 and 1"

    t, p = st.ttest_ind(data1, data2, equal_var=False)

    if tail == 1:
        alpha = 2 * alpha
    if p <= alpha:
        if t < 0:
            print(
                "\n\nResult of the Welch's t-test at level %02g: μ2>μ1, the test passed with p-value = %02g."
                % (alpha, p)
            )
        else:
            print(
                "\n\nResult of the Welch's t-test level %02g: μ1>μ2, the test passed with p-value = %02g."
                % (alpha, p)
            )
    else:
        print(
            "\n\nResults of the Welch's t-test level %02g: there is not enough evidence to prove any order relation between μ1 and μ2."
            % alpha
        )
    print("Welch's t-test done.")


def main(args):
    main_dict = {}

    for file in tqdm(glob.glob(args.json_dir)):
        print(f"Loading file: {file}")
        with open(file) as f:
            partial_model_dict = json.load(f)
        main_dict.update(partial_model_dict)

    for super_exp, exps in main_dict.items():
        for exp, data in exps.items():
            win_rate = data["win_rate"]
            ci = confidence_interval(win_rate)
            data["ci"] = ci

    date = datetime.date(datetime.now())
    random_number = randint(1, 99999)
    path = os.path.join(args.out_path, f"merged_eval_{random_number}_{date}.json")
    with open(path, "w") as fp:
        json.dump(main_dict, fp)

    print(f"Merged eval jsons saved to: {path}")


if __name__ == "__main__":
    global device
    args = parse_args()
    main(args)
