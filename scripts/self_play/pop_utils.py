import os
import itertools
import csv
import pprint
import pandas as pd

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_root = os.path.join(root, "pretrained_models")

model_dicts = {}
NUM_TRIALS = 10
base_model_dicts = {
    "bc": [
        {
            "cloned": True,
            "rule": 80,
            "best_coach": os.path.join(model_root, "coach_rnn500.pt"),
            "best_exec": os.path.join(model_root, "executor_rnn.pt"),
        }
    ]
    * NUM_TRIALS,
    "bc_zero": [
        {
            "cloned": True,
            "rule": 80,
            "best_exec": os.path.join(model_root, "executor_zero.pt"),
        }
    ],
    # "ft_both[80]": [
    #     {
    #         "rule": 80,
    #         "model": "decent-bush-217",
    #         "run_path": "apjacob/adapt-minirts/250vw2ty",
    #         "best_exec": "best_exec_checkpoint_421.pt",
    #         "best_coach": "best_coach_checkpoint_421.pt",
    #     }
    # ],
    "ft_zero[80]": [
        {
            "rule": 80,
            "model": "multitask-fixed_selfplay-3o13mcac-exec-zero-default-80",
            "run_path": "apjacob/adapt-minirts/3o13mcac",
            "best_exec": "best_exec_checkpoint_1801.pt",
            "best_coach": "best_coach_checkpoint_1801.pt",
        }
    ],
    "ft_coach[14]": [
        {
            "rule": 14,
            "model": "dark-elevator-188",
            "run_path": "apjacob/adapt-minirts/flkl5oy2",
            "best_exec": "best_exec_checkpoint_1961.pt",
            "best_coach": "best_coach_checkpoint_1961.pt",
        }
    ],
    "ft_coach[7]": [
        {
            "rule": 7,
            "model": "ruby-feather-199",
            "run_path": "apjacob/adapt-minirts/4isepghm",
            "best_exec": "best_exec_checkpoint_1801.pt",
            "best_coach": "best_coach_checkpoint_1801.pt",
        }
    ],
    # # "ft_pop[80,40,20]": [
    # #     {
    # #         "rule": [80, 40, 20],
    # #         "model": "lyric-leaf-1",
    # #         "run_path": "apjacob/adapt-minirts-pop/1qumof17",
    # #         "best_exec": "best_exec_checkpoint_40_1622.pt",
    # #         "best_coach_80": "best_coach_checkpoint_80_841.pt",
    # #         "best_coach_40": "best_coach_checkpoint_80_841.pt",
    # #         "best_coach_20": "best_coach_checkpoint_80_841.pt",
    # #     }
    # # ],
    # # "ft_pop[3,12,13]": [
    # #     {
    # #         "rule": [3, 12, 13],
    # #         "model": "lunar-fog-7",
    # #         "run_path": "apjacob/adapt-minirts-pop/1ygw8r46",
    # #         "best_exec": "best_exec_checkpoint_3_1561.pt",
    # #         "best_coach_3": "best_coach_checkpoint_3_1561.pt",
    # #         "best_coach_13": "best_coach_checkpoint_13_1443.pt",
    # #         "best_coach_12": "best_coach_checkpoint_12_1262.pt",
    # #     }
    # # ],
    # "ft_both[12]": [
    #     {
    #         "rule": 12,
    #         "model": "sandy-forest-196",
    #         "run_path": "apjacob/adapt-minirts/1zk0an30",
    #         "best_exec": "best_exec_checkpoint_1841.pt",
    #         "best_coach": "best_coach_checkpoint_1841.pt",
    #     }
    # ],
    # "ft_both[7]": [
    #     {
    #         "rule": 7,
    #         "model": "vibrant-monkey-195",
    #         "run_path": "apjacob/adapt-minirts/1czxzs7k",
    #         "best_exec": "best_exec_checkpoint_581.pt",
    #         "best_coach": "best_coach_checkpoint_581.pt",
    #     }
    # ],
    "ft_hier_exec[80]": [
        {
            "rule": 80,
            "model": "smooth-silence-209",
            "run_path": "apjacob/adapt-minirts/3e2vpwt1",
            "best_exec": "best_exec_checkpoint_1501.pt",
            "best_coach": "best_coach_checkpoint_1501.pt",
        }
    ],
    "ft_hier_exec[21]": [
        {
            "rule": 21,
            "model": "glowing-violet-207",
            "run_path": "apjacob/adapt-minirts/33hm1jco",
            "best_exec": "best_exec_checkpoint_1381.pt",
            "best_coach": "best_coach_checkpoint_1381.pt",
        }
    ],
    "ft_hier_exec[14]": [
        {
            "rule": 14,
            "model": "multitask-fixed_selfplay-exec14-1aqqm60r-",
            "run_path": "apjacob/adapt-minirts/1aqqm60r",
            "best_exec": "best_exec_checkpoint_1821.pt",
            "best_coach": "best_coach_checkpoint_1821.pt",
        }
    ],
    "ft_coach[21]": [
        {
            "rule": 21,
            "model": "icy-deluge-205",
            "run_path": "apjacob/adapt-minirts/2w738s03",
            "best_exec": "best_exec_checkpoint_1901.pt",
            "best_coach": "best_coach_checkpoint_1901.pt",
        }
    ],
    "ft_coach[80]": [
        {
            "rule": 80,
            "model": "dark-vortex-212",
            "run_path": "apjacob/adapt-minirts/1e4cctnv",
            "best_exec": "best_exec_checkpoint_1141.pt",
            "best_coach": "best_coach_checkpoint_1141.pt",
        }
    ],
    # "ft_both[3]": [
    #     {
    #         "rule": 3,
    #         "model": "glorious-plasma-197",
    #         "run_path": "apjacob/adapt-minirts/1iy0lk2d",
    #         "best_exec": "best_exec_checkpoint_1921.pt",
    #         "best_coach": "best_coach_checkpoint_1921.pt",
    #     }
    # ],
    "ft_zero[3]": [
        {
            "rule": 3,
            "model": "stellar-mountain-235",
            "run_path": "apjacob/adapt-minirts/1f9w7itv",
            "best_exec": "best_exec_checkpoint_1981.pt",
            "best_coach": "best_coach_checkpoint_1981.pt",
        }
    ],
    "ft_hier_exec[3]": [
        {
            "rule": 3,
            "model": "glad-wood-241",
            "run_path": "apjacob/adapt-minirts/81k5xzih",
            "best_exec": "best_exec_checkpoint_1761.pt",
            "best_coach": "best_coach_checkpoint_1761.pt",
        }
    ],
    "ft_coach[3]": [
        {
            "rule": 3,
            "model": "twilight-sea-201",
            "run_path": "apjacob/adapt-minirts/3g4vb5ez",
            "best_exec": "best_exec_checkpoint_1861.pt",
            "best_coach": "best_coach_checkpoint_1861.pt",
        }
    ],
}


def populate_fixed_model_dicts():
    pre = "ft"
    mids = ["coach", "zero", "hier_exec", "both"]
    rules = ["3", "80", "21", "12", "13"]

    full_results_df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results/adapt-minirts-exps.csv"
        )
    )

    # Filter out iterations less than K
    trained_models_df = full_results_df[
        (full_results_df["max_iterations"] >= 1500)
        & (full_results_df["rule"].isin(rules))
    ]

    for mid, cur_rule in itertools.product(mids, rules):
        key = f"{pre}_{mid}[{cur_rule}]"
        cur_coach1 = "rnn500"

        if mid == "coach":
            cur_train_mode = "coach"
            cur_executor1 = "rnn"
            lr = 3e-6
        elif mid == "zero":
            cur_train_mode = "executor"
            cur_executor1 = "zero"
            lr = 7e-6
        elif mid == "hier_exec":
            cur_train_mode = "executor"
            cur_executor1 = "rnn"
            lr = 5e-6
        elif mid == "both":
            cur_train_mode = "both"
            cur_executor1 = "rnn"
            lr = 6e-6
        else:
            raise ValueError("This mid does not exist.")

        dict_entry = model_dicts.get(key, [])
        filtered_df = trained_models_df[
            (trained_models_df["rule"].isin([cur_rule]))
            & (trained_models_df["coach1"] == cur_coach1)
            & (trained_models_df["executor1"] == cur_executor1)
            & (trained_models_df["train_mode"] == cur_train_mode)
            & (trained_models_df["lr"] == lr)
        ]

        for _, row in filtered_df.iterrows():
            if len(dict_entry) >= NUM_TRIALS:
                continue

            row_id = row["ID"]
            row_best_iteration = int(row["best_iteration"])
            best_exec_name = f"best_exec_checkpoint_{row_best_iteration}.pt"
            best_coach_name = f"best_coach_checkpoint_{row_best_iteration}.pt"
            entry = {
                "rule": row["rule"],
                "model": row["Name"],
                "run_path": f"apjacob/adapt-minirts/{row_id}",
                "best_exec": best_exec_name,
                "best_coach": best_coach_name,
            }
            dict_entry.append(entry)

        model_dicts[key] = dict_entry
    for k, v in base_model_dicts.items():
        if k in model_dicts:
            if not model_dicts[k]:
                model_dicts[k] = base_model_dicts[k]
        else:
            model_dicts[k] = base_model_dicts[k]


def populate_pop_model_dicts():
    full_results_df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results/adapt-minirts-pop-exps.csv",
        )
    )

    # Filter out iterations less than K
    trained_models_df = full_results_df[(full_results_df["max_iterations"] >= 1700)]

    for _, row in trained_models_df.iterrows():
        rule_series = (
            row["rule_series"].replace('"', "").replace("[", "").replace("]", "")
        )
        key = f"ft_pop[{rule_series}]"
        dict_entry = model_dicts.get(key, [])
        if len(dict_entry) >= NUM_TRIALS:
            continue

        row_id = row["ID"]
        entry = {
            "rule_series": row["rule_series"],
            "model": row["Name"],
            "run_path": f"apjacob/adapt-minirts-pop/{row_id}",
        }
        pop_rules = rule_series.split(",")
        max_iter = max([int(row[f"best_iteration_{rule}"]) for rule in pop_rules])
        for rule in pop_rules:
            rule_row_best_iteration = int(row[f"best_iteration_{rule}"])
            best_rule_exec_name = (
                f"best_exec_checkpoint_{rule}_{rule_row_best_iteration}.pt"
            )
            best_rule_coach_name = (
                f"best_coach_checkpoint_{rule}_{rule_row_best_iteration}.pt"
            )
            entry[f"best_exec_{rule}"] = best_rule_exec_name
            entry[f"best_coach_{rule}"] = best_rule_coach_name

            if rule_row_best_iteration == max_iter:
                best_rule_exec_name = (
                    f"best_exec_checkpoint_{rule}_{rule_row_best_iteration}.pt"
                )
                entry[f"best_exec"] = best_rule_exec_name

        dict_entry.append(entry)

        model_dicts[key] = dict_entry


populate_fixed_model_dicts()
populate_pop_model_dicts()
print(pprint.pprint(model_dicts))
