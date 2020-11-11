import wandb
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("apjacob/adapt-minirts", per_page=50)

results = []
merged_result_dict = {}
for run in tqdm(runs):
    summary = run.summary._json_dict
    config = run.config

    if "max_iterations" not in summary:
        continue

    if summary["max_iterations"] > 1500 and config["rule"] == 80:
        coach = config["coach1"]
        lr = config["lr"]
        cur_executor1 = config["executor1"]
        cur_train_mode = config["train_mode"]

        data = run.history()
        if cur_train_mode == "coach" and cur_executor1 == "rnn" and lr == 3e-6:
            key = "ft_coach[80]"
        elif cur_train_mode == "executor" and cur_executor1 == "zero" and lr == 7e-6:
            key = "ft_zero[80]"
        elif cur_train_mode == "executor" and cur_executor1 == "rnn" and lr == 5e-6:
            key = "ft_hier_exec[80]"
        elif cur_train_mode == "both" and cur_executor1 == "rnn" and lr == 6e-6:
            key = "ft_both[80]"
        else:
            continue

        l = merged_result_dict.get(key, [])
        point = data[["_step", "Train/Agent-1/Win"]]
        l.append(point)
        merged_result_dict[key] = l

        id = run.id
        name = run.name
        entry = {**summary, **config, "ID": id, "Name": name}
        results.append(entry)


sns.lineplot(
    data=pd.DataFrame.from_records(merged_result_dict["ft_coach[80]"]),
    x="_step",
    y="Train/Agent-1/Win",
)
plt.show()

full_results_df = pd.DataFrame.from_records(results)
