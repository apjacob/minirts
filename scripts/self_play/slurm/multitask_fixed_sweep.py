import os, sys
sys.path.append('../')

# Load the file names
dataDir = "../../../data/word_count/"
fnames = os.listdir(dataDir)

my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])

WANDB_MODE=dryrun python multitask-fixed.py --coach1 rnn500 --coach2 rnn500 --executor1 rnn --executor2 rnn --lr 1e-6 --train_epochs 2000 --sampling_freq 1.0 --seed 777 --tb_log 1 --save_folder /home/gridsan/apjacob/save --rule_dir /home/gridsan/apjacob/minirts/scripts/self_play/rules/ --wandb_dir /home/gridsan/apjacob/wandb/ --pg ppo --ppo_epochs 4 --train_batch_size 32 --num_rb=25 --num_sp=0 --train_mode=coach --rule 21
# Assign indices to this process/task
my_fnames = fnames[my_task_id - 1:len(fnames):num_tasks]

for fname in my_fnames:
    # Read in file and clean the text
    f = open(dataDir + fname, 'r', encoding='utf-8')
    text = cleantext(f.readlines())

    # Count number of times each word appears
    counts = countwords(text)

    # Sort and print the top 5 words with their counts
    top5 = sorted(counts, key=counts.get, reverse=True)[:5]
    str = ''
    for k in top5:
        str = str + "%s: %s" % (k, counts[k]) + "; "

    print(str[:-2])