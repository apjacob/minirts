import yaml
import itertools
import random

l = ['swordman', 'spearman', 'cavalry', 'archer', 'dragon']
full_list = list(itertools.permutations(l))
random.shuffle(full_list)

train_split = full_list[:80]
valid_split = full_list[80:90]
test_split = full_list[90:]

with open('train_permute.yaml', 'w') as f:
    yaml.dump(train_split, f)

with open('valid_permute.yaml', 'w') as f:
    yaml.dump(valid_split, f)

with open('test_permute.yaml', 'w') as f:
    yaml.dump(test_split, f)