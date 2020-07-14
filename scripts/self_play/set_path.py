# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys


def append_sys_path():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    pytube = os.path.join(root, 'tube')
    if pytube not in sys.path:
        sys.path.append(pytube)

    tube = os.path.join(root, 'build', 'tube')
    if tube not in sys.path:
        sys.path.append(tube)

    minirts = os.path.join(root, 'build', 'game')
    if minirts not in sys.path:
        sys.path.append(minirts)

    script = os.path.join(root, 'scripts')
    if script not in sys.path:
        sys.path.append(script)

if __name__ == '__main__':
    # import env for testing
    #import torch
    append_sys_path()
    import tube
    import pytube
    import minirts
    import behavior_clone
