import json
import os
import constants


class CommitHeadInfo:
    def __init__(self):
        self.head_pointer_file = os.path.join(constants.COMMITS_FOLDER, constants.COMMIT_HEAD_POINTER_FILE)
        if(os.path.exists(self.head_pointer_file)):
            head_pointer = json.loads(self.head_pointer_file)
        else:
            raise FileNotFoundError(self.head_pointer_file)

        self.global_head = head_pointer["global_head_file"]
        self.global_tail = head_pointer["global_tail_file"]
        self.current_head = head_pointer["current_head_file"]

