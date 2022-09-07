import json
import os
import constants


class CommitHeadInfo:
    def __init__(self):
        self.head_pointer_file = os.path.join(constants.COMMITS_FOLDER, constants.COMMIT_HEAD_POINTER_FILE)
        if(os.path.exists(self.head_pointer_file)):
            with open(self.head_pointer_file, "r") as f:
                self.head_pointer = json.load(f)
        else:
            raise FileNotFoundError(self.head_pointer_file)

        self.global_head = self.head_pointer["global_head"]
        self.global_tail = self.head_pointer["global_tail"]
        self.current_head = self.head_pointer["current_head"]
        self.global_head_model_path = self.head_pointer["global_head_model"]
        self.global_tail_model_path = self.head_pointer["global_tail_model"]

    def update(self, field, value):
        self.head_pointer[field] = value
        with open(self.head_pointer_file, "w") as f:
                f.write(json.dumps(self.head_pointer, indent=4, sort_keys=True))

        # Refresh pointers after writing updates
        self.__init__()