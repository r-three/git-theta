"""Plugins for Model Merging.

Note:
  In order to dynamically create the menu of possible actions that describe what
  each plug-in does, the plugins get imported at the start of the merge tool.
  Therefore, plug-ins must not have slow side-effects that happen at import-time.
"""

from git_theta.merges.base import Merge, all_merge_handlers
