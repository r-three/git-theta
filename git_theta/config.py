import dataclasses
import fnmatch
import json
import os
import re
from collections import OrderedDict
from typing import Dict, List, Tuple

import git

from git_theta import git_utils


@dataclasses.dataclass
class PatternAttributes:
    pattern: str
    attributes: str

    @classmethod
    def from_line(cls, line):
        # TODO(bdlester): Revisit this regex to see if it when the pattern
        # is escaped due to having spaces in it.
        match = re.match(
            r"^\s*(?P<pattern>[^\s]+)\s+(?P<attributes>.*)$", line.rstrip()
        )
        if not match:
            raise ValueError(f"{line} is an invalid git attribute line")
        return cls(pattern=match.group("pattern"), attributes=match.group("attributes"))

    def add_attribute(self, attribute):
        if attribute not in self.attributes:
            self.attributes = f"{self.attribute.rstrip()} {attribute}"

    def serialize(self):
        return f"{self.pattern} {self.attributes}"


class GitAttributesFile:
    def __init__(self, repo: git.Repo):
        self.repo = repo
        self.file = os.path.join(repo.working_dir, ".gitattributes")
        self.data = GitAttributesFile.read(self.file)

    @classmethod
    def read(cls, file: str):
        """
        Read contents of this repo's .gitattributes file

        Parameters
        ----------
        file
            Path to .gitattributes file

        Returns
        -------
        List[str]
            lines in .gitattributes file
        """
        if os.path.exists(file):
            with open(file, "r") as f:
                return [PatternAttributes.from_line(line) for line in f]
        else:
            return []

    def write(self):
        """
        Write list of attributes to this repo's .gitattributes file
        """
        with open(self.file, "w") as f:
            f.write(
                "\n".join([pattern_attrs.serialize() for pattern_attrs in self.data])
            )
            # End file with newline.
            f.write("\n")

    def add_theta(self, pattern: str) -> List[str]:
        """Set filter, merge, and diff attributes to theta for `pattern`.

        Parameters
        ----------
        pattern:
            The pattern we are adding theta attributes for
        """
        pattern = git_utils.get_relative_path_from_root(self.repo, pattern)
        pattern_found = False
        for pattern_attrs in self.data:
            if pattern == pattern_attrs.pattern:
                pattern_attrs.add("filter=theta")
                pattern_attrs.add("merge=theta")
                pattern_attrs.add("diff=theta")
                pattern_found = True
        # If we don't find a matching pattern, add a new line that covers this pattern
        if not pattern_found:
            self.data.append(
                PatternAttributes.from_line(
                    f"{pattern} filter=theta merge=theta diff=theta"
                )
            )

    def is_theta_tracked(self, path: str) -> bool:
        return any(
            [
                fnmatch.fnmatchcase(path, pattern_attr.pattern)
                and "filter=theta" in pattern_attr.attributes
                for pattern_attr in self.data
            ]
        )


@dataclasses.dataclass
class Config:
    @classmethod
    def from_dict(cls, params: Dict) -> "Config":
        fields = [field.name for field in dataclasses.fields(cls)]
        return cls(**{k: v for k, v in params.items() if k in fields})

    def serialize(self) -> Dict:
        return dataclasses.asdict(self, dict_factory=OrderedDict)


@dataclasses.dataclass
class PatternConfig(Config):
    pattern: str
    checkpoint_format: str


@dataclasses.dataclass
class RepoConfig(Config):
    parameter_atol: float = 1e-8
    parameter_rtol: float = 1e-5
    lsh_signature_size: int = 16
    lsh_threshold: float = 1e-6
    lsh_pool_size: int = 10_000
    max_concurrency: int = -1


class ThetaConfigFile:
    def __init__(self, repo):
        self.repo = repo
        self.file = os.path.join(repo.working_dir, ".thetaconfig")
        self.repo_config, self.pattern_configs = ThetaConfigFile.read(self.file)

    @classmethod
    def read(cls, file) -> Tuple[RepoConfig, List[PatternConfig]]:
        """
        Read contents of this repo's .thetaconfig file

        Returns
        -------
        repo_config: RepoConfig
            Repository-level configuration parameters

        pattern_configs: List[PatternConfig]
            Pattern-level configuration parameters
        """
        if os.path.exists(file):
            with open(file, "r") as f:
                config = json.load(f)
        else:
            config = {"repo": {}, "patterns": []}

        repo_config = RepoConfig.from_dict(config["repo"])
        pattern_configs = [PatternConfig.from_dict(d) for d in config["patterns"]]
        return repo_config, pattern_configs

    def write(self):
        """
        Write a dictionary config to this repo's .thetaconfig file

        Parameters
        ----------
        config_file:
            Path to this repo's .thetaconfig file

        config:
            Configuration dictionary to write to .thetaconfig
        """
        with open(self.file, "w") as f:
            json.dump(self.serialize(), f, indent=4)

    def serialize(self):
        repo_config = self.repo_config.serialize()
        pattern_configs = [pc.serialize() for pc in self.pattern_configs]
        config = {"repo": repo_config, "patterns": pattern_configs}
        return config

    def add_pattern(self, config: Dict):
        pattern_config = PatternConfig.from_dict(config)
        self.pattern_configs.append(pattern_config)

    def get_config(self, path: str) -> Dict:
        config = {}
        for pc in self.pattern_configs:
            if fnmatch.fnmatchcase(path, pc.pattern):
                config.update(pc.serialize())
        if "pattern" in config:
            config.pop("pattern")
        return config
