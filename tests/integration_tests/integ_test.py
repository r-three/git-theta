import git
import os
import pytest
import torch
import subprocess

REPO_NAME = "test_repo"
MODEL_NAME = "pytorch_model.bin"
TEXT_FILE = "hello.txt"


@pytest.fixture(scope="session")
def get_repo(tmp_path_factory):
    repo_path = tmp_path_factory.mktemp(REPO_NAME)

    class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output

    model = Feedforward(2, 10)
    torch.save(model.state_dict(), f"{repo_path}/{MODEL_NAME}")

    with open(f"{repo_path}/{TEXT_FILE}", "w") as f:
        f.write("Create a new text file!")

    subprocess.run("git init", cwd=repo_path)

    return repo_path


def test_repo_creation(get_repo):
    assert os.path.isdir(get_repo)
    assert os.path.exists(os.path.join(get_repo, MODEL_NAME))
    assert os.path.exists(os.path.join(get_repo, TEXT_FILE))
    assert os.path.isdir(os.path.join(get_repo, ".git"))


def test_git_theta_install():
    subprocess.run("git theta install")
    config = git.GitConfigParser(
        git.config.get_config_path("global"), config_level="global", read_only=True
    )
    config_sections = config.sections()
    assert 'filter "lfs"' in config_sections
    assert 'filter "theta"' in config_sections

    lfs_section = dict(config.items('filter "lfs"'))
    assert list(lfs_section.keys()) == ["clean", "smudge", "required"]
    assert lfs_section["clean"] == "git-lfs clean -- %f"
    assert lfs_section["smudge"] == "git-lfs smudge -- %f"
    assert lfs_section["required"]

    theta_section = dict(config.items('filter "theta"'))
    assert list(theta_section.keys()) == ["clean", "smudge", "required"]
    assert theta_section["clean"] == "git-theta-filter clean %f"
    assert theta_section["smudge"] == "git-theta-filter smudge %f"
    assert theta_section["required"]
