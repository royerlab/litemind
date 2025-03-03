import yaml


def default_folder_scanning_parameters(allowed_extensions, excluded_files):
    # Set default values for allowed_extensions and excluded_files:
    if allowed_extensions is None:
        allowed_extensions = [
            ".py",
            ".md",
            ".txt",
            ".toml",
            "LICENSE",
            ".tests",
            ".html",
        ]
    # Set default values for excluded_files:
    if excluded_files is None:
        excluded_files = [
            "litemind.egg-info",
            "dist",
            "build",
        ]
    return allowed_extensions, excluded_files


def parse_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data
