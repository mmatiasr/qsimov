import subprocess
import os

import unicodedata


def remove_accents(input_string):
    """
    Removes accents from a UTF-8 encoded string.

    Parameters
    ----------
    input_string : str
        The input string to remove accents from.

    Returns
    -------
    str
        The input string with accents removed.

    Examples
    --------
    >>> remove_accents("São Paulo")
    'Sao Paulo'
    """
    return "".join(
        c
        for c in unicodedata.normalize("NFD", input_string)
        if unicodedata.category(c) != "Mn"
    )


def get_git_hash():
    """Get the current git commit hash.

    Returns
    -------
    commit_hash : str
        The current git commit hash.
    """
    commit_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"])
        .decode("utf-8")
        .strip()
    )
    return commit_hash


def get_git_username():
    """Get the current git user name or OS user name if git is not installed.

    Returns
    -------
    username : str
        The current git user name or OS user name.
    """
    try:
        username = (
            subprocess.check_output(["git", "config", "--get", "user.name"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        print("Error: Git user name not found. Falling back to OS user name.")
        username = os.getlogin()

    return remove_accents(username)


def check_git_clean():
    """
    Check if the Git repository has uncommitted changes, ignoring untracked
    files.

    If uncommitted changes are detected, print the changes and prompt the user
    to continue or exit the script.

    Notes
    -----
    This function uses the `git status` command with the `--porcelain` and
    `--untracked-files=no` options to check for uncommitted changes in the
    repository.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        stdout=subprocess.PIPE,
    )
    if result.stdout.strip():
        print("Git repository has uncommitted changes: ")
        print(result.stdout.decode("utf-8"))

        if input("Continue? [y/n] ").lower() != "y":
            exit(1)


def git_commit_and_get_hash(commit_message, files_to_add):
    """Commit the specified files to the git repo and return the commit hash.

    Parameters
    ----------
    commit_message : str
        The commit message.
    files_to_add : list of str
        The files to add to the staging area.

    Returns
    -------
    commit_hash : str
        The commit hash.
    """
    print("The following files will be committed: " + "\n".join(files_to_add))
    print("With the following commit message: " + commit_message + "\n")
    question = input("Continue? (y/n) ")
    if question.lower() != "y":
        return None
    # Add specified files to the staging area
    for file in files_to_add:
        subprocess.run(["git", "add", file], check=True)

    # Commit the changes
    subprocess.run(["git", "commit", "-m", commit_message], check=True)

    # Return the commit hash
    commit_hash = get_git_hash()
    return commit_hash


if __name__ == "__main__":
    current_git_username = get_git_username()
    print("Current Git or OS user name:", current_git_username)
