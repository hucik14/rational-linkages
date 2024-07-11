import git
from datetime import datetime

# Constants (replace with your actual values)
REPO_PATH = 'C:/Gitlab/rational-linkages'  # Path to your local Git repository
SOURCE_BRANCH = 'develop'  # Replace with your source branch
TARGET_BRANCH = 'main'  # Replace with your target branch
CHANGELOG_FILE = 'CHANGELOG.md'  # Name of the changelog file


def get_changelog_commits(commits):
    changelog_commits = {
        "added": [],
        "removed": [],
        "changed": [],
        "fixed": [],
        "deprecated": [],
        "security": []
    }
    for commit in commits:
        message = commit.message
        if 'Changelog' in message:
            lines = message.split('\n')
            title = lines[0]
            description = '\n'.join(lines[1:]).strip()
            commit_date = datetime.fromtimestamp(commit.committed_date).strftime(
                '%Y-%m-%d')
            author = commit.author.name
            commit_hash = commit.hexsha[:8]

            # Determine the category based on commit message
            category = None
            if "Changelog: added" in commit.message:
                category = "added"
            elif "Changelog: removed" in commit.message:
                category = "removed"
            elif "Changelog: changed" in commit.message:
                category = "changed"
            elif "Changelog: fixed" in commit.message:
                category = "fixed"
            elif "Changelog: deprecated" in commit.message:
                category = "deprecated"
            elif "Changelog: security" in commit.message:
                category = "security"
            else:
                print(f"Uncategorized commit: {title}")

            if category:
                changelog_commits[category].append({
                    'title': title,
                    'description': description,
                    'commit_date': commit_date,
                    'commit_hash': commit_hash,
                })

    return changelog_commits


def format_release_notes(version, release_date, changelog_commits):
    notes = [f"## {version} ({release_date})\n"]

    for category, commits in changelog_commits.items():
        if commits:
            notes.append(
                f"### {category} ({len(commits)} change{'s' if len(commits) > 1 else ''})\n")
            for commit in commits:
                notes.append(
                    f"- [{commit['title']}](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/{commit['commit_hash']})"
                )
            notes.append("\n")

    return "\n".join(notes)


def write_changelog(file_path, release_notes):
    with open(file_path, 'w') as f:
        header = "# Changelog\n\n"
        f.write(header)
        f.write(release_notes)


def main():
    repo = git.Repo(REPO_PATH)

    # Ensure local repository is up-to-date
    repo.remotes.origin.fetch()

    # Fetch commits between source_branch and target_branch
    try:
        commits = list(repo.iter_commits(f'{TARGET_BRANCH}..{SOURCE_BRANCH}'))
    except git.exc.GitCommandError as e:
        print(f"Error fetching commits: {e}")
        return

    if not commits:
        print("No new commits found between the branches.")
        return

    print(f"Found {len(commits)} commits between {TARGET_BRANCH} and {SOURCE_BRANCH}.")

    changelog_commits = get_changelog_commits(commits)
    if not any(changelog_commits.values()):
        print("No 'Changelog' tagged commits found.")
        return

    release_date = datetime.now().strftime('%Y-%m-%d')
    version = "1.6.0"  # Replace with your versioning logic

    release_notes = format_release_notes(version, release_date, changelog_commits)
    write_changelog(CHANGELOG_FILE, release_notes)

    print(f"Release notes written to {CHANGELOG_FILE}")


if __name__ == "__main__":
    main()