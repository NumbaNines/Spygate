#!/usr/bin/env python
"""Script to create additional GitHub repositories for the project."""
import os
import sys
from typing import Dict, List

from github import Github
from github.GithubException import GithubException

# Configuration
REPOS = [
    {
        "name": "spygate-models",
        "description": "Machine learning models for Spygate",
        "topics": ["machine-learning", "computer-vision", "sports-analytics"],
        "private": True,
    },
    {
        "name": "spygate-docs",
        "description": "Documentation for Spygate",
        "topics": ["documentation", "sports-analytics"],
        "private": True,
    },
    {
        "name": "spygate-data",
        "description": "Data processing and analysis tools for Spygate",
        "topics": ["data-processing", "sports-analytics", "python"],
        "private": True,
    },
]


def create_repository(github: Github, config: dict) -> None:
    """Create a GitHub repository with the given configuration."""
    try:
        repo = github.get_user().create_repo(
            name=config["name"],
            description=config["description"],
            private=config.get("private", True),
            has_issues=True,
            has_wiki=True,
            has_projects=True,
            auto_init=True,
        )

        # Add topics
        if "topics" in config:
            repo.replace_topics(config["topics"])

        print(f"Successfully created repository: {config['name']}")
        print(f"URL: {repo.html_url}")

    except GithubException as e:
        if e.status == 422:  # Repository already exists
            print(f"Repository {config['name']} already exists")
        else:
            print(f"Error creating repository {config['name']}: {e}")


def create_all_repositories(token: str) -> None:
    """Create all configured repositories."""
    try:
        github = Github(token)

        # Test authentication
        github.get_user().login

        print("Authenticated successfully with GitHub")
        print(f"Creating {len(REPOS)} repositories...")

        for repo_config in REPOS:
            create_repository(github, repo_config)

    except GithubException as e:
        print(f"GitHub API Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Get GitHub token from environment
    github_token = os.getenv("GITHUB_TOKEN")

    if not github_token:
        print("Error: GITHUB_TOKEN environment variable not set")
        print("Please set your GitHub personal access token:")
        print("export GITHUB_token = os.getenv("TOKEN", "")")
        sys.exit(1)

    create_all_repositories(github_token)
