# This file is used to demonstrate Git branching and merge conflicts.

# List of favorite colors - add yours below!
favorite_colors = [
    "blue",
    "green",
    # Add your favorite color below this line
    "yellow"
]

def display_info():
    """Prints project information and the list of favorite colors."""
    print(f"Project: {project_name} (Version: {version})")
    print("\nFavorite Colors:") # Use double backslash for newline in string literal
    if favorite_colors:
        for color in favorite_colors:
            print(f"- {color}")
    else:
        print("No colors added yet.")

if __name__ == "__main__":
    display_info()
