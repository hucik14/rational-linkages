import subprocess
import os
import pytest

# Path to the directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(CURRENT_DIR, '..', 'examples')

# List all Python scripts in the specified directory
scripts = [f for f in os.listdir(SCRIPTS_DIR) if f.endswith('.py')]


@pytest.mark.parametrize("script", scripts)
def test_examples(script):
    script_path = os.path.join(SCRIPTS_DIR, script)

    # Check if the first line is "# NOT TESTED"
    with open(script_path, 'r') as file:
        first_line = file.readline().strip()
        if first_line == '# NOT TESTED':
            pytest.skip(f"Script {script} is marked as not to be tested")

    # Run the script and check for errors
    try:
        result = subprocess.run(['python', script_path],
                                check=True,
                                capture_output=True,
                                text=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"{script} failed with error: {e.stderr}\nOutput: {e.stdout}")


# Run the tests if this script is executed directly (for local testing)
if __name__ == "__main__":
    pytest.main([__file__])
