import subprocess

def execute_command(command):
    """
    Executes a shell command and returns its output (stdout and stderr).
    Args:
        command (str): The command to execute.
    Returns:
        output (str): The combined output of stdout and stderr.
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr