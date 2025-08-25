from memory import ShortTermMemory, LongTermMemory, EmbeddingManager, build_context
import threading
import time
# Module-level singletons for heavy objects
short_mem = ShortTermMemory()
long_mem = LongTermMemory()
embedder = EmbeddingManager()

def store_in_long_term_memory(long_mem, info_type, name, path):
    """
    Store important information in long-term memory.
    info_type: 'folder' or 'file'
    name: logical name (e.g. 'downloads')
    path: full path
    """
    if info_type == 'folder':
        long_mem.remember_folder(name, path)
    elif info_type == 'file':
        long_mem.remember_file(name, path)

def get_important_memory_context(long_mem):
    """
    Return a string summarizing important aspects of long-term memory for Gemini context.
    """
    context = []
    for folder in ['downloads', 'documents', 'desktop', 'pictures', 'music', 'videos']:
        path = long_mem.get_folder(folder)
        if path:
            context.append(f"{folder}: {path}")
    # You can add more logic for files or other info here
    if context:
        return "Known folders: " + ", ".join(context) + ". "
    return ""
from memory import ShortTermMemory, LongTermMemory
import re
def is_command_dangerous(command):
    # Patterns for dangerous or resource-intensive commands
    dangerous_patterns = [
        r"\bdel\b|\bremove-item\b|\brmdir\b|\bremove-directory\b|\bformat\b|\bshutdown\b|\bcopy\b|\bmove\b|\brename\b|\bmkfs\b|\bchkdsk\b|\breg\b|\bsc\b|\bnet user\b|\bnet localgroup\b|\bnet accounts\b|\bnet share\b|\bnet use\b|\bnet stop\b|\bnet start\b|\bnet session\b|\bnet computer\b|\bnet group\b|\bnet config\b|\bnet file\b|\bnet print\b|\bnet time\b|\bnet view\b|\bnetstat\b|\btaskkill\b|\bkill\b|\bshutdown\b|\brestart\b|\bpoweroff\b|\breboot\b",
        r"/s\b|/q\b|/f\b|/purge\b|/force\b|/y\b",  # forceful or recursive flags
        r"\bformat\b|\bcleanmgr\b|\bdefrag\b|\bcompact\b|\brobocopy\b|\bxcopy\b",
        r"\bGet-ChildItem\b.*-Recurse",
        r"\bfindstr\b.* /s",
        r"\bfor /r\b",
        r"\bRemove-Item\b.*-Recurse",
        r"\bRemove-Item\b.*-Force",
        r"\bRemove-Item\b.*-Recurse.*-Force",
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return True
    return False
import sys
import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from command import execute_command


def get_gemini_response(client, model, history):
    contents = []
    for entry in history:
        contents.append(
            types.Content(
                role=entry['role'],
                parts=[types.Part.from_text(text=entry['content'])],
            )
        )
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=0,
        ),
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(text=(
                "You are an agent that generates Windows command prompt or PowerShell commands. "
                "Your first priority is to build knowledge about the system before taking any action: always check the current directory, list folders and files, and if a folder or file is not found, search recursively from the user's home directory or root. "
                "Never use hardcoded paths or <username> or any placeholder—always use the current working directory context or discovered absolute paths. "
                "Only use commands and tools that are available by default on Windows. Do NOT use Unix tools like head, awk, grep, cat, tail, sed, etc. "
                "If you need to sort, filter, or select files, use native Windows commands or PowerShell cmdlets such as Get-ChildItem, Sort-Object, Select-Object, etc. "
                "If a task requires analyzing command output before proceeding (such as checking if a file exists), set 'done' to false and wait for the output before giving the next command. "
                "Before suggesting a command, always check if it is dangerous or resource-intensive (such as deleting files, formatting drives). If the command is dangerous, warn the user and ask for confirmation before proceeding. "
                "Before acting, always try to understand the system and environment: check the current directory, list folders/files, and check for the existence of required files or folders. If a folder or file is not found, search recursively using commands like 'dir /s /b' or PowerShell's Get-ChildItem -Recurse. "
                "Whenever you discover important information about the system (such as the path to a folder, a file, or an environment variable), explicitly instruct the agent to store this information in long-term memory for future use. "
                "Respond ONLY in JSON format as: { \"command\": \"<windows_command>\", \"done\": <true|false> }. "
                "If the command is ready to execute and no further analysis is needed, set 'done' to true. "
                "Do not include any explanation or extra text."
            )),
        ],
    )
    response = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text is not None:
            response += chunk.text
    try:
        return json.loads(response)
    except Exception as e:
        print("Error parsing Gemini response as JSON:", e)
        print("Raw response:", response)
        return {"command": "", "done": True}

def is_command_executable(command):
    if '<' in command and '>' in command:
        return False
    return bool(command.strip())


def main():
    def get_user_approval(command):
        print(f"\nWARNING: The following command may be dangerous or resource-intensive:\n{command}")
        approval = input("Do you want to proceed? (yes/no): ").strip().lower()
        return approval == "yes"
    if len(sys.argv) < 2:
        print("Usage: clai <your request>")
        sys.exit(1)
    user_input = " ".join(sys.argv[1:])
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash-lite"
    history = [
        {"role": "user", "content": user_input}
    ]
    while True:
        # Spinner for context building
        loading_ctx = show_loading("Building context...")
        context = build_context(user_input, short_mem, long_mem, embedder)
        loading_ctx.set()
        print()
        # Add context to the prompt
        history[-1]["content"] = context + "\n" + history[-1]["content"]
        print("\nPrompt sent to Gemini:\n" + history[-1]["content"] + "\n")

        loading = show_loading()
        gemini_resp = get_gemini_response(client, model, history)
        loading.set()  # after response
        print()  # for newline

        command = gemini_resp.get("command", "")
        done = gemini_resp.get("done", True)
        print(f"Gemini generated command: {command}")

        safe = not is_command_dangerous(command)
        if not is_command_executable(command):
            print("Command is not executable or contains placeholders. Sending output back to Gemini for clarification.")
            history.append({"role": "model", "content": f"The command you provided is not executable: {command}. Please provide a fully executable Windows command with no placeholders."})
            continue

        if not safe:
            approved = get_user_approval(command)
            if not approved:
                print("Command execution cancelled by user.")
                history.append({
                    "role": "model",
                    "content": f"The previous command was too dangerous or resource-intensive: {command}. Please suggest a safer, less risky, or less resource-intensive alternative that achieves the same goal. may be with multiple commands we can get what we want."
                })
                continue

        try:
            output = execute_command(command)
        except Exception as e:
            output = f"ERROR: {str(e)}"
            print("\nCommand Error:\n", output)
            history.append({"role": "model", "content": f"The command '{command}' resulted in an error: {output}. Please suggest an alternative or fix the command. Respond in JSON format."})
            continue
        print("\nCommand Output:\n", output)

        # Update short-term memory
        short_mem.update(command, output)
        # Log command in long-term memory with embedding
        cmd_emb = embedder.encode([command])[0]
        long_mem.log_command(command, output, embedding=cmd_emb)

        # Example: If Gemini instructs to store info, call store_in_long_term_memory(long_mem, info_type, name, path)
        # You can parse Gemini's response or add a convention for this.

        history.append({"role": "model", "content": command})
        history.append({"role": "user", "content": f"The output of the command '{command}' was: {output}. What should I do next? Respond in the same JSON format. The previous command was {'safe' if safe else 'dangerous'}."})

        if done:
            break

    long_mem.close()

def ensure_api_key():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found. Please enter your Gemini API key.")
        api_key = input("Enter your GEMINI_API_KEY: ").strip()
        with open(".env", "w") as f:
            f.write(f"GEMINI_API_KEY={api_key}\n")
        print(".env file created.")
    return api_key

def show_loading(message="Waiting for Gemini response..."):
    stop_event = threading.Event()
    def loader():
        spinner = ['|', '/', '-', '\\']
        idx = 0
        while not stop_event.is_set():
            sys.stdout.write(f"\r{message} {spinner[idx % len(spinner)]}")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.15)
        sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")
        sys.stdout.flush()
    t = threading.Thread(target=loader)
    t.start()
    return stop_event

if __name__ == "__main__":
    ensure_api_key()
    main()
