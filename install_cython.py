"""
Script to help install and compile the Cython extension for Reversi.
"""
import os
import sys
import subprocess
import platform

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text} ".center(60, '='))
    print("=" * 60 + "\n")

def run_command(command, cwd=None):
    """Run a shell command and return its output."""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        universal_newlines=True
    )
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Error: {stderr.strip()}")
        return False
    
    print(stdout.strip())
    return True

def install_build_tools():
    """Install build tools for Cython compilation."""
    system = platform.system().lower()
    
    if system == 'windows':
        print_header("Windows Detected")
        print("Please install Microsoft C++ Build Tools from:")
        print("https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        print("\nMake sure to select 'Desktop development with C++' workload during installation.")
        input("\nPress Enter after you've installed the build tools...")
        return True
    
    elif system == 'linux':
        print_header("Linux Detected")
        print("Installing build essentials...")
        return run_command("sudo apt-get update && sudo apt-get install -y build-essential python3-dev")
    
    elif system == 'darwin':  # macOS
        print_header("macOS Detected")
        print("Installing Xcode command line tools...")
        return run_command("xcode-select --install")
    
    else:
        print(f"Unsupported operating system: {system}")
        return False

def install_cython():
    """Install Cython if not already installed."""
    print_header("Installing Cython")
    return run_command(f"{sys.executable} -m pip install --upgrade pip setuptools wheel cython numpy")

def compile_cython():
    """Compile the Cython extension."""
    print_header("Compiling Cython Extension")
    
    # Create the build directory if it doesn't exist
    os.makedirs("build", exist_ok=True)
    
    # Compile the extension
    return run_command(f"{sys.executable} setup_cython.py build_ext --inplace")

def main():
    """Main installation function."""
    print_header("Reversi Cython Optimizations Setup")
    
    # Step 1: Install build tools if needed
    if not install_build_tools():
        print("Failed to install build tools.")
        return 1
    
    # Step 2: Install Cython and dependencies
    if not install_cython():
        print("Failed to install Cython and dependencies.")
        return 1
    
    # Step 3: Compile the Cython extension
    if not compile_cython():
        print("Failed to compile Cython extension.")
        return 1
    
    print_header("Installation Complete!")
    print("The optimized Cython board implementation is now ready to use.")
    print("\nYou can test it by running:")
    print("  python test_optimized_board.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
