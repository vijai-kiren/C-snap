"""
Compiler Simulation Driver Program
This program brings together the compiler simulation and visualization components.
"""

# Import necessary modules
import sys
import tkinter as tk
from compiler_simulation import run_compiler_simulation
from compiler_visualization import CompilerVisualizer

def run_console_demo():
    """Run a simple console demonstration of the compiler."""
    example_code = """
    int main() {
        // Declare variables
        int a = 5;
        int b = 7;
        int result;
        
        // Perform calculations
        result = a + b * 2;
        
        // Conditional logic
        if (result > 15) {
            return 1;
        } else {
            return 0;
        }
    }
    """
    
    print("==== COMPILER SIMULATION DEMO ====")
    print("This demo shows all stages of compilation for a simple C-like program.")
    print("\nPress Enter to begin...")
    input()
    
    # Run the simulation
    run_compiler_simulation(example_code)
    
    print("\nCompilation complete!")

def run_gui():
    """Launch the GUI compiler visualizer."""
    root = tk.Tk()
    app = CompilerVisualizer(root)
    root.mainloop()

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--console":
        run_console_demo()
    else:
        run_gui()

if __name__ == "__main__":
    main()