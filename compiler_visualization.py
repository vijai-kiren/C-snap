import tkinter as tk
from tkinter import ttk, scrolledtext
from tkinter import messagebox
import sys
from io import StringIO

class CompilerVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Compiler Simulation Visualizer")
        self.root.geometry("900x700")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs for each compilation stage
        self.tabs = {
            "Source Code": self._create_text_tab("Source Code"),
            "Lexical Analysis": self._create_text_tab("Lexical Analysis (Tokens)"),
            "Syntax Analysis": self._create_text_tab("Syntax Analysis (AST)"),
            "Semantic Analysis": self._create_text_tab("Semantic Analysis"),
            "IR Generation": self._create_text_tab("Intermediate Code"),
            "Optimization": self._create_text_tab("Optimized Code"),
            "Code Generation": self._create_text_tab("Target Assembly Code")
        }
        
        # Create input frame
        input_frame = ttk.Frame(root)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        # Add source code input
        ttk.Label(input_frame, text="Source Code:").pack(side='left')
        self.source_entry = ttk.Button(input_frame, text="Edit Source", command=self._edit_source)
        self.source_entry.pack(side='left', padx=5)
        
        # Add compile button
        self.compile_btn = ttk.Button(input_frame, text="Compile", command=self._compile_code)
        self.compile_btn.pack(side='left', padx=5)
        
        # Default source code
        self.default_source = """int main() {
    int x = 5;
    int y = 10;
    int z;
    
    // Calculate sum
    z = x + y;
    
    // Check if positive
    if (z > 0) {
        int positive = 1;
        return positive;
    } else {
        return 0;
    }
}"""
        
        # Initialize with default source code
        self.current_source = self.default_source
        self.tabs["Source Code"].insert('1.0', self.default_source)
        
    def _create_text_tab(self, title):
        """Create a tab with a scrolled text widget."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=title)
        
        text = scrolledtext.ScrolledText(tab, wrap=tk.WORD, font=("Courier New", 10))
        text.pack(fill='both', expand=True)
        
        return text
    
    def _edit_source(self):
        """Open a dialog to edit the source code."""
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Edit Source Code")
        edit_window.geometry("600x400")
        
        editor = scrolledtext.ScrolledText(edit_window, wrap=tk.WORD, font=("Courier New", 10))
        editor.pack(fill='both', expand=True, padx=10, pady=10)
        editor.insert('1.0', self.current_source)
        
        def save_and_close():
            self.current_source = editor.get('1.0', 'end-1c')
            self.tabs["Source Code"].delete('1.0', tk.END)
            self.tabs["Source Code"].insert('1.0', self.current_source)
            edit_window.destroy()
        
        save_btn = ttk.Button(edit_window, text="Save", command=save_and_close)
        save_btn.pack(pady=10)
    
    def _compile_code(self):
        """Compile the current source code and display results."""
        # Import the compiler simulation code
        from compiler_simulation import run_compiler_simulation
        
        # Redirect stdout to capture output
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        try:
            # Run the compiler simulation
            results = run_compiler_simulation(self.current_source)
            
            # Display results in each tab
            self._update_lexical_tab(results["stages"]["lexical_analysis"])
            self._update_syntax_tab(results["stages"]["syntax_analysis"])
            self._update_semantic_tab(results["stages"]["semantic_analysis"])
            self._update_ir_tab(results["stages"]["ir_generation"])
            self._update_optimization_tab(results["stages"]["optimization"])
            self._update_codegen_tab(results["stages"]["code_generation"])
            
            # Show compilation success
            tk.messagebox.showinfo("Compilation Complete", "Compilation completed successfully!")
            
        except Exception as e:
            # Show compilation error
           messagebox.showerror("Compilation Error", str(e))
            
        finally:
            # Restore stdout
            sys.stdout = old_stdout
    
    def _update_lexical_tab(self, tokens):
        """Update the lexical analysis tab with token information."""
        tab = self.tabs["Lexical Analysis"]
        tab.delete('1.0', tk.END)
        
        for token in tokens:
            tab.insert(tk.END, f"{token}\n")
    
    def _update_syntax_tab(self, ast):
        """Update the syntax analysis tab with AST information."""
        tab = self.tabs["Syntax Analysis"]
        tab.delete('1.0', tk.END)
        
        def print_ast(node, indent=0):
            tab.insert(tk.END, ' ' * indent + str(node) + '\n')
            for child in node.children:
                print_ast(child, indent + 2)
        
        print_ast(ast)
    
    def _update_semantic_tab(self, semantic_results):
        """Update the semantic analysis tab with semantic information."""
        tab = self.tabs["Semantic Analysis"]
        tab.delete('1.0', tk.END)
        
        errors = semantic_results["errors"]
        if errors:
            tab.insert(tk.END, "Semantic errors found:\n")
            for error in errors:
                tab.insert(tk.END, f"- {error}\n")
        else:
            tab.insert(tk.END, "No semantic errors found.\n")
    
    def _update_ir_tab(self, ir_code):
        """Update the IR generation tab with three-address code."""
        tab = self.tabs["IR Generation"]
        tab.delete('1.0', tk.END)
        
        tab.insert(tk.END, "Three-Address Code:\n")
        for instr in ir_code:
            tab.insert(tk.END, f"{instr}\n")
    
    def _update_optimization_tab(self, optimized_code):
        """Update the optimization tab with optimized code."""
        tab = self.tabs["Optimization"]
        tab.delete('1.0', tk.END)
        
        tab.insert(tk.END, "Optimized Three-Address Code:\n")
        for instr in optimized_code:
            tab.insert(tk.END, f"{instr}\n")
    
    def _update_codegen_tab(self, assembly_code):
        """Update the code generation tab with assembly code."""
        tab = self.tabs["Code Generation"]
        tab.delete('1.0', tk.END)
        
        tab.insert(tk.END, "Assembly Code:\n")
        for line in assembly_code:
            tab.insert(tk.END, f"{line}\n")

# Entry point
def main():
    root = tk.Tk()
    app = CompilerVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()