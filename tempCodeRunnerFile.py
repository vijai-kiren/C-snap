# # ========== GUI ==========
# class CompilerGUI:
#     def __init__(self, root):
#         self.root = root
#         root.title("Compiler Simulator")

#         self.code_input = tk.Text(root, height=10, width=80)
#         self.code_input.pack(pady=10)

#         self.run_button = tk.Button(root, text="Compile", command=self.run)
#         self.run_button.pack()

#         self.clear_button = tk.Button(root, text="Clear", command=self.clear_all)
#         self.clear_button.pack()

#         self.output = tk.Text(root, height=20, width=80)
#         self.output.pack(pady=10)

#     def clear_all(self):
#         self.code_input.delete("1.0", tk.END)
#         self.output.delete("1.0", tk.END)

#     def run(self):
#         code = self.code_input.get("1.0", tk.END)
#         try:
#             tokens = tokenize(code)
#             ast = parse(tokens)
#             errors = semantic_check(ast)
#             if errors:
#                 self.output.delete("1.0", tk.END)
#                 self.output.insert(tk.END, "Semantic Errors:\n" + "\n".join(errors))
#                 return
#             ir = generate_ir(ast)
#             optimized_ir = optimize_ir(ir)
#             final_code = generate_code(optimized_ir)

#             self.output.delete("1.0", tk.END)
#             self.output.insert(tk.END, "Tokens:\n" + str(tokens) + "\n\n")
#             self.output.insert(tk.END, "AST:\n" + pretty_print_ast(ast) + "\n")
#             self.output.insert(tk.END, "IR:\n" + "\n".join(map(str, ir)) + "\n\n")
#             self.output.insert(tk.END, "Optimized IR:\n" + "\n".join(map(str, optimized_ir)) + "\n\n")
#             self.output.insert(tk.END, "Generated Code:\n" + "\n".join(final_code))
#         except Exception as e:
#             self.output.delete("1.0", tk.END)
#             self.output.insert(tk.END, f"Error: {str(e)}")
