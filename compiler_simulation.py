import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import re

# ==============================
# LEXICAL ANALYSIS
# ==============================

KEYWORDS = {'int', 'return', 'if', 'else'}

def tokenize(code):
    TOKEN_SPEC = [
        ('COMMENT_MULTI', r'/\*[\s\S]*?\*/'),      # Multiline comment
        ('COMMENT_SINGLE', r'//.*'),               # Single line comment
        ('NUMBER',    r'\d+'),                # Integer
        ('ID',       r'[a-zA-Z_]\w*'),
        ('OP',       r'(==|!=|<=|>=|[+\-*/=<>])'),
        ('LPAREN',   r'\('),
        ('RPAREN',   r'\)'),
        ('LBRACE',   r'\{'),
        ('RBRACE',   r'\}'),
        ('SEMICOLON',r';'),
        ('COMMA',    r','),
        ('SKIP',     r'[ \t]+'),
        ('NEWLINE',  r'\n'),
        ('MISMATCH', r'.')
    ]
    token_regex = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPEC)
    line_num = 1
    tokens = []
    for mo in re.finditer(token_regex, code):
        kind = mo.lastgroup
        value = mo.group()
        if kind in ['COMMENT_SINGLE', 'COMMENT_MULTI']:
            continue  # Skip comments
        if kind == 'NUMBER':
            tokens.append(('NUMBER', value))
        elif kind == 'ID':
            if value in KEYWORDS:
                tokens.append(('KEYWORD', value))
            else:
                tokens.append(('ID', value))
        elif kind == 'OP':
            tokens.append(('OP', value))
        elif kind in ('LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'SEMICOLON', 'COMMA'):
            tokens.append((kind, value))
        elif kind == 'NEWLINE':
            line_num += 1
        elif kind == 'SKIP':
            continue
        elif kind == 'MISMATCH':
            raise RuntimeError(f'{value!r} unexpected on line {line_num}')
    return tokens

# ==============================
# AST NODE
# ==============================

class ASTNode:
    def __init__(self, type_, value=None, children=None):
        self.type = type_
        self.value = value
        self.children = children or []

    def __repr__(self, level=0):
        ret = '  ' * level + f'{self.type}: {self.value}\n'
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

# ==============================
# PARSER
# ==============================

def parse(tokens):
    i = 0

    def peek():
        return tokens[i] if i < len(tokens) else (None, None)

    def advance():
        nonlocal i
        i += 1

    def expect(kind):
        tok = peek()
        if tok[0] != kind:
            raise RuntimeError(f"Expected token {kind} but got {tok[0]}")
        advance()
        return tok

    def match(kind, value=None):
        tok = peek()
        if tok[0] == kind and (value is None or tok[1] == value):
            advance()
            return True
        return False

    def parse_primary():
        tok = peek()
        if tok[0] == 'NUMBER':
            advance()
            return ASTNode('NUM', tok[1])
        elif tok[0] == 'ID':
            advance()
            return ASTNode('VAR', tok[1])
        elif tok[0] == 'LPAREN':
            advance()
            expr = parse_expression()
            expect('RPAREN')
            return expr
        else:
            raise RuntimeError(f"Unexpected token {tok}")

    def parse_term():
        node = parse_primary()
        while peek()[0] == 'OP' and peek()[1] in ('*', '/'):
            op = peek()[1]
            advance()
            right = parse_primary()
            node = ASTNode('BINOP', op, [node, right])
        return node

    def parse_expression():
        node = parse_arithmetic()

        # Handle comparison operators like <, >, ==, etc.
        while peek()[0] == 'OP' and peek()[1] in ('==', '!=', '<', '>', '<=', '>='):
            op = peek()[1]
            advance()
            right = parse_arithmetic()
            node = ASTNode('BINOP', op, [node, right])
        return node

    def parse_arithmetic():
        node = parse_term()
        while peek()[0] == 'OP' and peek()[1] in ('+', '-'):
            op = peek()[1]
            advance()
            right = parse_term()
            node = ASTNode('BINOP', op, [node, right])
        return node

    def parse_block():
        expect('LBRACE')
        stmts = []
        while peek()[0] != 'RBRACE':
            stmts.append(parse_statement())
        expect('RBRACE')
        return ASTNode('BLOCK', None, stmts)


    def parse_condition():
        expect('LPAREN')
        cond = parse_expression()
        expect('RPAREN')
        return cond

    def parse_if_statement():
        expect('KEYWORD')  # 'if'
        cond = parse_condition()
        then_branch = parse_block()
        else_branch = None
        if match('KEYWORD', 'else'):
            else_branch = parse_block()
        return ASTNode('IF', None, [cond, then_branch, else_branch] if else_branch else [cond, then_branch])

    def parse_main_function():
        expect('KEYWORD')  # int
        expect('ID')       # main
        expect('LPAREN')
        expect('RPAREN')
        body = parse_block()
        return ASTNode('FUNC', 'main', [body])

    def parse_statement():
        tok = peek()

        if tok == ('KEYWORD', 'return'):
            match('KEYWORD', 'return')
            expr = parse_expression()
            expect('SEMICOLON')
            return ASTNode('RETURN', children=[expr])

        elif tok == ('KEYWORD', 'int'):
            match('KEYWORD', 'int')
            ident = expect('ID')[1]
            if match('OP', '='):
                expr = parse_expression()
                expect('SEMICOLON')
                # You can return it as both a declaration and assignment
                return ASTNode('DECL_INIT', ident, [expr])
            else:
                expect('SEMICOLON')
                return ASTNode('DECL', ident)

        elif tok == ('KEYWORD', 'if'):
            return parse_if_statement()

        elif tok[0] == 'ID':
            ident = expect('ID')[1]
            if match('OP', '='):
                expr = parse_expression()
                expect('SEMICOLON')
                return ASTNode('ASSIGN', ident, [expr])
            else:
                raise RuntimeError(f"Expected '=' after identifier '{ident}' but got {peek()}")

        else:
            raise RuntimeError(f"Unexpected token {tok}")

    def parse_program():
        tok = peek()
        if tok == ('KEYWORD', 'int'):
            # Try to parse main function
            lookahead = tokens[i+1] if i+1 < len(tokens) else (None, None)
            if lookahead == ('ID', 'main'):
                return parse_main_function()
        raise RuntimeError(f"Expected 'int main()' function, got {tok}")

    return parse_program()

# ==============================
# SEMANTIC ANALYSIS
# ==============================

def semantic_check(ast):
    symbol_table = {}
    def check(node):
        if node.type == 'DECL':
            if node.value in symbol_table:
                raise RuntimeError(f"Semantic Error: Variable '{node.value}' already declared")
            symbol_table[node.value] = 'int'
        elif node.type == 'ASSIGN':
            if node.value not in symbol_table:
                raise RuntimeError(f"Semantic Error: Variable '{node.value}' not declared")
            check(node.children[0])
        elif node.type == 'VAR':
            if node.value not in symbol_table:
                raise RuntimeError(f"Semantic Error: Variable '{node.value}' not declared")
        elif node.type == 'BINOP':
            check(node.children[0])
            check(node.children[1])
        elif node.type in ('BLOCK', 'PROGRAM'):
            for child in node.children:
                check(child)
        elif node.type == 'IF':
            check(node.children[0])  # condition
            check(node.children[1])  # then
            if len(node.children) == 3:
                check(node.children[2])  # else
        elif node.type == 'FUNC':
            check(node.children[0])  # body (block)
        elif node.type == 'DECL_INIT':
            if node.value in symbol_table:
                raise RuntimeError(f"Semantic Error: Variable '{node.value}' already declared")
            symbol_table[node.value] = 'int'
            check(node.children[0])

    check(ast)

# ==============================
# INTERMEDIATE CODE GENERATION
# ==============================

def generate_ir(ast):
    ir = []
    temp_id = [0]
    label_id = [0]

    def new_temp():
        temp_id[0] += 1
        return f"t{temp_id[0]}"

    def new_label():
        label_id[0] += 1
        return f"L{label_id[0]}"

    def emit(op, a1=None, a2=None, res=None):
        ir.append((op, a1, a2, res))

    def gen(node):
        if node.type == 'NUM':
            tmp = new_temp()
            emit('LOAD_CONST', node.value, None, tmp)
            return tmp
        elif node.type == 'VAR':
            return node.value
        elif node.type == 'BINOP':
            l = gen(node.children[0])
            r = gen(node.children[1])
            tmp = new_temp()
            emit(node.value, l, r, tmp)
            return tmp
        elif node.type == 'ASSIGN':
            val = gen(node.children[0])
            emit('STORE', val, None, node.value)
        elif node.type == 'DECL':
            emit('DECL', None, None, node.value)
        elif node.type == 'DECL_INIT':
            emit('DECL', None, None, node.value)
            val = gen(node.children[0])
            emit('STORE', val, None, node.value)
        elif node.type == 'BLOCK':
            for ch in node.children:
                gen(ch)
        elif node.type == 'RETURN':
            val = gen(node.children[0])
            emit('RETURN', val, None, None)
        elif node.type == 'IF':
            cond = gen(node.children[0])
            label_else = new_label()
            label_end = new_label() if len(node.children) == 3 else None
            emit('JZ', cond, None, label_else)
            gen(node.children[1])  # then branch
            if label_end:
                emit('JMP', None, None, label_end)
            emit('LABEL', None, None, label_else)
            if len(node.children) == 3:
                gen(node.children[2])  # else branch
                emit('LABEL', None, None, label_end)
        elif node.type == 'FUNC':
            emit('LABEL', None, None, node.value)
            gen(node.children[0])  # function body
        elif node.type == 'PROGRAM':
            for ch in node.children:
                gen(ch)


    gen(ast)
    return ir

def ir_to_three_address_code(ir_tuples):
    tac_lines = []

    for instr in ir_tuples:
        op, arg1, arg2, result = instr

        if op == 'LABEL':
            tac_lines.append(f"{result}:")
        elif op == 'LOAD_CONST':
            tac_lines.append(f"{result} = {arg1}")
        elif op == 'ASSIGN':
            tac_lines.append(f"{result} = {arg1}")
        elif op in ('+', '-', '*', '/'):
            tac_lines.append(f"{result} = {arg1} {op.lower()} {arg2}")
        elif op == 'RETURN':
            tac_lines.append(f"return {arg1}")
        elif op == 'IFGOTO':
            tac_lines.append(f"if {arg1} goto {result}")
        elif op == 'GOTO' or op == 'JMP':
            tac_lines.append(f"goto {result}")
        elif op == 'JZ':  # Jump if zero (false)
            tac_lines.append(f"ifFalse {arg1} goto {result}")
        elif op == '>':
            tac_lines.append(f"{result} = {arg1} > {arg2}")
        elif op == '<':
            tac_lines.append(f"{result} = {arg1} < {arg2}")
        elif op == '==':
            tac_lines.append(f"{result} = {arg1} == {arg2}")
        elif op == 'STORE':
            tac_lines.append(f"{result} = {arg1}")
        elif op == 'DECL':
            tac_lines.append(f"declare {result}")
        else:
            tac_lines.append(f"# Unrecognized op: {instr}")

    return "\n".join(tac_lines)


# ==============================
# OPTIMIZATION (constant folding)
# ==============================

def optimize_ir(ir_tuples):
    optimized = []
    temp_values = {}  # Tracks temp values (like {'t1': '5'})
    var_assignments = {}  # Tracks which temp gets assigned to which variable

    i = 0
    while i < len(ir_tuples):
        op, arg1, arg2, result = ir_tuples[i]

        if op == 'LOAD_CONST':
            # Store the constant associated with the temp
            temp_values[result] = arg1

        elif op == 'STORE':
            # Check if storing a temp that holds a constant
            if arg1 in temp_values:
                optimized.append(('ASSIGN', temp_values[arg1], None, result))
                temp_values.pop(arg1)  # temp no longer needed
            else:
                optimized.append((op, arg1, arg2, result))

        elif op in {'+', '-', '*', '/', '>', '<', '==', '!=', '>=', '<='}:
            # Binary operations - pass through, keep temps
            optimized.append((op, arg1, arg2, result))

        elif op == 'DECL':
            # Keep declarations
            optimized.append((op, arg1, arg2, result))

        elif op in {'JZ', 'JMP', 'LABEL', 'RETURN'}:
            # Control flow ops
            optimized.append((op, arg1, arg2, result))

        else:
            # Pass through anything else
            optimized.append((op, arg1, arg2, result))

        i += 1

    return optimized



# ==============================
# CODE GENERATION (pseudo-ASM)
# ==============================

def generate_code(ir):
    asm = []
    reg_map = {}  # Mapping variables to registers
    next_reg = 1  # Start with R1

    def get_register(var):
        nonlocal next_reg
        if var not in reg_map:
            reg_map[var] = f"R{next_reg}"
            next_reg += 1
        return reg_map[var]

    for instr in ir:
        op, arg1, arg2, result = instr

        if op == 'LABEL':
            asm.append(f"{result}:")
        elif op == 'LOAD_CONST':
            reg = get_register(result)
            asm.append(f"MOV {reg}, {arg1}")
        elif op == 'STORE':
            src = get_register(arg1)
            dest = get_register(result)
            asm.append(f"MOV {dest}, {src}")
        elif op == 'ASSIGN':
            dest = get_register(result)
            asm.append(f"MOV {dest}, {arg1}")
        elif op == 'DECL':
            continue  # Declarations don’t need to generate assembly
        elif op in ('+', '-', '*', '/'):
            reg1 = get_register(arg1)
            reg2 = get_register(arg2)
            dest = get_register(result)

            if op == '+':
                asm_op = 'ADD'
            elif op == '-':
                asm_op = 'SUB'
            elif op == '*':
                asm_op = 'MUL'
            elif op == '/':
                asm_op = 'DIV'

            asm.append(f"{asm_op} {dest}, {reg1}, {reg2}")
        elif op == '>':
            reg1 = get_register(arg1)
            reg2 = get_register(arg2)
            dest = get_register(result)
            asm.append(f"CMPGT {dest}, {reg1}, {reg2}")
        elif op == 'JZ':
            cond = get_register(arg1)
            asm.append(f"JZ {cond}, {result}")
        elif op == 'JMP':
            asm.append(f"JMP {result}")
        elif op == 'RETURN':
            reg = get_register(arg1)
            asm.append(f"RET {reg}")
        else:
            asm.append(f"# Unrecognized operation: {op}")

    return "\n".join(asm)


# ==============================
# GUI
# ==============================
def clear_all_tabs():
    for tab in tabs.values():
        tab.delete("1.0", tk.END)

def run_compiler():
    try:
        clear_all_tabs()
        code = code_input.get("1.0", tk.END)
        tokens = tokenize(code)
        ast = parse(tokens)
        semantic_check(ast)
        ir = generate_ir(ast)
        optimized_ir = optimize_ir(ir)
        codegen = generate_code(optimized_ir)

        tac = ir_to_three_address_code(ir)
        optimized_tac = ir_to_three_address_code(optimized_ir)

        ir_text = "=== IR (Tuples) ===\n"
        ir_text += "\n".join([str(instr) for instr in ir])
        ir_text += "\n\n=== IR (Three Address Code) ===\n"
        ir_text += tac

        optimized_ir_text = "=== Optimized IR (Tuples) ===\n"
        optimized_ir_text += "\n".join([str(instr) for instr in optimized_ir])
        optimized_ir_text += "\n\n=== Optimized IR (Three Address Code) ===\n"
        optimized_ir_text += optimized_tac

        tabs["Lexical"].delete("1.0", tk.END)
        tabs["Lexical"].insert(tk.END, '\n'.join(f"{typ}: {val}" for typ, val in tokens))

        tabs["Syntax"].delete("1.0", tk.END)
        tabs["Syntax"].insert(tk.END, repr(ast))

        tabs["Semantic"].delete("1.0", tk.END)
        tabs["Semantic"].insert(tk.END, "No semantic errors.")

        tabs["IR"].delete("1.0", tk.END)
        tabs["IR"].insert(tk.END, ir_text)

        tabs["Optimized"].delete("1.0", tk.END)
        tabs["Optimized"].insert(tk.END, optimized_ir_text)

        tabs["CodeGen"].delete("1.0", tk.END)
        tabs["CodeGen"].insert(tk.END, codegen)

        tabs["Errors"].delete("1.0", tk.END)

    except Exception as e:
        tabs["Errors"].delete("1.0", tk.END)
        tabs["Errors"].insert(tk.END, str(e))

root = tk.Tk()
root.title("Compiler Simulation")

frame = ttk.Frame(root)
frame.pack(fill="both", expand=True)

code_input = tk.Text(frame, height=10)
code_input.pack(fill="x")

compile_button = ttk.Button(frame, text="Compile", command=run_compiler)
compile_button.pack()

notebook = ttk.Notebook(frame)
notebook.pack(fill="both", expand=True)

tabs = {}
for tab_name in ["Lexical", "Syntax", "Semantic", "IR", "Optimized", "CodeGen", "Errors"]:
    text = tk.Text(notebook)
    notebook.add(text, text=tab_name)
    tabs[tab_name] = text

root.mainloop()