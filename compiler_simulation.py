# Compiler Simulation for C-like Language
# This simulation demonstrates the main stages of compilation:
# 1. Lexical Analysis (Scanner/Tokenizer)
# 2. Syntax Analysis (Parser)
# 3. Semantic Analysis
# 4. Intermediate Code Generation
# 5. Code Optimization
# 6. Code Generation

import re
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple

# ----- LEXICAL ANALYSIS (SCANNER) -----

class TokenType(Enum):
    INTEGER = auto()
    FLOAT = auto()
    IDENTIFIER = auto()
    KEYWORD = auto()
    OPERATOR = auto()
    DELIMITER = auto()
    STRING = auto()
    EOF = auto()

class Token:
    def __init__(self, token_type: TokenType, value: str, line: int, column: int):
        self.type = token_type
        self.value = value
        self.line = line
        self.column = column
    
    def __str__(self):
        return f"Token({self.type}, '{self.value}', line={self.line}, col={self.column})"
    
    def __repr__(self):
        return self.__str__()

class Lexer:
    def __init__(self, source_code: str):
        self.source = source_code
        self.position = 0
        self.line = 1
        self.column = 1
        self.current_char = self.source[0] if self.source else None
        
        # Define language elements
        self.keywords = {'int', 'float', 'if', 'else', 'while', 'return', 'void', 'for'}
        self.operators = {'+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>=', '&&', '||'}
        self.delimiters = {'{', '}', '(', ')', '[', ']', ';', ','}
    
    def advance(self):
        self.position += 1
        self.column += 1
        
        if self.position >= len(self.source):
            self.current_char = None
        else:
            self.current_char = self.source[self.position]
            if self.current_char == '\n':
                self.line += 1
                self.column = 0
    
    def peek(self) -> Optional[str]:
        peek_pos = self.position + 1
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]
    
    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()
    
    def skip_comment(self):
        # Skip single line comment
        if self.current_char == '/' and self.peek() == '/':
            while self.current_char is not None and self.current_char != '\n':
                self.advance()
            self.advance()  # Skip the newline
        # Skip multi-line comment
        elif self.current_char == '/' and self.peek() == '*':
            self.advance()  # Skip /
            self.advance()  # Skip *
            while self.current_char is not None:
                if self.current_char == '*' and self.peek() == '/':
                    self.advance()  # Skip *
                    self.advance()  # Skip /
                    break
                self.advance()
    
    def identifier(self) -> Token:
        result = ''
        start_column = self.column
        
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        
        if result in self.keywords:
            return Token(TokenType.KEYWORD, result, self.line, start_column)
        else:
            return Token(TokenType.IDENTIFIER, result, self.line, start_column)
    
    def number(self) -> Token:
        result = ''
        start_column = self.column
        is_float = False
        
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                is_float = True
            result += self.current_char
            self.advance()
        
        if is_float:
            return Token(TokenType.FLOAT, result, self.line, start_column)
        else:
            return Token(TokenType.INTEGER, result, self.line, start_column)
    
    def string(self) -> Token:
        result = ''
        start_column = self.column
        self.advance()  # Skip opening quote
        
        while self.current_char is not None and self.current_char != '"':
            result += self.current_char
            self.advance()
        
        if self.current_char is None:
            raise Exception(f"Unterminated string at line {self.line}, column {start_column}")
        
        self.advance()  # Skip closing quote
        return Token(TokenType.STRING, result, self.line, start_column)
    
    def operator(self) -> Token:
       start_column = self.column
    
    # Handle multi-character operators first
       two_char = self.current_char + (self.peek() or '')
       if two_char in {'==', '!=', '<=', '>=', '&&', '||'}:
          result = two_char
          self.advance()
          self.advance()
          return Token(TokenType.OPERATOR, result, self.line, start_column)
    
    # Single character operator
       if self.current_char in '+-*/=<>!&|':
          result = self.current_char
          self.advance()
          return Token(TokenType.OPERATOR, result, self.line, start_column)
    
       raise Exception(f"Unexpected operator character '{self.current_char}'")
    
    def get_next_token(self) -> Token:
        while self.current_char is not None:
            # Skip whitespace
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            # Skip comments
            if self.current_char == '/' and (self.peek() == '/' or self.peek() == '*'):
                self.skip_comment()
                continue
            
            # Identifiers and keywords
            if self.current_char.isalpha() or self.current_char == '_':
                return self.identifier()
            
            # Numbers
            if self.current_char.isdigit():
                return self.number()
            
            # Strings
            if self.current_char == '"':
                return self.string()
            
            # Operators
            if self.current_char in '+-*/=<>!&|':
                return self.operator()
            
            # Delimiters
            if self.current_char in self.delimiters:
                token = Token(TokenType.DELIMITER, self.current_char, self.line, self.column)
                self.advance()
                return token
            
            # Unrecognized character
            raise Exception(f"Unexpected character '{self.current_char}' at line {self.line}, column {self.column}")
        
        # End of file
        return Token(TokenType.EOF, '', self.line, self.column)
    
    def tokenize(self) -> List[Token]:
        tokens = []
        while True:
            token = self.get_next_token()
            tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens

# ----- SYNTAX ANALYSIS (PARSER) -----

class ASTNode:
    def __init__(self, node_type: str):
        self.node_type = node_type
        self.children = []
    
    def add_child(self, child):
        self.children.append(child)
    
    def __str__(self):
        return self.node_type
    
    def __repr__(self):
        return self.__str__()

# ----- PARSER IMPROVEMENTS -----
# The main issue is in the parse_statement method and expression parsing capabilities
# Let's update these sections

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
        self.current_token = self.tokens[0] if self.tokens else None
    
    def advance(self):
        self.position += 1
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
        else:
            self.current_token = None
    
    def peek(self):
        peek_pos = self.position + 1
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return None
    
    def eat(self, token_type: TokenType):
        if self.current_token and self.current_token.type == token_type:
            token = self.current_token
            self.advance()
            return token
        else:
            expected = token_type
            got = self.current_token.type if self.current_token else "None"
            line = self.current_token.line if self.current_token else "unknown"
            col = self.current_token.column if self.current_token else "unknown"
            raise Exception(f"Expected {expected}, got {got}, line={line}, col={col} in statement")
    
    def parse(self):
        """Parse the tokens into an AST."""
        # Root of the AST
        root = ASTNode("Program")
        
        # Parse declarations and statements
        while self.current_token and self.current_token.type != TokenType.EOF:
            try:
                node = self.parse_statement()
                if node:  # Skip None nodes
                    root.add_child(node)
            except Exception as e:
                print(f"Parser error: {e}")
                # Skip to next semicolon or block end to recover from errors
                self._recover_from_error()
        
        return root
    
    def _recover_from_error(self):
        """Error recovery: skip until next semicolon or block end."""
        while (self.current_token and 
               not (self.current_token.type == TokenType.DELIMITER and 
                    self.current_token.value in {';', '}'})):
            self.advance()
        
        if self.current_token and self.current_token.value == ';':
            self.advance()  # Skip the semicolon
    
    def parse_statement(self):
        """Parse a statement."""
        if not self.current_token:
            raise Exception("Unexpected end of input")
        
        # Variable declaration
        if (self.current_token.type == TokenType.KEYWORD and 
            self.current_token.value in ('int', 'float', 'void')):
            return self.parse_variable_declaration()
        
        # Assignment
        elif self.current_token.type == TokenType.IDENTIFIER:
            if self.peek() and self.peek().type == TokenType.DELIMITER and self.peek().value == '(':
                return self.parse_function_call()
            else:
                return self.parse_assignment()
        
        # Control structures
        elif self.current_token.type == TokenType.KEYWORD:
            if self.current_token.value == 'if':
                return self.parse_if_statement()
            elif self.current_token.value == 'while':
                return self.parse_while_statement()
            elif self.current_token.value == 'for':
                return self.parse_for_statement()
            elif self.current_token.value == 'return':
                return self.parse_return_statement()
        
        # Block
        elif (self.current_token.type == TokenType.DELIMITER and 
              self.current_token.value == '{'):
            return self.parse_block()
        
        # Handle semicolons as empty statements
        elif (self.current_token.type == TokenType.DELIMITER and
              self.current_token.value == ';'):
            self.advance()
            return ASTNode("EmptyStatement")
        
        raise Exception(f"Unexpected token {self.current_token} in statement at line {self.current_token.line}")
    
    def parse_variable_declaration(self):
        """Parse variable declarations like: int x = 5;"""
        node = ASTNode("VariableDeclaration")
        
        # Get the type
        type_token = self.eat(TokenType.KEYWORD)
        node.add_child(ASTNode(f"Type:{type_token.value}"))
        
        # Get the variable name
        var_token = self.eat(TokenType.IDENTIFIER)
        node.add_child(ASTNode(f"Identifier:{var_token.value}"))
        
        # Check for array declaration
        if self.current_token.type == TokenType.DELIMITER and self.current_token.value == '[':
            self.eat(TokenType.DELIMITER)  # eat [
            size_expr = self.parse_expression()
            node.add_child(size_expr)
            self.eat(TokenType.DELIMITER)  # eat ]
        
        # Check for assignment
        if self.current_token.type == TokenType.OPERATOR and self.current_token.value == '=':
            self.eat(TokenType.OPERATOR)  # eat =
            expr = self.parse_expression()
            node.add_child(expr)
        
        # Expect semicolon
        self.eat(TokenType.DELIMITER)  # eat semicolon
        
        return node
    
    def parse_function_call(self):
        """Parse function calls: foo(arg1, arg2);"""
        node = ASTNode("FunctionCall")
        
        # Get function name
        func_token = self.eat(TokenType.IDENTIFIER)
        node.add_child(ASTNode(f"Identifier:{func_token.value}"))
        
        # Parse arguments
        self.eat(TokenType.DELIMITER)  # eat (
        
        # Check if there are arguments
        if self.current_token.type != TokenType.DELIMITER or self.current_token.value != ')':
            args_node = ASTNode("Arguments")
            
            # Parse first argument
            arg = self.parse_expression()
            args_node.add_child(arg)
            
            # Parse remaining arguments
            while self.current_token.type == TokenType.DELIMITER and self.current_token.value == ',':
                self.eat(TokenType.DELIMITER)  # eat ,
                arg = self.parse_expression()
                args_node.add_child(arg)
            
            node.add_child(args_node)
        
        self.eat(TokenType.DELIMITER)  # eat )
        
        # Check for semicolon if this is a statement
        if self.current_token.type == TokenType.DELIMITER and self.current_token.value == ';':
            self.eat(TokenType.DELIMITER)  # eat ;
        
        return node
    
    def parse_assignment(self):
        """Parse assignment statements: x = 10;"""
        node = ASTNode("Assignment")
        
        # Get the variable name
        var_token = self.eat(TokenType.IDENTIFIER)
        node.add_child(ASTNode(f"Identifier:{var_token.value}"))
        
        # Handle array access
        if self.current_token.type == TokenType.DELIMITER and self.current_token.value == '[':
            self.eat(TokenType.DELIMITER)  # eat [
            index_expr = self.parse_expression()
            node.add_child(index_expr)
            self.eat(TokenType.DELIMITER)  # eat ]
        
        # Expect =
        self.eat(TokenType.OPERATOR)  # eat =
        
        # Get the expression
        expr = self.parse_expression()
        node.add_child(expr)
        
        # Expect semicolon
        self.eat(TokenType.DELIMITER)  # eat semicolon
        
        return node
    
    def parse_expression(self):
        """Parse expressions."""
        # Parse expression based on precedence
        return self.parse_logical_or()
    
    def parse_logical_or(self):
        """Parse logical OR expressions (||)."""
        node = self.parse_logical_and()
        
        while (self.current_token and self.current_token.type == TokenType.OPERATOR 
               and self.current_token.value == '||'):
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            
            right = self.parse_logical_and()
            
            # Create a binary operation node
            op_node = ASTNode(f"BinaryOp:{op}")
            op_node.add_child(node)
            op_node.add_child(right)
            node = op_node
        
        return node
    
    def parse_logical_and(self):
        """Parse logical AND expressions (&&)."""
        node = self.parse_equality()
        
        while (self.current_token and self.current_token.type == TokenType.OPERATOR 
               and self.current_token.value == '&&'):
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            
            right = self.parse_equality()
            
            # Create a binary operation node
            op_node = ASTNode(f"BinaryOp:{op}")
            op_node.add_child(node)
            op_node.add_child(right)
            node = op_node
        
        return node
    
    def parse_equality(self):
        """Parse equality expressions (==, !=)."""
        node = self.parse_relational()
        
        while (self.current_token and self.current_token.type == TokenType.OPERATOR 
               and self.current_token.value in {'==', '!='}):
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            
            right = self.parse_relational()
            
            # Create a binary operation node
            op_node = ASTNode(f"BinaryOp:{op}")
            op_node.add_child(node)
            op_node.add_child(right)
            node = op_node
        
        return node
    
    def parse_relational(self):
        """Parse relational expressions (<, >, <=, >=)."""
        node = self.parse_addition()
        
        while (self.current_token and self.current_token.type == TokenType.OPERATOR 
               and self.current_token.value in {'<', '>', '<=', '>='}):
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            
            right = self.parse_addition()
            
            # Create a binary operation node
            op_node = ASTNode(f"BinaryOp:{op}")
            op_node.add_child(node)
            op_node.add_child(right)
            node = op_node
        
        return node
    
    def parse_addition(self):
        """Parse addition and subtraction."""
        node = self.parse_multiplication()
        
        while (self.current_token and self.current_token.type == TokenType.OPERATOR
               and self.current_token.value in {'+', '-'}):
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            
            right = self.parse_multiplication()
            
            # Create a binary operation node
            op_node = ASTNode(f"BinaryOp:{op}")
            op_node.add_child(node)
            op_node.add_child(right)
            node = op_node
        
        return node
    
    def parse_multiplication(self):
        """Parse multiplication and division."""
        node = self.parse_unary()
        
        while (self.current_token and self.current_token.type == TokenType.OPERATOR
               and self.current_token.value in {'*', '/'}):
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            
            right = self.parse_unary()
            
            # Create a binary operation node
            op_node = ASTNode(f"BinaryOp:{op}")
            op_node.add_child(node)
            op_node.add_child(right)
            node = op_node
        
        return node
    
    def parse_unary(self):
        """Parse unary expressions (+, -, !)."""
        if (self.current_token and self.current_token.type == TokenType.OPERATOR
            and self.current_token.value in {'+', '-', '!'}):
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            
            operand = self.parse_unary()  # Handle nested unary operations
            
            # Create a unary operation node
            op_node = ASTNode(f"UnaryOp:{op}")
            op_node.add_child(operand)
            return op_node
        
        return self.parse_primary()
    
    def parse_primary(self):
        """Parse primary expressions: literals, identifiers, parenthesized expressions."""
        if not self.current_token:
            raise Exception("Unexpected end of input in expression")
        
        if self.current_token.type == TokenType.INTEGER:
            value = self.current_token.value
            self.eat(TokenType.INTEGER)
            return ASTNode(f"IntegerLiteral:{value}")
        
        elif self.current_token.type == TokenType.FLOAT:
            value = self.current_token.value
            self.eat(TokenType.FLOAT)
            return ASTNode(f"FloatLiteral:{value}")
        
        elif self.current_token.type == TokenType.STRING:
            value = self.current_token.value
            self.eat(TokenType.STRING)
            return ASTNode(f"StringLiteral:{value}")
        
        elif self.current_token.type == TokenType.IDENTIFIER:
            value = self.current_token.value
            self.eat(TokenType.IDENTIFIER)
            
            # Handle function calls
            if self.current_token and self.current_token.type == TokenType.DELIMITER and self.current_token.value == '(':
                node = ASTNode("FunctionCall")
                node.add_child(ASTNode(f"Identifier:{value}"))
                
                self.eat(TokenType.DELIMITER)  # eat (
                
                # Parse arguments
                args_node = ASTNode("Arguments")
                
                # Check if there are any arguments
                if self.current_token.type != TokenType.DELIMITER or self.current_token.value != ')':
                    # Parse first argument
                    arg = self.parse_expression()
                    args_node.add_child(arg)
                    
                    # Parse remaining arguments
                    while (self.current_token and self.current_token.type == TokenType.DELIMITER 
                           and self.current_token.value == ','):
                        self.eat(TokenType.DELIMITER)  # eat ,
                        arg = self.parse_expression()
                        args_node.add_child(arg)
                
                node.add_child(args_node)
                self.eat(TokenType.DELIMITER)  # eat )
                return node
            
            # Handle array access
            elif self.current_token and self.current_token.type == TokenType.DELIMITER and self.current_token.value == '[':
                node = ASTNode("ArrayAccess")
                node.add_child(ASTNode(f"Identifier:{value}"))
                
                self.eat(TokenType.DELIMITER)  # eat [
                index_expr = self.parse_expression()
                node.add_child(index_expr)
                self.eat(TokenType.DELIMITER)  # eat ]
                return node
            
            return ASTNode(f"Identifier:{value}")
        
        elif self.current_token.type == TokenType.DELIMITER and self.current_token.value == '(':
            self.eat(TokenType.DELIMITER)  # eat (
            expr = self.parse_expression()
            self.eat(TokenType.DELIMITER)  # eat )
            return expr
        
        raise Exception(f"Unexpected token {self.current_token} in expression")
    
    def parse_if_statement(self):
        """Parse if statements: if (condition) { ... } else { ... }"""
        node = ASTNode("IfStatement")
        
        self.eat(TokenType.KEYWORD)  # eat if
        
        # Parse condition
        self.eat(TokenType.DELIMITER)  # eat (
        condition = self.parse_expression()
        node.add_child(condition)
        self.eat(TokenType.DELIMITER)  # eat )
        
        # Parse if block
        if_block = self.parse_statement()
        node.add_child(if_block)
        
        # Check for else
        if (self.current_token and self.current_token.type == TokenType.KEYWORD 
            and self.current_token.value == 'else'):
            self.eat(TokenType.KEYWORD)  # eat else
            else_block = self.parse_statement()
            node.add_child(else_block)
        
        return node
    
    def parse_while_statement(self):
        """Parse while statements: while (condition) { ... }"""
        node = ASTNode("WhileStatement")
        
        self.eat(TokenType.KEYWORD)  # eat while
        
        # Parse condition
        self.eat(TokenType.DELIMITER)  # eat (
        condition = self.parse_expression()
        node.add_child(condition)
        self.eat(TokenType.DELIMITER)  # eat )
        
        # Parse body
        body = self.parse_statement()
        node.add_child(body)
        
        return node
    
    def parse_for_statement(self):
        """Parse for statements: for (init; cond; update) { ... }"""
        node = ASTNode("ForStatement")
        
        self.eat(TokenType.KEYWORD)  # eat for
        self.eat(TokenType.DELIMITER)  # eat (
        
        # Parse initialization (can be empty)
        if not (self.current_token.type == TokenType.DELIMITER and self.current_token.value == ';'):
            init = self.parse_statement()  # This will consume the semicolon
            node.add_child(init)
        else:
            self.eat(TokenType.DELIMITER)  # eat ;
            node.add_child(ASTNode("EmptyStatement"))
        
        # Parse condition (can be empty)
        if not (self.current_token.type == TokenType.DELIMITER and self.current_token.value == ';'):
            condition = self.parse_expression()
            node.add_child(condition)
        else:
            node.add_child(ASTNode("EmptyExpression"))
        self.eat(TokenType.DELIMITER)  # eat ;
        
        # Parse update (can be empty)
        if not (self.current_token.type == TokenType.DELIMITER and self.current_token.value == ')'):
            update = self.parse_expression()
            node.add_child(update)
        else:
            node.add_child(ASTNode("EmptyExpression"))
        self.eat(TokenType.DELIMITER)  # eat )
        
        # Parse body
        body = self.parse_statement()
        node.add_child(body)
        
        return node
    
    def parse_return_statement(self):
        """Parse return statements: return expression;"""
        node = ASTNode("ReturnStatement")
        
        self.eat(TokenType.KEYWORD)  # eat return
        
        # Parse expression if exists
        if not (self.current_token.type == TokenType.DELIMITER and self.current_token.value == ';'):
            expr = self.parse_expression()
            node.add_child(expr)
        
        self.eat(TokenType.DELIMITER)  # eat semicolon
        
        return node
    
    def parse_block(self):
        """Parse a block of statements: { ... }"""
        node = ASTNode("Block")
        
        self.eat(TokenType.DELIMITER)  # eat {
        
        while (self.current_token and 
               not (self.current_token.type == TokenType.DELIMITER and self.current_token.value == '}')):
            statement = self.parse_statement()
            if statement:  # Skip None statements
                node.add_child(statement)
            
            if self.current_token is None:
                raise Exception("Unexpected end of file while parsing block")
        
        self.eat(TokenType.DELIMITER)  # eat }
        
        return node
# ----- SEMANTIC ANALYSIS -----

class Symbol:
    def __init__(self, name: str, symbol_type: str):
        self.name = name
        self.type = symbol_type

class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None
    
    def define(self, name: str, symbol_type: str) -> Symbol:
        symbol = Symbol(name, symbol_type)
        self.symbols[name] = symbol
        return symbol
    
    def lookup(self, name: str) -> Optional[Symbol]:
        symbol = self.symbols.get(name)
        if symbol is not None:
            return symbol
        
        if self.parent is not None:
            return self.parent.lookup(name)
        
        return None

class SemanticAnalyzer:
    def __init__(self, ast):
        self.ast = ast
        self.current_scope = SymbolTable()
        self.errors = []
    
    def analyze(self):
        self._analyze_node(self.ast)
        return self.errors
    
    def _analyze_node(self, node):
        method_name = f"_analyze_{node.node_type.split(':')[0].lower()}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method(node)
        
        # Default behavior: analyze all children
        for child in node.children:
            self._analyze_node(child)
    
    def _analyze_variabledeclaration(self, node):
        # Get type and identifier
        type_node = node.children[0]
        var_type = type_node.node_type.split(':')[1]
        
        id_node = node.children[1]
        var_name = id_node.node_type.split(':')[1]
        
        # Check if variable already defined in current scope
        if self.current_scope.lookup(var_name) is not None:
            self.errors.append(f"Variable '{var_name}' already defined")
        else:
            self.current_scope.define(var_name, var_type)
        
        # Analyze initialization expression if it exists
        if len(node.children) > 2:
            self._analyze_node(node.children[2])
    
    def _analyze_assignment(self, node):
        # Get identifier
        id_node = node.children[0]
        var_name = id_node.node_type.split(':')[1]
        
        # Check if variable is defined
        if self.current_scope.lookup(var_name) is None:
            self.errors.append(f"Variable '{var_name}' used before definition")
        
        # Analyze expression
        self._analyze_node(node.children[1])
    
    def _analyze_identifier(self, node):
        var_name = node.node_type.split(':')[1]
        
        # Check if variable is defined
        if self.current_scope.lookup(var_name) is None:
            self.errors.append(f"Variable '{var_name}' used before definition")
    
    def _analyze_block(self, node):
        # Create new scope
        old_scope = self.current_scope
        self.current_scope = SymbolTable()
        self.current_scope.parent = old_scope
        
        # Analyze children
        for child in node.children:
            self._analyze_node(child)
        
        # Restore old scope
        self.current_scope = old_scope
    
    def _analyze_ifstatement(self, node):
        # Analyze condition
        self._analyze_node(node.children[0])
        
        # Analyze blocks
        for i in range(1, len(node.children)):
            self._analyze_node(node.children[i])
    
    def _analyze_whilestatement(self, node):
        # Analyze condition
        self._analyze_node(node.children[0])
        
        # Analyze body
        self._analyze_node(node.children[1])
    
    def _analyze_returnstatement(self, node):
        # Analyze expression if exists
        if node.children:
            self._analyze_node(node.children[0])

# ----- INTERMEDIATE CODE GENERATION -----

class ThreeAddressCode:
    def __init__(self):
        self.instructions = []
        self.temp_counter = 0
        self.label_counter = 0
    
    def new_temp(self):
        temp = f"t{self.temp_counter}"
        self.temp_counter += 1
        return temp
    
    def new_label(self):
        label = f"L{self.label_counter}"
        self.label_counter += 1
        return label
    
    def append(self, instruction):
        self.instructions.append(instruction)
    
    def get_code(self):
        return self.instructions

class IRGenerator:
    def __init__(self, ast):
        self.ast = ast
        self.code = ThreeAddressCode()
    
    def generate(self):
        self._generate_code(self.ast)
        return self.code.get_code()
    
    def _generate_code(self, node):
        method_name = f"_generate_{node.node_type.split(':')[0].lower()}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method(node)
        
        # Default behavior: generate code for all children
        results = []
        for child in node.children:
            result = self._generate_code(child)
            if result is not None:
                results.append(result)
        
        return results[0] if results else None
    
    def _generate_program(self, node):
        # Generate code for all children
        for child in node.children:
            self._generate_code(child)
        return None
    
    def _generate_variabledeclaration(self, node):
        # Get variable name
        var_name = node.children[1].node_type.split(':')[1]
        
        # Check if there's an initialization
        if len(node.children) > 2:
            expr_result = self._generate_code(node.children[2])
            self.code.append(f"{var_name} = {expr_result}")
        
        return var_name
    
    def _generate_assignment(self, node):
        # Get variable name
        var_name = node.children[0].node_type.split(':')[1]
        
        # Generate code for expression
        expr_result = self._generate_code(node.children[1])
        
        # Create assignment instruction
        self.code.append(f"{var_name} = {expr_result}")
        
        return var_name
    
    def _generate_binaryop(self, node):
        # Generate code for left and right operands
        left = self._generate_code(node.children[0])
        right = self._generate_code(node.children[1])
        
        # Get operator
        op = node.node_type.split(':')[1]
        
        # Create a new temporary variable
        temp = self.code.new_temp()
        
        # Create instruction
        self.code.append(f"{temp} = {left} {op} {right}")
        
        return temp
    
    def _generate_integerliteral(self, node):
        return node.node_type.split(':')[1]
    
    def _generate_floatliteral(self, node):
        return node.node_type.split(':')[1]
    
    def _generate_stringliteral(self, node):
        return f'"{node.node_type.split(":", 1)[1]}"'
    
    def _generate_identifier(self, node):
        return node.node_type.split(':')[1]
    
    def _generate_ifstatement(self, node):
        # Generate code for condition
        condition = self._generate_code(node.children[0])
        
        # Create labels
        else_label = self.code.new_label()
        end_label = self.code.new_label()
        
        # Create conditional jump
        self.code.append(f"if not {condition} goto {else_label}")
        
        # Generate code for if block
        self._generate_code(node.children[1])
        
        # Jump to end after if block
        self.code.append(f"goto {end_label}")
        
        # Else label
        self.code.append(f"{else_label}:")
        
        # Generate code for else block if exists
        if len(node.children) > 2:
            self._generate_code(node.children[2])
        
        # End label
        self.code.append(f"{end_label}:")
        
        return None
    
    def _generate_whilestatement(self, node):
        # Create labels
        start_label = self.code.new_label()
        end_label = self.code.new_label()
        
        # Start label
        self.code.append(f"{start_label}:")
        
        # Generate code for condition
        condition = self._generate_code(node.children[0])
        
        # Create conditional jump
        self.code.append(f"if not {condition} goto {end_label}")
        
        # Generate code for body
        self._generate_code(node.children[1])
        
        # Jump back to start
        self.code.append(f"goto {start_label}")
        
        # End label
        self.code.append(f"{end_label}:")
        
        return None
    
    def _generate_returnstatement(self, node):
        # Generate code for expression if exists
        if node.children:
            expr_result = self._generate_code(node.children[0])
            self.code.append(f"return {expr_result}")
        else:
            self.code.append("return")
        
        return None
    
    def _generate_block(self, node):
        # Generate code for all statements in block
        for child in node.children:
            self._generate_code(child)
        
        return None

# ----- CODE OPTIMIZATION -----

class Optimizer:
    def __init__(self, ir_code):
        self.ir_code = ir_code
        self.optimized_code = []
    
    def optimize(self):
        """Apply various optimization techniques to the IR code."""
        # Make a copy of the original code
        self.optimized_code = self.ir_code.copy()
        
        # Apply optimizations
        self._constant_folding()
        self._dead_code_elimination()
        
        return self.optimized_code
    
    def _constant_folding(self):
        """Fold constant expressions."""
        for i in range(len(self.optimized_code)):
            line = self.optimized_code[i]
            
            # Look for operations with constants
            match = re.match(r'^(t\d+) = (\d+) ([+\-*/]) (\d+)$', line)
            if match:
                dest, left, op, right = match.groups()
                left, right = int(left), int(right)
                
                # Evaluate the constant expression
                if op == '+':
                    result = left + right
                elif op == '-':
                    result = left - right
                elif op == '*':
                    result = left * right
                elif op == '/' and right != 0:
                    result = left / right
                else:
                    continue
                
                # Replace the instruction with constant assignment
                self.optimized_code[i] = f"{dest} = {result}"
    
    def _dead_code_elimination(self):
        """Eliminate unused variables and unreachable code."""
        # This is a simple implementation that only removes temporary variables
        # that are never used after their definition
        
        # Find used variables
        used_vars = set()
        for line in self.optimized_code:
            # Extract variables used on the right side of assignments
            match = re.match(r'^[^=]+ = (.+)$', line)
            if match:
                expr = match.group(1)
                for var in re.findall(r't\d+', expr):
                    used_vars.add(var)
        
        # Remove assignments to unused temporary variables
        new_code = []
        for line in self.optimized_code:
            # Check if this is an assignment to an unused temporary variable
            match = re.match(r'^(t\d+) = .+$', line)
            if match and match.group(1) not in used_vars:
                continue  # Skip this line
            new_code.append(line)
        
        self.optimized_code = new_code

# ----- TARGET CODE GENERATION -----

class TargetCodeGenerator:
    def __init__(self, ir_code):
        self.ir_code = ir_code
        self.assembly_code = []
        self.registers = ['eax', 'ebx', 'ecx', 'edx']
        self.register_map = {}  # Maps variables to registers
        self.memory_map = {}   # Maps variables to memory locations
        self.memory_offset = 0
    
    def generate(self):
        """Generate x86-like assembly code from the optimized IR code."""
        # Add data section
        self.assembly_code.append(".data")
        
        # Add code section
        self.assembly_code.append(".code")
        self.assembly_code.append("main:")
        
        # Process each IR instruction
        for instr in self.ir_code:
            # Add comment with original IR instruction
            self.assembly_code.append(f"  ; {instr}")
            
            # Generate assembly code for this instruction
            assembly_instrs = self._translate_instruction(instr)
            for asm in assembly_instrs:
                self.assembly_code.append(f"  {asm}")
        
        # Add return
        self.assembly_code.append("  ret")
        
        return self.assembly_code
    
    def _get_location(self, var):
        """Get register or memory location for a variable."""
        # If it's a constant, return the constant
        if re.match(r'^-?\d+(\.\d+)?$', var):
            return var
        
        # If it's a register, return register name
        if var in self.register_map:
            return self.register_map[var]
        
        # If it's in memory, return memory reference
        if var in self.memory_map:
            return f"[ebp-{self.memory_map[var]}]"
        
        # Allocate new memory location
        self.memory_offset += 4
        self.memory_map[var] = self.memory_offset
        return f"[ebp-{self.memory_map[var]}]"
    
    def _translate_instruction(self, instr):
        """Translate a single IR instruction to x86-like assembly."""
        result = []
        
        # Handle assignments: x = y
        assignment_match = re.match(r'^(\w+) = (\w+|".*")$', instr)
        if assignment_match:
            dest, src = assignment_match.groups()
            
            # Handle string literals
            if src.startswith('"'):
                # In real assembly we'd allocate the string in data section
                # For simplicity, we'll just use a comment
                result.append(f"mov {self._get_location(dest)}, {src} ; string literal")
            else:
                dest_loc = self._get_location(dest)
                src_loc = self._get_location(src)
                result.append(f"mov {dest_loc}, {src_loc}")
            
            return result
        
        # Handle binary operations: x = y op z
        binary_op_match = re.match(r'^(\w+) = (\w+|\d+) ([+\-*/]) (\w+|\d+)$', instr)
        if binary_op_match:
            dest, left, op, right = binary_op_match.groups()
            
            dest_loc = self._get_location(dest)
            left_loc = self._get_location(left)
            right_loc = self._get_location(right)
            
            # Move left operand to destination
            result.append(f"mov {dest_loc}, {left_loc}")
            
            # Perform operation
            if op == '+':
                result.append(f"add {dest_loc}, {right_loc}")
            elif op == '-':
                result.append(f"sub {dest_loc}, {right_loc}")
            elif op == '*':
                result.append(f"imul {dest_loc}, {right_loc}")
            elif op == '/':
                # Division is more complex in x86, simplifying here
                result.append(f"mov eax, {left_loc}")
                result.append(f"cdq")  # Sign-extend EAX into EDX:EAX
                result.append(f"idiv {right_loc}")
                result.append(f"mov {dest_loc}, eax")
            
            return result
        
        # Handle labels: label:
        label_match = re.match(r'^(\w+):$', instr)
        if label_match:
            label = label_match.group(1)
            result.append(f"{label}:")
            return result
        
        # Handle conditional jumps: if not condition goto label
        cond_jump_match = re.match(r'^if not (\w+) goto (\w+)$', instr)
        if cond_jump_match:
            condition, label = cond_jump_match.groups()
            cond_loc = self._get_location(condition)
            
            result.append(f"cmp {cond_loc}, 0")
            result.append(f"je {label}")
            return result
        
        # Handle unconditional jumps: goto label
        goto_match = re.match(r'^goto (\w+)$', instr)
        if goto_match:
            label = goto_match.group(1)
            result.append(f"jmp {label}")
            return result
        
        # Handle return statements
        return_match = re.match(r'^return( (\w+|\d+))?$', instr)
        if return_match:
            if return_match.group(1):  # Return with value
                value = return_match.group(2)
                value_loc = self._get_location(value)
                result.append(f"mov eax, {value_loc}")
            result.append("mov esp, ebp")
            result.append("pop ebp")
            result.append("ret")
            return result
        
        # If we can't recognize the instruction, add a comment
        result.append(f"; Unhandled IR instruction: {instr}")
        return result


# ----- COMPILER DRIVER -----

def print_ast(node, indent=0):
    """Pretty print the AST."""
    print(' ' * indent + str(node))
    for child in node.children:
        print_ast(child, indent + 2)

def run_compiler_simulation(source_code):
    """Run the compiler simulation on the given source code."""
    results = {
        "source_code": source_code,
        "stages": {}
    }
    
    print("==== SOURCE CODE ====")
    print(source_code)
    results["source_code"] = source_code
    
    # Lexical Analysis
    print("\n==== LEXICAL ANALYSIS ====")
    lexer = Lexer(source_code)
    tokens = lexer.tokenize()
    for token in tokens:
        print(token)
    results["stages"]["lexical_analysis"] = tokens
    
    # Syntax Analysis
    print("\n==== SYNTAX ANALYSIS ====")
    parser = Parser(tokens)
    ast = parser.parse()
    print("Abstract Syntax Tree:")
    print_ast(ast)
    results["stages"]["syntax_analysis"] = ast
    
    # Semantic Analysis
    print("\n==== SEMANTIC ANALYSIS ====")
    semantic_analyzer = SemanticAnalyzer(ast)
    errors = semantic_analyzer.analyze()
    
    if errors:
        print("Semantic errors found:")
        for error in errors:
            print(f"- {error}")
        results["stages"]["semantic_analysis"] = {"errors": errors}
    else:
        print("No semantic errors found.")
        results["stages"]["semantic_analysis"] = {"errors": []}
    
    # Intermediate Code Generation
    print("\n==== INTERMEDIATE CODE GENERATION ====")
    ir_generator = IRGenerator(ast)
    ir_code = ir_generator.generate()
    print("Three-Address Code:")
    for instr in ir_code:
        print(instr)
    results["stages"]["ir_generation"] = ir_code
    
    # Code Optimization
    print("\n==== CODE OPTIMIZATION ====")
    optimizer = Optimizer(ir_code)
    optimized_code = optimizer.optimize()
    print("Optimized Three-Address Code:")
    for instr in optimized_code:
        print(instr)
    results["stages"]["optimization"] = optimized_code
    
    # Target Code Generation
    print("\n==== TARGET CODE GENERATION ====")
    code_generator = TargetCodeGenerator(optimized_code)
    assembly_code = code_generator.generate()
    print("Assembly Code:")
    for line in assembly_code:
        print(line)
    results["stages"]["code_generation"] = assembly_code
    
    return results

# ----- EXAMPLE USAGE -----

def main():
    # Example C-like program
    example_code = """
    int main() {
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
    }
    """
    
    # Run the simulation
    results = run_compiler_simulation(example_code)
    
    return results

if __name__ == "__main__":
    main()