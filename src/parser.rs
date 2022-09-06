use std::borrow::Borrow;
use std::collections::HashMap;
use std::env;
use std::io::{self, Write};
use std::iter::Peekable;
use std::ops::DerefMut;
use std::str::Chars;

use crate::parser::Token::*;

const ANONYMOUS_FUNCTION_NAME: &str = "anonymous";

// ======================================================================================
// LEXER ================================================================================
// ======================================================================================

/// Represents a primitive syntax token.
#[derive(Debug, Clone)]
pub enum Token {
    Binary,
    Comma,
    Comment,
    Def,
    EOF,
    Ident(String),
    ParenLeft,
    ParenRight,
    Number(f64),
    Op(char),
    Unary,
    Var,
    TensorBegin,
    TensorEnd,
    Semicolon,
    BlockLeft,
    BlockRight,
    AngleLeft,
    AngleRight,
    Return,
}

pub struct VarType {
    shape: Vec<usize>,
}

impl VarType {
    pub fn new() -> Self {
        Self { shape: Vec::new() }
    }
}

impl Default for VarType {
    fn default() -> Self {
        Self::new()
    }
}

/// Defines an error encountered by the `Lexer`.
pub struct LexError {
    pub error: &'static str,
    pub index: usize,
}

impl LexError {
    pub fn new(msg: &'static str) -> LexError {
        LexError {
            error: msg,
            index: 0,
        }
    }

    pub fn with_index(msg: &'static str, index: usize) -> LexError {
        LexError { error: msg, index }
    }
}

#[derive(Clone, Debug)]
pub struct Location {
    filename: String,
    row: usize,
    col: usize,
}

impl Location {
    pub fn new(filename: &str, row: usize, col: usize) -> Location {
        Location {
            filename: filename.to_string(),
            row,
            col,
        }
    }
}

/// Defines the result of a lexing operation; namely a
/// `Token` on success, or a `LexError` on failure.
pub type LexResult = Result<Token, LexError>;

/// Defines a lexer which transforms an input `String` into
/// a `Token` stream.
pub struct Lexer<'a> {
    input: &'a str,
    chars: Box<Peekable<Chars<'a>>>,
    pos: usize,
    last_location: Location,
}

impl<'a> Lexer<'a> {
    /// Creates a new `Lexer`, given its source `input`.
    pub fn new(input: &'a str) -> Lexer<'a> {
        Lexer {
            input,
            chars: Box::new(input.chars().peekable()),
            pos: 0,
            last_location: Location {
                filename: "filename.toy".to_string(),
                row: 0,
                col: 0,
            },
        }
    }

    /// Lexes and returns the next `Token` from the source code.
    pub fn lex(&mut self) -> LexResult {
        let chars = self.chars.deref_mut();
        let src = self.input;

        let mut pos = self.pos;

        // Skip whitespaces
        loop {
            // Note: the following lines are in their own scope to
            // limit how long 'chars' is borrowed, and in order to allow
            // it to be borrowed again in the loop by 'chars.next()'.
            {
                let ch = chars.peek();

                if ch.is_none() {
                    self.pos = pos;

                    return Ok(Token::EOF);
                }

                if !ch.unwrap().is_whitespace() {
                    break;
                }
            }

            chars.next();
            pos += 1;
        }

        let start = pos;
        let next = chars.next();

        if next.is_none() {
            return Ok(Token::EOF);
        }

        pos += 1;

        // Actually get the next token.
        let result = match next.unwrap() {
            '(' => Ok(Token::ParenLeft),
            ')' => Ok(Token::ParenRight),
            ',' => Ok(Token::Comma),

            '#' => {
                // Comment
                loop {
                    let ch = chars.next();
                    pos += 1;

                    if ch == Some('\n') {
                        break;
                    }
                }

                Ok(Token::Comment)
            }

            '.' | '0'..='9' => {
                // Parse number literal
                loop {
                    let ch = match chars.peek() {
                        Some(ch) => *ch,
                        None => return Ok(Token::EOF),
                    };

                    // Parse float.
                    if ch != '.' && !ch.is_digit(16) {
                        break;
                    }

                    chars.next();
                    pos += 1;
                }

                Ok(Token::Number(src[start..pos].parse().unwrap()))
            }

            'a'..='z' | 'A'..='Z' | '_' => {
                // Parse identifier
                loop {
                    let ch = match chars.peek() {
                        Some(ch) => *ch,
                        None => return Ok(Token::EOF),
                    };

                    // A word-like identifier only contains underscores and alphanumeric characters.
                    if ch != '_' && !ch.is_alphanumeric() {
                        break;
                    }

                    chars.next();
                    pos += 1;
                }

                match &src[start..pos] {
                    "def" => Ok(Token::Def),
                    "unary" => Ok(Token::Unary),
                    "binary" => Ok(Token::Binary),
                    "var" => Ok(Token::Var),
                    "return" => Ok(Token::Return),

                    ident => Ok(Token::Ident(ident.to_string())),
                }
            }

            '[' => Ok(Token::TensorBegin),
            ']' => Ok(Token::TensorEnd),
            ';' => Ok(Token::Semicolon),
            '{' => Ok(Token::BlockLeft),
            '}' => Ok(Token::BlockRight),
            '<' => Ok(Token::AngleLeft),
            '>' => Ok(Token::AngleRight),

            op => {
                // Parse operator
                Ok(Token::Op(op))
            }
        };

        // Update stored position, and return
        self.pos = pos;

        result
    }

    pub fn last_location(&self) -> Location {
        self.last_location.clone()
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token;

    /// Lexes the next `Token` and returns it.
    /// On EOF or failure, `None` will be returned.
    fn next(&mut self) -> Option<Self::Item> {
        match self.lex() {
            Ok(EOF) | Err(_) => None,
            Ok(token) => Some(token),
        }
    }
}

// ======================================================================================
// PARSER ===============================================================================
// ======================================================================================

/// Defines a primitive expression.
#[derive(Debug, Clone)]
pub enum Expr {
    Binary {
        op: char,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    Call {
        fn_name: String,
        args: Vec<Expr>,
    },

    Number(f64),

    Variable(String),

    VarDecl {
        name: String,
        value: Box<Expr>,
    },

    Tensor {
        location: Location,
        values: Vec<Expr>,
        dims: Vec<usize>,
    },
    ExprList {
        expressions: Vec<Box<Expr>>,
    },
    Return {
        location: Location,
        expression: Box<Expr>,
    },
}

/// Defines the prototype (name and parameters) of a function.
#[derive(Debug, Clone)]
pub struct Prototype {
    pub name: String,
    pub args: Vec<String>,
    pub is_op: bool,
    pub prec: usize,
    pub location: Location,
}

/// Defines a user-defined or external function.
#[derive(Debug, Clone)]
pub struct Function {
    pub prototype: Prototype,
    pub body: Option<Expr>,
}

#[derive(Debug)]
pub struct Module {
    pub functions: Vec<Function>,
}

/// Represents the `Expr` parser.
pub struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    prec: &'a mut HashMap<char, i32>,
}

// I'm ignoring the 'must_use' lint in order to call 'self.advance' without checking
// the result when an EOF is acceptable.
#[allow(unused_must_use)]
impl<'a> Parser<'a> {
    /// Creates a new parser, given an input `str` and a `HashMap` binding
    /// an operator and its precedence in binary expressions.
    pub fn new(input: String, op_precedence: &'a mut HashMap<char, i32>) -> Self {
        let mut lexer = Lexer::new(input.as_str());
        let tokens = lexer.by_ref().collect();

        Parser {
            tokens,
            prec: op_precedence,
            pos: 0,
        }
    }

    pub fn parse_module(&mut self) -> Result<Module, &'static str> {
        let mut functions = Vec::new();
        loop {
            match self.current()? {
                Def => {
                    functions.push(self.parse_def()?);
                }
                Comment => {
                    self.advance();
                    continue;
                }
                EOF => {
                    println!("end of file reached!");
                    break;
                }
                _ => {
                    // TODO: error situation
                    break;
                }
            }
            if self.at_end() {
                break;
            }
        }

        Ok(Module { functions })
    }

    /// Returns the current `Token`, without performing safety checks beforehand.
    fn curr(&self) -> Token {
        self.tokens[self.pos].clone()
    }

    /// Returns the current `Token`, or an error that
    /// indicates that the end of the file has been unexpectedly reached if it is the case.
    fn current(&self) -> Result<Token, &'static str> {
        if self.pos >= self.tokens.len() {
            Err("Unexpected end of file.")
        } else {
            Ok(self.tokens[self.pos].clone())
        }
    }

    fn last_location(&self) -> Location {
        Location::new("filename", 0, 0)
    }

    /// Advances the position, and returns an empty `Result` whose error
    /// indicates that the end of the file has been unexpectedly reached.
    /// This allows to use the `self.advance()?;` syntax.
    fn advance(&mut self) -> Result<(), &'static str> {
        let npos = self.pos + 1;

        self.pos = npos;

        if npos < self.tokens.len() {
            Ok(())
        } else {
            Err("Unexpected end of file.")
        }
    }

    /// Returns a value indicating whether or not the `Parser`
    /// has reached the end of the input.
    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    /// Returns the precedence of the current `Token`, or 0 if it is not recognized as a binary operator.
    fn get_tok_precedence(&self) -> i32 {
        if let Ok(Op(op)) = self.current() {
            *self.prec.get(&op).unwrap_or(&100)
        } else {
            -1
        }
    }

    /// Parses the prototype of a function, whether external or user-defined.
    fn parse_prototype(&mut self) -> Result<Prototype, &'static str> {
        let location = self.last_location();
        let (id, is_operator, precedence) = match self.curr() {
            Ident(id) => {
                self.advance()?;

                (id, false, 0)
            }

            Binary => {
                self.advance()?;

                let op = match self.curr() {
                    Op(ch) => ch,
                    _ => return Err("Expected operator in custom operator declaration."),
                };

                self.advance()?;

                let mut name = String::from("binary");

                name.push(op);

                let prec = if let Number(prec) = self.curr() {
                    self.advance()?;

                    prec as usize
                } else {
                    0
                };

                self.prec.insert(op, prec as i32);

                (name, true, prec)
            }

            Unary => {
                self.advance()?;

                let op = match self.curr() {
                    Op(ch) => ch,
                    _ => return Err("Expected operator in custom operator declaration."),
                };

                let mut name = String::from("unary");

                name.push(op);

                self.advance()?;

                (name, true, 0)
            }

            _ => return Err("Expected identifier in prototype declaration."),
        };

        match self.curr() {
            ParenLeft => (),
            _ => return Err("Expected '(' character in prototype declaration."),
        }

        self.advance()?;

        if let ParenRight = self.curr() {
            self.advance();

            return Ok(Prototype {
                name: id,
                args: vec![],
                is_op: is_operator,
                prec: precedence,
                location,
            });
        }

        let mut args = vec![];

        loop {
            match self.curr() {
                Ident(name) => args.push(name),
                _ => return Err("Expected identifier in parameter declaration."),
            }

            self.advance()?;

            match self.curr() {
                ParenRight => {
                    self.advance();
                    break;
                }
                Comma => {
                    self.advance();
                }
                _ => return Err("Expected ',' or ')' character in prototype declaration."),
            }
        }

        Ok(Prototype {
            name: id,
            args,
            is_op: is_operator,
            prec: precedence,
            location,
        })
    }

    /// Parses a user-defined function.
    fn parse_def(&mut self) -> Result<Function, &'static str> {
        // Eat 'def' keyword
        self.pos += 1;

        // Parse signature of function
        let proto = self.parse_prototype()?;

        // Parse body of function
        // let body = self.parse_expr()?;
        let body = self.parse_block()?;

        // Return new function
        Ok(Function {
            prototype: proto,
            body: Some(body),
        })
    }

    /// Parses any expression.
    fn parse_expr(&mut self) -> Result<Expr, &'static str> {
        match self.parse_unary_expr() {
            Ok(left) => self.parse_binary_expr(0, left),
            err => err,
        }
    }

    /// Parses a literal number.
    fn parse_nb_expr(&mut self) -> Result<Expr, &'static str> {
        // Simply convert Token::Number to Expr::Number
        match self.curr() {
            Number(nb) => {
                self.advance();
                println!("Found number {}", nb);
                Ok(Expr::Number(nb))
            }
            _ => Err("Expected number literal."),
        }
    }

    /// Parses an expression enclosed in parenthesis.
    fn parse_paren_expr(&mut self) -> Result<Expr, &'static str> {
        match self.current()? {
            ParenLeft => (),
            _ => return Err("Expected '(' character at start of parenthesized expression."),
        }

        self.advance()?;

        let expr = self.parse_expr()?;

        match self.current()? {
            ParenRight => (),
            _ => return Err("Expected ')' character at end of parenthesized expression."),
        }

        self.advance();

        Ok(expr)
    }

    /// Parses an expression that starts with an identifier (either a variable or a function call).
    fn parse_id_expr(&mut self) -> Result<Expr, &'static str> {
        let id = match self.curr() {
            Ident(id) => id,
            _ => return Err("Expected identifier."),
        };

        if self.advance().is_err() {
            return Ok(Expr::Variable(id));
        }

        match self.curr() {
            ParenLeft => {
                self.advance()?;

                if let ParenRight = self.curr() {
                    return Ok(Expr::Call {
                        fn_name: id,
                        args: vec![],
                    });
                }

                let mut args = vec![];

                loop {
                    args.push(self.parse_expr()?);

                    match self.current()? {
                        Comma => (),
                        ParenRight => break,
                        _ => return Err("Expected ',' character in function call."),
                    }

                    self.advance()?;
                }

                self.advance();

                Ok(Expr::Call { fn_name: id, args })
            }

            _ => Ok(Expr::Variable(id)),
        }
    }

    /// Parses an unary expression.
    fn parse_unary_expr(&mut self) -> Result<Expr, &'static str> {
        let op = match self.current()? {
            Op(ch) => {
                self.advance()?;
                ch
            }
            _ => return self.parse_primary(),
        };

        let mut name = String::from("unary");

        name.push(op);

        Ok(Expr::Call {
            fn_name: name,
            args: vec![self.parse_unary_expr()?],
        })
    }

    /// Parses a binary expression, given its left-hand expression.
    fn parse_binary_expr(&mut self, prec: i32, mut left: Expr) -> Result<Expr, &'static str> {
        loop {
            let curr_prec = self.get_tok_precedence();

            if curr_prec < prec || self.at_end() {
                return Ok(left);
            }

            let op = match self.curr() {
                Op(op) => op,
                _ => return Err("Invalid operator."),
            };

            self.advance()?;

            let mut right = self.parse_unary_expr()?;

            let next_prec = self.get_tok_precedence();

            if curr_prec < next_prec {
                right = self.parse_binary_expr(curr_prec + 1, right)?;
            }

            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
    }

    /// Parses a var..in expression.
    fn parse_var_expr(&mut self) -> Result<Expr, &'static str> {
        // eat 'var' token
        self.advance()?;

        let name = match self.curr() {
            Ident(name) => name,
            _ => return Err("Expected identifier in 'var..in' declaration."),
        };

        self.advance()?;

        let _var_type = match self.curr() {
            AngleLeft => {
                println!("AngleLeft");
                self.parse_var_type()?
            }
            _ => VarType::new(),
        };

        // read (optional) initializer
        let initializer = match self.curr() {
            Op('=') => Some({
                self.advance()?;
                self.parse_expr()?
            }),

            _ => None,
        };

        match self.curr() {
            Comma => {
                self.advance()?;
            }
            Semicolon => {
                println!("Semicolon");
            }
            _ => return Err("Expected comma or 'in' keyword in variable declaration."),
        }

        Ok(Expr::VarDecl {
            name,
            value: Box::new(initializer.unwrap()),
        })
    }

    fn parse_var_type(&mut self) -> Result<VarType, &'static str> {
        // skip <
        self.advance();

        let mut shape = Vec::new();
        loop {
            match self.curr() {
                Token::Number(number) => {
                    shape.push(number as usize);
                    self.advance();
                }
                Token::Comma => {
                    self.advance();
                }
                Token::AngleRight => {
                    self.advance();
                    break;
                }
                _ => return Err("Cannot define variable type"),
            }
        }

        Ok(VarType { shape })
    }

    fn parse_tensor_literal_expr(&mut self) -> Result<Expr, &'static str> {
        let location = self.last_location();
        // eat [
        self.advance();
        let mut values = Vec::new();
        let mut dims = Vec::new();
        loop {
            match self.curr() {
                Token::TensorBegin => {
                    println!("Found TensorBegin");
                    values.push(self.parse_tensor_literal_expr()?);
                }
                Token::Comma => {
                    println!("Found comma");
                    self.advance();
                }
                Token::Number(_) => {
                    println!("Found TensorNumber");
                    values.push(self.parse_nb_expr()?);
                }
                Token::TensorEnd => {
                    println!("Found TensorEnd");
                    self.advance();
                    break;
                }
                _ => return Err("Cannot parse tensor expression"),
            }
            // TODO: handling error siutation
        }
        dims.push(values.len());

        Ok(Expr::Tensor {
            location,
            values,
            dims,
        })
    }

    /// Parses a primary expression (an identifier, a number or a parenthesized expression).
    fn parse_primary(&mut self) -> Result<Expr, &'static str> {
        match self.curr() {
            Ident(_) => self.parse_id_expr(),
            Number(_) => self.parse_nb_expr(),
            ParenLeft => self.parse_paren_expr(),
            Var => self.parse_var_expr(),
            TensorBegin => self.parse_tensor_literal_expr(),
            _ => Err("Unknown expression."),
        }
    }

    fn parse_block(&mut self) -> Result<Expr, &'static str> {
        // skip {,
        // TODO: check that block starts with bracket
        self.advance();

        let mut expressions = Vec::new();
        loop {
            match self.curr() {
                Token::BlockRight => {
                    println!("BlockRight");
                    self.advance();
                    break;
                }
                Token::Semicolon => {
                    println!("Semicolon");
                }
                Token::Return => {
                    let expr = self.parse_return()?;
                    expressions.push(Box::new(expr));
                    println!("Return");
                }
                Comment => {
                    self.advance();
                    continue;
                }
                _ => {
                    let expr = self.parse_primary()?;
                    expressions.push(Box::new(expr));
                }
            }
            self.advance();
        }

        Ok(Expr::ExprList { expressions })
    }

    fn parse_return(&mut self) -> Result<Expr, &'static str> {
        let location = self.last_location();
        // skip return
        self.advance();

        let expression = match self.curr() {
            // TODO: argument is optional, no need to return 0 if no argument
            Token::Semicolon => Box::new(Expr::Number(0.0)),
            _ => Box::new(self.parse_expr()?),
        };

        Ok(Expr::Return {
            location,
            expression,
        })
    }
}
