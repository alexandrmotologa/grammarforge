# GBNF Grammar Guide

This guide covers GGML BNF (GBNF) syntax, rule construction, and compiler formatting standards used by llama.cpp and vLLM.

## What is GBNF?

GBNF (GGML Backus-Naur Form) is an extension of standard BNF designed for grammar-constrained sampling in llama.cpp and local LLM engines. During autoregressive token generation, the runtime parses candidate tokens against the GBNF grammar, setting logits for non-matching tokens to negative infinity.

## Syntax Specification

### 1. Production Rules

A rule consists of a non-terminal name, the assignment operator `::=`, and an expression:

```ebnf
root ::= object
```

Rule names may contain alphanumeric characters, underscores, and hyphens (`[a-zA-Z0-9_-]+`). The first rule or a rule explicitly named `root` serves as the entry point for the grammar.

### 2. Terminals (Literals)

Terminals are enclosed in double quotes:

```ebnf
comma ::= ","
bool-true ::= "true"
```

Inside literal strings, special characters must be escaped:
- Quote: `\"`
- Backslash: `\\`
- Line break: `\n`
- Carriage return: `\r`
- Tab: `\t`

### 3. Character Classes

Character classes define sets or ranges of allowed single characters:

```ebnf
digit ::= [0-9]
hex ::= [0-9a-fA-F]
identifier-char ::= [a-zA-Z_]
```

Inverted classes match any character except those listed:

```ebnf
non-quote ::= [^"]
unescaped-char ::= [^"\\]
```

### 4. Operators

- **Sequence** (Juxtaposition): `A B` matches `A` followed by `B`.
- **Alternation** (`|`): `A | B` matches either `A` or `B`.
- **Zero or More** (`*`): `A*` matches zero or more occurrences of `A`.
- **One or More** (`+`): `A+` matches one or more occurrences of `A`.
- **Optional** (`?`): `A?` matches zero or one occurrence of `A`.
- **Grouping** (`(...)`): `(A | B) C` groups expressions for precedence.

## Compiling JSON Schema to GBNF

JSON requires structural whitespace between delimiters (`{`, `}`, `[`, `]`, `:`, `,`).

GrammarForge provides three whitespace modes:

1. **Lenient (`WhitespaceMode.LENIENT`)**:
   Standard GBNF whitespace rule:
   ```ebnf
   ws ::= [ \t\n\r]*
   ```
   Inserted between object braces, array brackets, and delimiters.

2. **Compact (`WhitespaceMode.COMPACT`)**:
   Eliminates all whitespace rules. Forces minified JSON output:
   ```ebnf
   root ::= "{" "key" ":" "val" "}"
   ```

3. **Pretty (`WhitespaceMode.PRETTY`)**:
   Enforces 2-space indented formatting:
   ```ebnf
   ws ::= [ \n]*
   ```

## Best Practices for LLM Constrained Decoding

- **Enforce Canonical Ordering**: When object schemas contain many properties, enforcing fixed key order prevents state explosions and keeps step latency low.
- **Constrain Strings with Regular Expressions**: Open strings allow arbitrary generation. Whenever possible, specify `pattern` or `format` constraints to narrow allowed token sets.
- **Set Upper Bounds**: For arrays, provide `maxItems` to prevent the model from generating infinite lists.
