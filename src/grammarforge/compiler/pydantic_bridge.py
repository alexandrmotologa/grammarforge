"""Bridge connecting Pydantic models and dataclasses directly to GBNF grammars."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, TypeAdapter
from grammarforge.compiler.json_schema import compile_json_schema
from grammarforge.core.config import CompilerConfig
from grammarforge.core.exceptions import SchemaCompilationError


def get_schema_from_type(type_cls: Type[Any]) -> Dict[str, Any]:
    """Extract standard JSON Schema from a Pydantic model, dataclass, or typed structure."""
    if hasattr(type_cls, "model_json_schema"):
        # Pydantic v2
        return type_cls.model_json_schema()
    elif hasattr(type_cls, "schema"):
        # Pydantic v1
        return type_cls.schema()
    else:
        # Dataclass, TypedDict, or standard type via TypeAdapter
        adapter = TypeAdapter(type_cls)
        return adapter.json_schema()


def compile_pydantic_model(model_cls: Type[Any], config: Optional[CompilerConfig] = None) -> str:
    """Compile a Pydantic model or dataclass directly into a GBNF grammar string."""
    schema = get_schema_from_type(model_cls)
    return compile_json_schema(schema, config=config)


def load_model_from_spec(spec: str) -> Type[Any]:
    """Load a model class from an import specification.

    Supported formats:
    - "module.submodule:ModelName"
    - "path/to/file.py:ModelName"
    """
    if ":" not in spec:
        raise SchemaCompilationError(
            f"Invalid model specification '{spec}'. Format must be 'module_or_file:ClassName'."
        )

    target_path, class_name = spec.rsplit(":", 1)
    target_path = target_path.strip()
    class_name = class_name.strip()

    if target_path.endswith(".py") or os.path.sep in target_path or "/" in target_path:
        # Load from file path
        abs_path = os.path.abspath(target_path)
        if not os.path.isfile(abs_path):
            raise SchemaCompilationError(f"Model file not found: {abs_path}")

        module_name = f"_grammarforge_dyn_{os.path.splitext(os.path.basename(abs_path))[0]}"
        mod_spec = importlib.util.spec_from_file_location(module_name, abs_path)
        if mod_spec is None or mod_spec.loader is None:
            raise SchemaCompilationError(f"Could not load module specification from {abs_path}")

        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[module_name] = module
        try:
            mod_spec.loader.exec_module(module)
        except Exception as err:
            raise SchemaCompilationError(f"Failed to execute module {abs_path}: {err}") from err
    else:
        # Standard import
        try:
            module = importlib.import_module(target_path)
        except ImportError as err:
            raise SchemaCompilationError(f"Failed to import module '{target_path}': {err}") from err

    if not hasattr(module, class_name):
        raise SchemaCompilationError(
            f"Class '{class_name}' not found in '{target_path}'."
        )

    cls = getattr(module, class_name)
    return cls


def compile_pydantic_source(source_code: str, class_name: str, config: Optional[CompilerConfig] = None) -> str:
    """Execute Python source code containing a Pydantic model and compile it to GBNF."""
    namespace: Dict[str, Any] = {}
    try:
        exec(source_code, namespace)
    except Exception as err:
        raise SchemaCompilationError(f"Failed to execute Python source: {err}") from err

    if class_name not in namespace:
        raise SchemaCompilationError(f"Class '{class_name}' was not defined in source code.")

    model_cls = namespace[class_name]
    return compile_pydantic_model(model_cls, config=config)
