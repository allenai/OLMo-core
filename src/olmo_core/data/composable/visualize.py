from typing import List, Optional

import rich

from .source_abc import SourceABC
from .utils import format_token_count


def visualize_source(source, icons: bool = True):
    skip_connector, child_connector, last_child_connector = "│  ", "├─ ", "└─ "

    def _format_label(label: str) -> str:
        if len(label) > 53:
            return label[:13] + "..." + label[-40:]
        else:
            return label

    def _format_source(
        source_cls,
        *,
        tokens: int,
        label: Optional[str],
        indent_spec: List[bool],
        is_leaf: bool,
        count: int = 1,
        fingerprint: Optional[str] = None,
    ) -> str:
        del is_leaf
        indents = []
        for is_last_child in indent_spec[:-1]:
            indents.append("   " if is_last_child else skip_connector)
        if indent_spec:
            indents.append(last_child_connector if indent_spec[-1] else child_connector)
        indent = "".join(indents)
        icon_str = f"{source_cls.DISPLAY_ICON} " if icons and source_cls.DISPLAY_ICON else ""
        count_str = f" x {count}" if count > 1 else ""
        label_str = rf" \[{_format_label(label)}]" if label else ""
        token_str = format_token_count(tokens)
        fingerprint_str = f"({fingerprint[:7]})" if fingerprint else ""
        return (
            f"{indent}[b cyan]{icon_str}{source_cls.__name__}[/][cyan]{fingerprint_str}[/]{count_str}: "
            f"[green]{token_str}[/] tokens[magenta]{label_str}[/]"
        )

    def _visualize_source(source: SourceABC, indent_spec: List[bool]) -> str:
        lines = [
            _format_source(
                type(source),
                tokens=source.num_tokens,
                label=source.label,
                indent_spec=indent_spec,
                is_leaf=source.is_leaf,
                fingerprint=source.fingerprint,
            )
        ]
        children = list(source.children())

        children_types = set()
        children_labels = set()
        children_are_leafs = True
        for child in children:
            children_types.add(type(child))
            children_labels.add(child.label)
            if not child.is_leaf:
                children_are_leafs = False
                break

        if (
            len(children) > 1
            and len(children_types) == 1
            and len(children_labels) == 1
            and children_are_leafs
        ):
            total_child_tokens = sum(child.num_tokens for child in children)
            label = list(children_labels)[0]
            lines.append(
                _format_source(
                    children[0].__class__,
                    tokens=total_child_tokens,
                    label=label,
                    indent_spec=indent_spec + [True],
                    is_leaf=True,
                    count=len(children),
                )
            )
        else:
            for i, child in enumerate(children):
                lines.append(_visualize_source(child, indent_spec + [i == len(children) - 1]))

        return "\n".join(lines)

    rich.get_console().print(_visualize_source(source, []), highlight=False)
