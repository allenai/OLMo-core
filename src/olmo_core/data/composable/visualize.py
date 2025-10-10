from typing import List, Optional, Union

import rich

from .instance_source import InstanceSource
from .token_source import TokenSource
from .utils import format_token_count


def visualize_source(self):
    skip_connector, child_connector, leaf_child_connector = "│  ", "├─ ", "└─ "

    def _format_label(label: str) -> str:
        if len(label) > 53:
            return label[:13] + "…" + label[-40:]
        else:
            return label

    def _format_source(
        source_cls, tokens: int, label: Optional[str], indent_spec: List[bool], count: int = 1
    ) -> str:
        indents = []
        for is_leaf in indent_spec[:-1]:
            indents.append("   " if is_leaf else skip_connector)
        if indent_spec:
            indents.append(leaf_child_connector if indent_spec[-1] else child_connector)
        indent = "".join(indents)
        count_str = f" x {count}" if count > 1 else ""
        label_str = rf" \[{_format_label(label)}]" if label else ""
        token_str = format_token_count(tokens)
        return f"{indent}[b cyan]{source_cls.__name__}[/]{count_str}: [green]{token_str}[/] tokens[magenta]{label_str}[/]"

    def _visualize_source(
        source: Union[InstanceSource, TokenSource], indent_spec: List[bool]
    ) -> str:
        lines = [_format_source(type(source), source.num_tokens, source.label, indent_spec)]
        children = list(source.children())

        children_types = set()
        children_labels = set()
        children_are_leafs = True
        for child in children:
            children_types.add(type(child))
            children_labels.add(child.label)
            if list(child.children()):
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
                    total_child_tokens,
                    label,
                    indent_spec + [True],
                    count=len(children),
                )
            )
        else:
            for i, child in enumerate(children):
                lines.append(_visualize_source(child, indent_spec + [i == len(children) - 1]))

        return "\n".join(lines)

    rich.get_console().print(_visualize_source(self, []), highlight=False)
