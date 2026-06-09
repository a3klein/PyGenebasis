"""
pygenebasis — command line interface entry point.
"""
from __future__ import annotations

try:
    import rich_click as click  # type: ignore
    click.rich_click.USE_RICH_MARKUP = True
    click.rich_click.SHOW_ARGUMENTS = True
except ImportError:
    import click  # type: ignore

from ._panel import panel_group
from ._reliability import reliability_group


@click.group()
@click.version_option()
def cli():
    """pygenebasis — gene panel design from scRNA-seq data.

    \b
    Commands
    --------
      panel search      Greedy gene panel selection
      panel trim        Remove redundant genes from a panel
      panel evaluate    Evaluate neighbourhood preservation
      reliability score-coexp   Score co-expression reliability (MERFISH vs scRNA-seq)
      reliability perturb       Expression perturbation analysis
    """


cli.add_command(panel_group)
cli.add_command(reliability_group)


def main():
    cli()


if __name__ == "__main__":
    main()
