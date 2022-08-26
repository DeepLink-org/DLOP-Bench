import argparse
from bench.common import (
    Settings,
    FRAMEWORK,
    SAMPLE_IMPL,
    BENCH_DEBUG,
    SampleTag,
    PatExecMode,
)
from bench.core import registry
from bench.core.engine import Engine

# Register all cases
from bench.samples import basic  # noqa
from bench.samples import long_tail   # noqa

if BENCH_DEBUG:
    from bench.core.executer import log_debug_info

    log_debug_info()


def parse_args():
    """Parse arguments.

    See readme for detail uasge of api argement.

    Returns:
        argparse.Namespace: Args parsed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cases",
        help="the cases to run, split by `,`",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--show_all_cases",
        help="show all cases registered",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-sg",
        "--show_all_tags",
        help="show all tags of samples supported",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-sc",
        "--show_sample_config",
        help="show config of running samples",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-st",
        "--stages",
        help="The stages to run, split by `,`",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-sot",
        "--samples_of_tag",
        help="show all samples of a tag type",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-pem",
        "--parrots_exec_mode",
        help="parrots exec mode, `sync` or `async`",
        type=str,
        default="sync",
    )

    args = parser.parse_args()
    args.parrots_exec_mode = PatExecMode(args.parrots_exec_mode)
    args.cases = str(args.cases).split(",") if args.cases is not None else None
    if args.stages is not None:
        args.stages = str(args.stages).split(",")
        args.stages = [int(st) for st in args.stages]
        if 1 not in args.stages:
            args.stages = [1] + args.stages
    return args


def run(
    run_case_names=None,
    run_stages=None,
    show_config=False,
    parrots_exec_mode=PatExecMode.SYNC,
):
    """Start running cases on one framework.
    Only support parrots, torch or xla now. You should prepare the environment
    it needs first and set environment variable FRAMEWORK=parrots or torch、xla.
    
    Args: 
        run_case_names(list): The sample names you want to run.
        run_stages(list): The stage number you want to run. 1: eager stage, 
            2: fixed shape jit stage, 3: fixed shape coder stage,
            4: dynamic shape jit stage, 5: dynamic shape coder stage, 
            2、3、4 just for parrots compiler.
        show_config(bool): Whether show config of samples.
        parrots_exec_mode(PatExecMode): The execution mode of parrots backend.
    """
    impl = FRAMEWORK if SAMPLE_IMPL is None else SAMPLE_IMPL
    print("Executer Backend:", FRAMEWORK.value)
    print("Sample Impl:", impl.value)

    if run_case_names is not None:
        print("case names to run:", run_case_names)
        for case_name in run_case_names:
            registry.exists(case_name)
    benchmark_engine = Engine(
        FRAMEWORK,
        Settings(),
        registry,
        run_case_names,
        run_stages,
        show_config,
        parrots_exec_mode,
    )
    benchmark_engine.run()


def show_all_cases():
    """Print all sample names.
    """
    for idx, case_name in enumerate(registry.key_iters()):
        print(idx, ":", case_name)


def show_all_tags():
    """Print all sample tags.
    """
    for tag in SampleTag:
        print(tag.value)


def show_samples_of_tag(samples_of_tag):
    """Show all sample names which have the specified tag.
    """
    tag = SampleTag(samples_of_tag)
    for sample_name in registry.key_iters():
        sample_config = registry.get(sample_name).sample_config_getter()
        if tag in sample_config.tags:
            print(sample_name)


def main():
    """Parse args and run benchmark.
    """
    args = parse_args()
    if args.show_all_cases:
        show_all_cases()
        return
    if args.show_all_tags:
        show_all_tags()
        return
    if args.samples_of_tag is not None:
        show_samples_of_tag(args.samples_of_tag)
        return
    run(
        args.cases, args.stages, args.show_sample_config,
        args.parrots_exec_mode
    )


if __name__ == "__main__":
    main()
