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
    """
    @description: Start running cases on one framework, only support
        parrots, torch or xla now. You should prepare the environment
        it needs first and set environment variable FRAMEWORK=parrots
        or torch、xla.
    @param {
        run_case_names: The cases you want to run.
    }
    @return {
        It will print the running stage and time it costs if it works,
        and save function and performance info in json, you can use
        export_result_to_excel.py to transfer it to csv or xlsx.
    }
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


def set_running_config():
    raise NotImplementedError("Do not support setting running config yet.")


def show_all_cases():
    for idx, case_name in enumerate(registry.key_iters()):
        print(idx, ":", case_name)


def show_all_tags():
    for tag in SampleTag:
        print(tag.value)


def show_samples_of_tag(samples_of_tag):
    tag = SampleTag(samples_of_tag)
    for sample_name in registry.key_iters():
        sample_config = registry.get(sample_name).sample_config_getter()
        if tag in sample_config.tags:
            print(sample_name)


def main():
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
