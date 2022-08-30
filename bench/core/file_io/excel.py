# Copyright (c) OpenComputeLab. All Rights Reserved.

import os
import csv
import json
import math
from bench.common import FrameType
from bench.common.types import PatModes


def gen_json_file_path(settings, frame_type):
    """Generate json file path.
    
    Args:
        settings(Settings): Benchmark settings.
        frame_type(FrameType): Backend frame type.
    Returns
        str: json path.
        """
    return os.path.join(settings.result_dir, frame_type.value + ".json")


def get_all_case_names(settings):
    """Load sample performance json file and 
    get all sample names.

    Args:
        settings(Settings): Benchmark settings.
    Returns:
        list: Sample names.
    """
    case_names = set()
    for frame_type in FrameType:
        json_file_path = gen_json_file_path(settings, frame_type)
        if not os.path.exists(json_file_path):
            continue
        with open(json_file_path, "r") as f:
            case_results = json.load(f)
        names = set([k for k in case_results])
        case_names = case_names.union(names)
    case_names = list(case_names)
    case_names.sort()
    return case_names


def cal_geometric_mean(speedup):
    a = 1
    for b in speedup:
        a *= b
    return math.pow(a, 1.0 / len(speedup))


def get_table_head(frame_types, frame_type_to_frame_modes):
    """Generate excel table head name.

    Args:
        frame_types(FrameType): Backend frame types.
        frame_type_to_frame_modes(dict): FrameType to stage modes.
    Returns:
        list: excel table head names.
    """
    heads = ["name"]
    for frame in frame_types:
        for mode in frame_type_to_frame_modes[frame]:
            if mode.value is None:
                continue
            heads.append(frame.value + "_" + mode.value)
        heads.append(frame.value + "_" + "Errors")
    heads.append("Source")
    heads.append("Url")
    heads.append("Tags")
    heads.append("Pat_S5_SpeedUp")
    return heads


def collect_func_and_perf_info(
    case_names, frame_types, settings, frame_type_to_frame_modes
):
    """Collect sample performance info.

    Args:
        case_names(list): sample names.
        frame_types(FrameType): Backend frame types.
        settings(Settings): Benchmark settings.
        frame_type_to_frame_modes(dict): FrameType to stage modes.
    Returns:
        dict: Sample perforrmance info.
    """
    json_re = {}
    for frame in frame_types:
        json_file_path = gen_json_file_path(settings, frame)
        if not os.path.exists(json_file_path):
            continue
        with open(json_file_path, "r") as f:
            case_results = json.load(f)
        json_re[frame.value] = case_results

    results = {}
    for case_name in case_names:
        re = [case_name]
        for frame in frame_types:
            for mode in frame_type_to_frame_modes[frame]:
                if mode.value is None:
                    continue
                if (
                    frame.value in json_re
                    and case_name in json_re[frame.value]
                    and mode.value in json_re[frame.value][case_name]["times"]
                ):
                    re.append(
                        json_re[frame.value][case_name]["times"][mode.value]
                    )
                else:
                    re.append(None)

            if frame.value in json_re and case_name in json_re[frame.value]:
                error_info = ""
                for k in json_re[frame.value][case_name]["errors"]:
                    error_info = (
                        error_info
                        + str(k)
                        + " Error:"
                        + json_re[frame.value][case_name]["errors"][k]
                        + "\n"
                    )
                re.append(error_info)
            else:
                re.append(None)

        if (
            FrameType.Parrots.value in json_re
            and case_name in json_re[FrameType.Parrots.value]
        ):
            re.append(json_re[FrameType.Parrots.value][case_name]["source"])
            re.append(json_re[FrameType.Parrots.value][case_name]["url"])
            re.append(json_re[FrameType.Parrots.value][case_name]["tags"])
        else:
            re.append([None, None, None])

        if (
            FrameType.Parrots.value in json_re
            and case_name in json_re[FrameType.Parrots.value]
            and PatModes.S5.value
            in json_re[FrameType.Parrots.value][case_name]["times"]
            and PatModes.S1.value
            in json_re[FrameType.Parrots.value][case_name]["times"]
            and json_re[FrameType.Parrots.value][case_name]["times"][
                PatModes.S1.value
            ]
            is not None
            and json_re[FrameType.Parrots.value][case_name]["times"][
                PatModes.S5.value
            ]
            is not None
        ):
            s1_time = json_re[FrameType.Parrots.value][case_name]["times"][
                PatModes.S1.value
            ]
            s5_time = json_re[FrameType.Parrots.value][case_name]["times"][
                PatModes.S5.value
            ]
            re.append(s1_time / s5_time)
        else:
            re.append(None)
        results[case_name] = re
    return results


def save_to_csv(heads, case_names, all_results, result_dir):
    """Save sample performance info to csv.
    """
    file_path = os.path.join(result_dir, "results.csv")

    content_to_csv = []
    s5_count = 0
    s5_speedup_count = 0
    s5_speedup = []
    for case_name in case_names:
        if case_name in all_results:
            content_to_csv.append(all_results[case_name])
            if all_results[case_name][5] is not None:
                s5_count = s5_count + 1
            if all_results[case_name][16] is not None:
                s5_speedup.append(all_results[case_name][16])
            if (
                all_results[case_name][16] is not None
                and all_results[case_name][16] >= 1.5
            ):
                s5_speedup_count = s5_speedup_count + 1
        else:
            content_to_csv.append([])

    content_to_csv.append(["Sample Number Passed Pat_S5:", s5_count])
    content_to_csv.append(
        ["Sample Number Pat_S5 SpeedUp >= 1.5:", s5_speedup_count]
    )
    content_to_csv.append(
        ["Samples SpeedUp Geometric Mean:", cal_geometric_mean(s5_speedup)]
    )
    with open(file_path, "w") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(heads)
        f_csv.writerows(content_to_csv)


def save_to_xlsx(heads, case_names, all_results, result_dir):
    """Save sample performance info to xlsx.
    """
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active

    ws.append(heads)
    for case_name in case_names:
        if case_name in all_results:
            ws.append(all_results[case_name])
        else:
            ws.append([])
    print(all_results)
    wb.save(os.path.join(result_dir, "results.xlsx"))
