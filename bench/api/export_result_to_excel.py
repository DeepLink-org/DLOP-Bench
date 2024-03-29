# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import Settings, FrameType
from bench.core import excel


def export_to_excel(saving_format):
    """Transform perf time info saved in json file to excel.

    Args: 
        saving_format(str): `xlsx` or `csv`.
    """
    settings = Settings()
    case_names = excel.get_all_case_names(settings)
    heads = excel.get_table_head(FrameType, settings.frame_type_to_frame_modes)
    results = excel.collect_func_and_perf_info(
        case_names, FrameType, settings, settings.frame_type_to_frame_modes)

    if saving_format == "xlsx":
        excel.save_to_xlsx(heads, case_names, results, settings.result_dir)
    elif saving_format == "csv":
        excel.save_to_csv(heads, case_names, results, settings.result_dir)
    else:
        raise Exception("Do not support format:{}".format(saving_format))


if __name__ == "__main__":
    export_to_excel("csv")
