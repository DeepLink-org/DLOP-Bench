from long_tail_bench.common import Settings, FrameType
from long_tail_bench.core import excel


def export_to_excel(saving_format):
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
