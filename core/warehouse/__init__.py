"""
数据仓库层（core/warehouse）。

手册核心论点：「报告是切片，仓库是全集；截图只能做证据，不能做数据本体」。
本包把已采集的八维 TestRun 数据变成可筛选、可追溯、可对外口径的全集行：

- templates：手册 #templates 的三套字段模板（verbatim）。
- query：跨八维筛选 + 单条投影 + 硬件×模型透视矩阵。
- export：按模板顺序导出 CSV / JSON / ZIP。
"""

from __future__ import annotations

from core.warehouse.capability_sheet import (
    CAPABILITY_COLUMNS,
    build_capability_markdown,
    build_capability_sheet,
)
from core.warehouse.export import (
    export_all_templates_zip,
    export_template_csv,
    export_template_json,
    template_header_note,
)
from core.warehouse.query import (
    CrossMatrix,
    WarehouseFilter,
    build_cross_matrix,
    build_hardware_inventory_rows,
    build_hm_test_rows,
    build_ma_test_rows,
    build_ma_test_rows_from_cases,
    distinct_values,
    project_run,
    query_runs,
)
from core.warehouse.templates import (
    HARDWARE_INVENTORY_FIELDS,
    HM_TEST_FIELDS,
    MA_TEST_FIELDS,
    TEMPLATE_DESCRIPTIONS,
    TEMPLATE_FIELDS,
    TEMPLATE_TITLES,
    all_template_names,
    template_fields,
)

__all__ = [
    # templates
    "HARDWARE_INVENTORY_FIELDS",
    "HM_TEST_FIELDS",
    "MA_TEST_FIELDS",
    "TEMPLATE_FIELDS",
    "TEMPLATE_TITLES",
    "TEMPLATE_DESCRIPTIONS",
    "template_fields",
    "all_template_names",
    # query
    "WarehouseFilter",
    "CrossMatrix",
    "project_run",
    "query_runs",
    "distinct_values",
    "build_hm_test_rows",
    "build_hardware_inventory_rows",
    "build_ma_test_rows",
    "build_ma_test_rows_from_cases",
    "build_cross_matrix",
    # capability sheet
    "CAPABILITY_COLUMNS",
    "build_capability_sheet",
    "build_capability_markdown",
    # export
    "export_template_csv",
    "export_template_json",
    "export_all_templates_zip",
    "template_header_note",
]
