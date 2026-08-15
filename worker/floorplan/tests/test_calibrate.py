from floorplan2dxf.calibrate import standalone_lengths_mm
from floorplan2dxf.schema import TextItem


def _text(content: str) -> TextItem:
    return TextItem(content=content, position=(0, 0), kind="other")


def test_metric_plan_bare_overall_dimensions_are_inferred_as_metres():
    values = standalone_lengths_mm([
        _text("GROUND FLOOR PLAN"), _text("SCALE 1:100 MTS."),
        _text("9.00"), _text("11.00"),
    ])
    assert values == [9000.0, 11000.0]


def test_bare_values_without_metric_plan_context_are_not_assumed():
    assert standalone_lengths_mm([_text("9.00"), _text("11.00")]) == []
