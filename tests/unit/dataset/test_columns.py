"""Unit tests for column definitions and types."""

import pytest

from src.dataset.columns import BaseColumn, CategoricalColumn, FreeTextColumn, NumericColumn


class TestNumericColumn:
    """Test NumericColumn class."""

    def test_init_with_valid_parameters(self):
        """Test NumericColumn initialization with valid parameters."""
        col = NumericColumn("age", mean=35.5, std=10.2)

        assert col.column_name == "age"
        assert col.mean == 35.5
        assert col.std == 10.2

    def test_init_with_zero_values(self):
        """Test NumericColumn with zero mean and std."""
        col = NumericColumn("constant_feature", mean=0.0, std=0.0)

        assert col.column_name == "constant_feature"
        assert col.mean == 0.0
        assert col.std == 0.0

    def test_init_with_negative_mean(self):
        """Test NumericColumn with negative mean."""
        col = NumericColumn("temperature", mean=-5.3, std=15.0)

        assert col.column_name == "temperature"
        assert col.mean == -5.3
        assert col.std == 15.0

    def test_column_name_attribute(self):
        """Test that column_name attribute is properly inherited."""
        col = NumericColumn("salary", mean=50000, std=20000)

        assert hasattr(col, "column_name")
        assert col.column_name == "salary"

    def test_numeric_column_is_base_column(self):
        """Test that NumericColumn is a subclass of BaseColumn."""
        col = NumericColumn("value", mean=100, std=20)

        assert isinstance(col, BaseColumn)


class TestCategoricalColumn:
    """Test CategoricalColumn class."""

    def test_init_with_categories(self):
        """Test CategoricalColumn initialization."""
        categories = ["small", "medium", "large"]
        col = CategoricalColumn("size", categories)

        assert col.column_name == "size"
        assert col.categories == categories

    def test_init_with_empty_categories(self):
        """Test CategoricalColumn with empty categories list."""
        col = CategoricalColumn("empty", [])

        assert col.column_name == "empty"
        assert col.categories == []

    def test_init_with_single_category(self):
        """Test CategoricalColumn with single category."""
        col = CategoricalColumn("status", ["active"])

        assert col.column_name == "status"
        assert len(col.categories) == 1
        assert col.categories[0] == "active"

    def test_init_with_numeric_categories(self):
        """Test CategoricalColumn can store numeric category values."""
        col = CategoricalColumn("class", [0, 1, 2])

        assert col.categories == [0, 1, 2]

    def test_categorical_column_is_base_column(self):
        """Test that CategoricalColumn is a subclass of BaseColumn."""
        col = CategoricalColumn("color", ["red", "blue"])

        assert isinstance(col, BaseColumn)


class TestFreeTextColumn:
    """Test FreeTextColumn class."""

    def test_init_with_column_name(self):
        """Test FreeTextColumn initialization."""
        col = FreeTextColumn("description")

        assert col.column_name == "description"

    def test_init_with_empty_string_name(self):
        """Test FreeTextColumn with empty string column name."""
        col = FreeTextColumn("")

        assert col.column_name == ""

    def test_freetext_column_is_base_column(self):
        """Test that FreeTextColumn is a subclass of BaseColumn."""
        col = FreeTextColumn("notes")

        assert isinstance(col, BaseColumn)

    def test_different_freetext_instances_independent(self):
        """Test that different FreeTextColumn instances are independent."""
        col1 = FreeTextColumn("text1")
        col2 = FreeTextColumn("text2")

        assert col1.column_name != col2.column_name
        assert col1 is not col2


class TestBaseColumn:
    """Test BaseColumn abstract class."""

    def test_base_column_cannot_be_instantiated(self):
        """Test that BaseColumn cannot be directly instantiated."""
        with pytest.raises(TypeError):
            BaseColumn("test_column")

    def test_base_column_subclasses_can_be_instantiated(self):
        """Test that BaseColumn subclasses can be instantiated."""
        col1 = NumericColumn("num", 0, 1)
        col2 = CategoricalColumn("cat", [])
        col3 = FreeTextColumn("text")

        assert isinstance(col1, BaseColumn)
        assert isinstance(col2, BaseColumn)
        assert isinstance(col3, BaseColumn)


class TestColumnInheritance:
    """Test inheritance and polymorphism of column types."""

    def test_all_columns_have_column_name(self):
        """Test that all column types have column_name attribute."""
        numeric = NumericColumn("num", 10, 2)
        categorical = CategoricalColumn("cat", ["a", "b"])
        freetext = FreeTextColumn("text")

        assert hasattr(numeric, "column_name")
        assert hasattr(categorical, "column_name")
        assert hasattr(freetext, "column_name")

    def test_column_names_are_preserved(self):
        """Test that column names are correctly preserved."""
        columns = [
            NumericColumn("feature1", 0, 1),
            CategoricalColumn("feature2", ["x", "y"]),
            FreeTextColumn("feature3"),
        ]

        names = [col.column_name for col in columns]
        assert names == ["feature1", "feature2", "feature3"]
