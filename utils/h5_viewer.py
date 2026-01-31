# Trajecto: Real-time 3D Trajectory Reconstruction System
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# This file incorporates code from H5View (https://github.com/rossant/h5view)
# Modified by Eunkyum Kim to add Sample Delete Feature.
#
# Original H5View License:
# Copyright (c) 2012, Cyrille Rossant. All rights reserved.
# Licensed under the BSD 3-Clause License.
#
# Entire Work Licensed under the Apache License, Version 2.0.
# See the LICENSE and NOTICE files for details.

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, OptionList, Static, DataTable, Button
from textual.containers import VerticalScroll, Horizontal, Container, Vertical
from textual.binding import Binding
from textual.screen import ModalScreen
from textual_plotext import PlotextPlot

import h5py
import numpy as np
import pandas as pd

import sys
import os
import argparse
from typing import Any, List, Optional, cast

UNICODE_SUPPORT = sys.stdout.encoding.lower().startswith("utf")


class ConfirmationScreen(ModalScreen[bool]):
    """A modal screen to confirm a destructive action."""

    BINDINGS = [
        Binding("y", "confirm_yes", "Yes", show=True),
        Binding("n", "confirm_no", "No", show=True),
    ]

    def __init__(self, message: str) -> None:
        """Initialize the confirmation screen.

        Args:
            message: The message to display to the user.
        """
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        """Compose the confirmation screen.

        Returns:
            The composed screen.
        """
        with Vertical(id="dialog", classes="confirm-dialog"):
            yield Static(self.message, classes="confirm-prompt")
            with Horizontal(classes="confirm-buttons"):
                yield Button("Yes (y)", variant="error", id="yes")
                yield Button("No (n)", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events.

        Args:
            event: The button press event.
        """
        if event.button.id == "yes":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_confirm_yes(self) -> None:
        """Confirm the action and dismiss the screen."""
        self.dismiss(True)

    def action_confirm_no(self) -> None:
        """Cancel the action and dismiss the screen."""
        self.dismiss(False)


class AttributeScreen(ModalScreen[None]):
    """A modal screen to display the attributes of an HDF5 item."""

    BINDINGS = [
        Binding(
            "left,h,q",
            "quit_attrs",
            "Return",
            show=True,
            priority=True,
        ),
        Binding("down,j", "cursor_down", "Down", show=True, priority=True),
        Binding("up,k", "cursor_up", "Up", show=True, priority=True),
        Binding("J", "scroll_content_down", "Scroll Down", priority=True),
        Binding("K", "scroll_content_up", "Scroll Up", priority=True),
        Binding("u", "scroll_content_page_up", "Scroll Down", priority=True),
        Binding("d", "scroll_content_page_down", "Scroll Up", priority=True),
    ]

    def __init__(
        self,
        h5file: h5py.File,
        cur_dir: str,
        itemname: str,
        id: Optional[str] = None,
    ) -> None:
        """Initialize the attribute screen.

        Args:
            h5file: The HDF5 file object.
            cur_dir: The current directory in the HDF5 file.
            itemname: The name of the item whose attributes are to be
                displayed.
            id: The ID of the screen. Defaults to None.
        """
        super().__init__(id=id)
        self._file = h5file
        self._cur_dir = cur_dir
        self._itemname = itemname
        self._item = self._file[self._cur_dir + f"/{self._itemname}"]
        self._attrs = list(self._item.attrs.keys())

        self._cur_attr = self._attrs[0]

        self._selector_widget = MyOptionList(*self._attrs, markup=False)
        self._selector_widget.border_title = f"Attributes for {self._itemname}"

        self._content_widget = Static(id="attr_content", markup=False)
        self._vertical_widget = VerticalScroll(
            self._content_widget, id="attr_content_scroll"
        )

        self.update_content()

    def compose(self) -> ComposeResult:
        """Compose the attribute screen.

        Returns:
            The composed screen.
        """
        with Vertical(id="dialog"):
            yield self._selector_widget
            yield self._vertical_widget
        yield Footer()

    def update_content(self) -> None:
        """Update the content of the attribute view."""
        content = str(self._item.attrs[self._cur_attr])
        self._content_widget.update(content)

    def action_quit_attrs(self) -> None:
        """Quit the attribute screen."""
        self.app.pop_screen()

    def action_cursor_down(self) -> None:
        """Move the cursor down in the attribute list."""
        self._selector_widget.action_cursor_down()
        highlighted = self._selector_widget.highlighted
        if highlighted is not None:
            self._cur_attr = self._attrs[highlighted]
            self.update_content()

    def action_cursor_up(self) -> None:
        """Move the cursor up in the attribute list."""
        self._selector_widget.action_cursor_up()
        highlighted = self._selector_widget.highlighted
        if highlighted is not None:
            self._cur_attr = self._attrs[highlighted]
            self.update_content()

    def action_scroll_content_down(self) -> None:
        """Scroll the content down."""
        self._vertical_widget.scroll_down()

    def action_scroll_content_up(self) -> None:
        """Scroll the content up."""
        self._vertical_widget.scroll_up()

    def action_scroll_content_page_down(self) -> None:
        """Scroll the content down by a page."""
        self._vertical_widget.scroll_page_down()

    def action_scroll_content_page_up(self) -> None:
        """Scroll the content up by a page."""
        self._vertical_widget.scroll_page_up()


class MyDataTable(DataTable[Any]):
    """A custom data table with key bindings for navigation."""

    BINDINGS = [
        Binding("enter", "select_cursor", "Select", show=False),
        Binding("up,k", "cursor_up", "Cursor up", show=False),
        Binding("down,j", "cursor_down", "Cursor down", show=False),
        Binding("right,L", "cursor_right", "Cursor right", show=False),
        Binding("left,H", "cursor_left", "Cursor left", show=False),
        Binding("pageup,u", "page_up", "Page up", show=False),
        Binding("pagedown,d", "page_down", "Page down", show=False),
        Binding("g", "scroll_top", "Top", show=False),
        Binding("G", "scroll_bottom", "Bottom", show=False),
    ]

    def __init__(self, id: Optional[str] = None) -> None:
        """Initialize the data table.

        Args:
            id: The ID of the data table. Defaults to None.
        """
        super().__init__(id=id)

    def update(self, value: np.ndarray[Any, Any]) -> None:
        """Update the data table with new data.

        Args:
            value: The new data to display.
        """
        self.clear(columns=True)
        colnames = get_colnames(value)
        if colnames:
            self.add_columns(*colnames)
        for row in value:
            # Decode bytes to utf8 for display, otherwise keep the element as is.
            row_cleaned = [
                e.decode("utf8") if isinstance(e, bytes) else e
                for e in cast(np.ndarray[Any, Any], row).item()
            ]
            self.add_row(*row_cleaned)


def get_colnames(obj: np.ndarray[Any, Any]) -> Optional[List[str]]:
    """Return the column names of a NumPy structured array.

    Args:
        obj: The NumPy structured array.

    Returns:
        A list of column names, or None if the array is
            not structured.
    """
    return obj.dtype.names


def is_dataframe(obj: np.ndarray[Any, Any]) -> bool:
    """Checks if a NumPy array is a structured array.

    Args:
        obj: The NumPy array to check.

    Returns:
        True if the array is a structured array, False otherwise.
    """
    return len(obj.dtype) != 0


def is_plotable(obj: np.ndarray[Any, Any]) -> bool:
    """Checks if a NumPy array is plotable.

    A plotable array is not a composite type and is 1D or 2D.

    Args:
        obj: The NumPy array to check.

    Returns:
        True if the array is plotable, False otherwise.
    """
    is_not_composity_type = len(obj.dtype) == 0
    squeezed = np.squeeze(obj)
    return is_not_composity_type and (squeezed.ndim == 1 or squeezed.ndim == 2)


def is_aggregatable(obj: np.ndarray[Any, Any]) -> bool:
    """Checks if a NumPy array is aggregatable.

    An aggregatable array is a numeric numpy array with more than one element.

    Args:
        obj: The NumPy array to check.

    Returns:
        True if the array is aggregatable, False otherwise.
    """
    return (
        isinstance(obj, np.ndarray)
        and np.issubdtype(obj.dtype, np.number)
        and obj.size > 1
    )


def add_escape_chars(string: str) -> str:
    """Add escape characters to a string for rich printing.

    Args:
        string: The string to add escape characters to.

    Returns:
        The string with escape characters.
    """
    return string.replace("[", r"[[")


def remove_escaped_chars(string: str) -> str:
    """Remove escape characters from a string.

    Args:
        string: The string to remove escape characters from.

    Returns:
        The string without escape characters.
    """
    return string.replace(r"[[", "[")


class MyOptionList(OptionList):
    """A custom option list with key bindings for navigation."""

    BINDINGS = [
        Binding("down,j", "cursor_down", "Down", show=True),
        Binding("up,k", "cursor_up", "Up", show=True),
        Binding("G", "page_down", "Bottom", show=False),
        Binding("g", "page_up", "Top", show=False),
    ]

    def action_cursor_down(self) -> None:
        """Move the cursor down in the option list."""
        self.refresh_bindings()
        return super().action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move the cursor up in the option list."""
        self.refresh_bindings()
        return super().action_cursor_up()

    def check_action(self, action: str, parameters: Any) -> bool:
        """Check if an action is allowed.

        Args:
            action: The action to check.
            parameters: The parameters for the action.

        Returns:
            True if the action is allowed, False otherwise.
        """
        # Disable cursor movement in the option list when viewing a dataset.
        if action in ["cursor_down", "cursor_up"] and self.app.has_class(
            "view-dataset"
        ):
            return False
        else:
            return True


class ColumnContent(VerticalScroll):
    """A column that displays the content of a dataset."""

    BINDINGS = [
        Binding("down,j", "scroll_down", "Down", show=True),
        Binding("up,k", "scroll_up", "Up", show=True),
        Binding("pageup,u", "page_up", "Page up", show=False),
        Binding("pagedown,d", "page_down", "Page down", show=False),
        Binding("G", "scroll_end", "Bottom", show=False),
        Binding("g", "scroll_home", "Top", show=False),
    ]

    def compose(self) -> ComposeResult:
        """Compose the column content.

        Returns:
            The composed column content.
        """
        self._content: Static = Static(id="data", markup=False)
        self._plot: PlotextPlot = PlotextPlot(id="plot")
        self._df: MyDataTable = MyDataTable(id="dtable")
        yield self._content
        yield self._plot
        yield self._df

    def update_value(self, value: np.ndarray[Any, Any]) -> None:
        """Update the value of the content.

        Args:
            value: The new value to display.
        """
        # save value to be able to reference it when toggling display options
        self._value = value

    def reprint(self) -> None:
        """Reprint the content, used when numpy formatting is modified."""
        if is_dataframe(self._value):
            self.notify("Entering data table: use capital H and L to navigate columns")
            self._df.update(self._value)
            self._df.focus()

        else:
            self._content.update(f"{self._value}")

    def replot(self) -> None:
        """Plot the data, currently only supports 1D and 2D data."""
        data = np.squeeze(self._value)
        if is_plotable(data):
            self._plot.plt.clear_figure()
            if data.ndim == 1:
                self._plot.plt.xlabel("Index")
                self._plot.plt.plot(
                    np.arange(data.shape[0]),
                    data,
                    color="cyan",
                    marker="braille",
                )
            elif data.ndim == 2:
                nrows, ncols = data.shape
                # Set the plot size to match the data shape for better visualization.
                self._plot.plt.plot_size(nrows, ncols)
                # Use heatmap for small matrices and matrix_plot for larger ones for performance.
                size_threshold = 100
                if nrows < size_threshold and ncols < size_threshold:
                    self._plot.plt.heatmap(pd.DataFrame(data))
                    # heatmap has a default title, remove it
                    self._plot.plt.title("")
                else:
                    self._plot.plt.matrix_plot(data.tolist())
                self._plot.plt.xlabel("Column")
                self._plot.plt.ylabel("Row")


class Column(Container):
    """A container for the directory structure and content."""

    def __init__(self, dirs: List[str], focus: bool = False) -> None:
        """Initialize the column.

        Args:
            dirs: A list of directories to display.
            focus: Whether the column should be focused.
                Defaults to False.
        """
        super().__init__()
        self._focus = focus
        self._selector_widget: MyOptionList = MyOptionList(*dirs, id="dirs", markup=False)
        self._content_widget: ColumnContent = ColumnContent(id="content")

    def compose(self) -> ComposeResult:
        """Compose the column.

        Returns:
            The composed column.
        """
        yield self._selector_widget
        yield self._content_widget
        if self._focus:
            self._selector_widget.focus()

    def update_list(self, dirs: List[str], prev_highlighted: int) -> None:
        """Redraw the option list with the contents of the current directory.

        Args:
            dirs: A list of directories to display.
            prev_highlighted: The previously highlighted item.
        """
        self._selector_widget.clear_options()
        self._selector_widget.add_options(dirs)
        self._selector_widget.highlighted = prev_highlighted


class H5TUIApp(App[None]):
    """A simple TUI application for displaying and navigating HDF5 files."""

    BINDINGS = [
        Binding("i", "toggle_dark", "Toggle dark mode", show=False),
        Binding("q", "quit", "Quit", show=False),
        Binding("left,h", "goto_parent", "Back", show=True),
        Binding("right,l", "goto_child", "Select", show=True, priority=True),
        Binding("a", "view_attrs", "Attributes", show=True),
        Binding("d", "delete_sample", "Delete Sample", show=True),
        Binding("x", "delete_row", "Delete Row", show=True),
        Binding("t", "truncate_print", "Truncate", show=True),
        Binding("s", "suppress_print", "Suppress", show=True),
        Binding("p", "toggle_plot", "Plot", show=True),
        Binding("A", "aggregate_data", "Aggregate", show=True),
    ]
    CSS_PATH = "h5tui.tcss"
    TITLE = "h5tui"

    def __init__(self, fname: str) -> None:
        """Initialize the H5TUIApp.

        Args:
            fname: The path to the HDF5 file.
        """
        super().__init__()

        self._fname = fname
        self._file = h5py.File(fname, "a")

        self._cur_dir = str(self._file.name)
        self._dirs = self.get_dir_content(self._cur_dir)

        self._prev_highlighted = 0

        self._truncate_print = True
        self._suppress_print = False
        np.set_printoptions(linewidth=self.size.width)

        self.is_aggregated = False

    def compose(self) -> ComposeResult:
        """Compose the main layout of the application.

        Returns:
            The composed layout.
        """
        yield Header()
        yield Footer()

        self._header_widget = Static("Path: /", id="header", markup=False)
        yield self._header_widget
        with Horizontal():
            dir_with_metadata = self.add_dir_metadata()
            self._column1 = Column(dir_with_metadata, focus=True)
            yield self._column1

    def group_or_dataset(self, elem: str) -> str:
        """Return an icon or text indicating whether the element is a group or a dataset.

        Args:
            elem: The name of the element.

        Returns:
            An icon or text indicating the element type.
        """
        h5elem = self._file[self._cur_dir + f"/{elem}"]
        # Use unicode icons if supported for a richer UI.
        if UNICODE_SUPPORT:
            if isinstance(h5elem, h5py.Group):
                return "📁  "
            if isinstance(h5elem, h5py.Dataset):
                return "📊  "
        else:
            if isinstance(h5elem, h5py.Group):
                return "(Group)    "
            if isinstance(h5elem, h5py.Dataset):
                return "(DataSet)  "
        return ""

    def has_attr(self) -> bool:
        """Check if the currently selected item has attributes.

        Returns:
            True if the item has attributes, False otherwise.
        """
        highlighted = self._column1._selector_widget.highlighted
        if highlighted is not None:
            prompt = self._column1._selector_widget.get_option_at_index(
                highlighted
            ).prompt
            selected_item = self.get_itemname_from_prompt(str(prompt))
            return self.build_attr_str(selected_item) != ""
        return False

    def build_attr_str(self, elem: str) -> str:
        """Create a string indicating if an element has attributes.

        Args:
            elem: The name of the element.

        Returns:
            A string indicating the number of attributes, or an empty
                string if there are no attributes.
        """
        h5elem = self._file[self._cur_dir + f"/{elem}"]
        num_attrs = len(h5elem.attrs)
        if num_attrs > 0:
            return f"▼ ({num_attrs})"
        else:
            return ""

    def add_dir_metadata(self) -> List[str]:
        """Adds metadata to the directory listing.

        This includes icons for groups and datasets, and attribute indicators.

        Returns:
            A list of strings with metadata for each item in the directory.
        """
        items = list(self._file[self._cur_dir].keys())
        with_type_icon = [self.group_or_dataset(item) + item for item in items]
        with_attrs = [
            with_type + f"    {self.build_attr_str(item)}"
            for with_type, item in zip(with_type_icon, items)
        ]
        return with_attrs

    def get_itemname_from_prompt(self, prompt: str) -> str:
        """
        Returns the item name from the selected item in the option list.

        The selected item contains an icon for group or dset, the item name,
        and the number of attributes. This function extracts the item name.

        Args:
            prompt: The prompt from the option list.

        Returns:
            The extracted item name.
        """
        return prompt.split()[1]

    def get_dir_content(self, directory: str) -> List[str]:
        """Return contents of current path.

        Args:
            directory: The directory to get the content from.

        Returns:
            A list of strings with the content of the directory.
        """
        return list(self._file[directory].keys())

    def update_content(self, path: str) -> None:
        """Updates the content view with the selected dataset.

        Args:
            path: The path to the dataset.
        """
        dset = self._file[path]
        dset_name = os.path.basename(path)
        dset_shape = dset.shape
        dset_data = dset[...]

        self._data = dset_data
        self._datapath = path

        self.add_class("view-dataset")
        if is_dataframe(self._data):
            self.add_class("view-dtable")
            dset_dtype = [str(field[0]) for field in dset.dtype.fields.values()]
        else:
            dset_dtype = dset.dtype

        self._column1._content_widget.update_value(self._data)
        self._column1._content_widget.reprint()

        self.update_header(
            f"Path: {self._cur_dir}\nDataset: {dset_name} <{dset_dtype}> {dset_shape}"
        )

    def update_header(self, text: str) -> None:
        """Updates the header widget with the provided text.

        Args:
            text: The text to display in the header.
        """
        self._header_widget.update(text)

    def aggregate_data(self) -> dict[str, float]:
        """Aggregates the data and returns a dictionary of statistics.

        Returns:
            A dictionary of statistics.
        """
        stats: dict[str, float] = {
            "mean": float(np.mean(self._data)),
            "std": float(np.std(self._data)),
            "max": float(np.max(self._data)),
            "min": float(np.min(self._data)),
            "L2 norm": float(np.linalg.norm(self._data)),
        }
        return stats

    def check_action(self, action: str, parameters: Any) -> bool:
        """Check if an action is allowed.

        This is used to disable bindings when they are not applicable.

        Args:
            action: The action to check.
            parameters: The parameters for the action.

        Returns:
            True if the action is allowed, False otherwise.
        """
        # Disable printing, plotting, and aggregation when not viewing a dataset.
        if (
            action
            in [
                "truncate_print",
                "suppress_print",
                "toggle_plot",
                "aggregate_data",
            ]
            and not self.has_class("view-dataset")
        ):
            return False
        # Disable sample deletion when viewing a dataset.
        elif action == "delete_sample" and self.has_class("view-dataset"):
            return False
        # Disable row deletion when not viewing a dataset.
        elif action == "delete_row" and not self.has_class("view-dataset"):
            return False
        # Disable child navigation when viewing a dataset.
        elif action == "goto_child" and self.has_class("view-dataset"):
            return False
        # Disable cursor movement in the option list when viewing a dataset.
        elif action in ["cursor_down", "cursor_up"] and self.has_class("view-dataset"):
            return False
        # Disable attribute view when the item has no attributes.
        elif action == "view_attrs" and not self.has_attr():
            return False
        else:
            return True

    def action_delete_row(self) -> None:
        """Delete the selected row from the dataset."""
        if self.has_class("view-dataset"):
            datatable = self.query_one(MyDataTable)
            row_index = datatable.cursor_row
            if row_index is not None and row_index >= 0:

                def check_delete(delete: bool) -> None:
                    if delete:
                        try:
                            # Delete the row from the numpy array
                            new_data = np.delete(self._data, row_index, axis=0)

                            # Delete the old dataset and create a new one with the modified data.
                            # This is necessary because h5py datasets are immutable.
                            del self._file[self._datapath]
                            self._file.create_dataset(self._datapath, data=new_data)

                            self.notify(f"Deleted row {row_index}", timeout=2)

                            # Refresh the view
                            self.update_content(self._datapath)

                        except Exception as e:
                            self.notify(
                                f"Error deleting row: {e}",
                                severity="error",
                                timeout=5,
                            )

                self.push_screen(
                    ConfirmationScreen(
                        f"Are you sure you want to delete row {row_index}?"
                    ),
                    check_delete,
                )

    def action_delete_sample(self) -> None:
        """Delete the selected sample (group) from the HDF5 file."""
        if not self.has_class("view-dataset"):
            highlighted = self._column1._selector_widget.highlighted
            if highlighted is not None:
                prompt = self._column1._selector_widget.get_option_at_index(
                    highlighted
                ).prompt
                selected_item = self.get_itemname_from_prompt(str(prompt))
                path = os.path.join(self._cur_dir, selected_item)

                if path in self._file and isinstance(self._file[path], h5py.Group):

                    def check_delete(delete: bool) -> None:
                        if delete:
                            try:
                                del self._file[path]
                                self.notify(
                                    f"Deleted sample '{selected_item}'", timeout=2
                                )
                                # Refresh the list after deletion
                                self._column1.update_list(self.add_dir_metadata(), 0)
                            except Exception as e:
                                self.notify(
                                    f"Error deleting sample: {e}",
                                    severity="error",
                                    timeout=5,
                                )

                    self.push_screen(
                        ConfirmationScreen(
                            f"Are you sure you want to delete '{selected_item}'?"
                        ),
                        check_delete,
                    )

    def action_view_attrs(self) -> None:
        """Action to display the attribute screen."""
        highlighted = self._column1._selector_widget.highlighted
        if highlighted is not None:
            prompt = self._column1._selector_widget.get_option_at_index(
                highlighted
            ).prompt
            selected_item = self.get_itemname_from_prompt(str(prompt))
            if self.build_attr_str(selected_item) != "":
                self.push_screen(
                    AttributeScreen(self._file, self._cur_dir, selected_item)
                )
            else:
                self.notify(
                    "Selected item does not have attributes",
                    severity="warning",
                    timeout=2,
                )

    def action_aggregate_data(self) -> None:
        """Aggregates the data and displays statistics in the header."""
        if self.has_class("view-dataset"):
            if not is_aggregatable(self._data):
                self.notify(
                    "Only numeric arrays may be aggregated",
                    severity="warning",
                    timeout=2,
                )
                return

            if not self.is_aggregated:
                content = self._header_widget._renderable
                stats = self.aggregate_data()
                agg_string = (
                    "\nSummary: "
                    + "; ".join(
                        [f"{key} = {value:.5g}" for key, value in stats.items()]
                    )
                    + "; "
                )
                self.update_header(str(content) + agg_string)
                self.notify("Summarizing...", timeout=2)
                self.is_aggregated = True

    def action_toggle_dark(self) -> None:
        """Toggles dark mode."""
        self.dark = not self.dark

    def action_goto_parent(self) -> None:
        """Navigates to the parent directory or hides the dataset view."""
        has_parent_dir = self._cur_dir != "/"
        if has_parent_dir and not self.has_class("view-dataset"):
            self._cur_dir = os.path.dirname(self._cur_dir)
            self._header_widget.update(f"Path: {self._cur_dir}")
            self._column1.update_list(self.add_dir_metadata(), self._prev_highlighted)

        # Reset view states and focus
        self.is_aggregated = False
        self.remove_class("view-dataset", "view-plot", "view-dtable")
        self._column1._selector_widget.focus()
        # Reset numpy print options to default
        np.set_printoptions(suppress=False, threshold=1000)
        self.update_header(f"Path: {self._cur_dir}")
        self.refresh_bindings()

    def action_goto_child(self) -> None:
        """Navigates to a child group or displays a dataset."""
        if self.has_class("view-dataset"):
            # Do nothing if a dataset is already being viewed
            return

        highlighted = self._column1._selector_widget.highlighted
        if highlighted is not None:
            prompt = self._column1._selector_widget.get_option_at_index(
                highlighted
            ).prompt
            selected_item = self.get_itemname_from_prompt(str(prompt))
            path = os.path.join(self._cur_dir, selected_item)

            if path in self._file:
                if isinstance(self._file[path], h5py.Group):
                    self._prev_highlighted = highlighted
                    self._cur_dir = path
                    self._header_widget.update(f"Path: {self._cur_dir}")
                    self._column1.update_list(self.add_dir_metadata(), 0)
                else:
                    self.update_content(path)
        self.refresh_bindings()

    def action_truncate_print(self) -> None:
        """Change numpy printing by toggling truncation."""
        if self.has_class("view-dataset") and not self.has_class("view-plot"):
            self._truncate_print = not self._truncate_print
            if self._truncate_print:
                # Default numpy truncation threshold
                default_numpy_truncate = 1000
                np.set_printoptions(threshold=default_numpy_truncate)
                self.notify("Truncation: ON", timeout=2)
            else:
                np.set_printoptions(threshold=sys.maxsize)
                self.notify("Truncation: OFF", timeout=2)
            self._column1._content_widget.reprint()

    def action_suppress_print(self) -> None:
        """Change numpy printing by toggling suppression of scientific notation."""
        if self.has_class("view-dataset") and not self.has_class("view-plot"):
            self._suppress_print = not self._suppress_print
            if self._suppress_print:
                np.set_printoptions(suppress=True)
                self.notify("Suppression: ON", timeout=2)
            else:
                np.set_printoptions(suppress=False)
                self.notify("Suppression: OFF", timeout=1)
            self._column1._content_widget.reprint()

    def action_toggle_plot(self) -> None:
        """Toggles the plot view for plotable data."""
        if self.has_class("view-dataset"):
            if is_plotable(self._data):
                if not self.has_class("view-plot"):
                    self.notify("Plotting...", timeout=2)
                self.toggle_class("view-plot")
                self._column1._content_widget.replot()
            else:
                self.notify(
                    "Currently only 1D and 2D data is plotable", severity="warning"
                )


def check_file_validity(fname: Optional[str]) -> bool:
    """Checks if the provided file is a valid HDF5 file.

    Args:
        fname: The path to the file.

    Returns:
        True if the file is valid, False otherwise.
    """
    if not fname:
        print("No HDF5 file provided")
        print("Usage: h5tui <file>.h5")
        return False

    if not os.path.exists(fname):
        print(f"Error: File '{fname}' not found.")
        return False

    if not h5py.is_hdf5(fname):
        print(f"Provided argument '{fname}' is not a valid HDF5 file.")
        print("Usage: h5tui <file>.h5")
        return False

    return True


def h5tui() -> None:
    """Main function to run the H5TUI application."""
    parser = argparse.ArgumentParser(description="A TUI for viewing HDF5 files.")
    parser.add_argument("file", type=str, help="Path to the HDF5 file.")
    args = parser.parse_args()
    h5file = args.file
    if check_file_validity(h5file):
        app = H5TUIApp(h5file)
        app.run()


if __name__ == "__main__":
    h5tui()
