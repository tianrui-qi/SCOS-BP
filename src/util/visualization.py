import torch
import numpy as np
import pandas as pd

import plotly
import plotly.express
import plotly.subplots
import plotly.graph_objects
import ipywidgets
import IPython.display


class VisualizationRepresentation:
    """
    Interactive embedding viewer (per stage) with waveform preview.

    - Left: 2D embedding scatter (UMAP / top PCs for a given stage)
    - Right top: BP waveform (data.y)
    - Right bottom: 3 optical channels (data.x[:, 0:3])

    Filters:
      - health / system / repeat / split (train/test)
      - subject / condition
    Plot:
      - All samples are ALWAYS drawn (gray, low opacity)
      - Samples matching filters + subject/condition selection are 
        HIGHLIGHTED with chosen color-by scheme.
    """

    def __init__(
        self,
        x: torch.Tensor,        # (N, 3, T)
        y: torch.Tensor,        # (N, T)
        # includes S1_*/S2_* coordinates, diastole/systole/T, etc.
        profile: pd.DataFrame,  
        stage: str = "S1",      # "S1" or "S2"
    ):
        assert stage in ["S1", "S2"], "stage must be 'S1' or 'S2'"
        self.stage = stage

        # ---- move tensors to numpy ----
        self.x = x  # (N, 3, T)
        self.y = y  # (N, T)

        N = self.x.shape[0]
        assert self.y.shape[0] == N, "x, y must have same N"
        assert len(profile) == N, "profile length must match N"

        # ---- work on a copy of profile ----
        df = profile.reset_index(drop=True).copy()

        # ensure T is int (sample index within subject)
        if "T" in df.columns:
            df["T"] = df["T"].astype(int)
        else:
            df["T"] = df.groupby("subject").cumcount().astype(int)

        # patient: strip "_R"
        df["patient"] = df["subject"].str.replace("_R", "", regex=False)

        # split group: 0 -> train, 1/2 -> test
        df["_split_group"] = np.where(df["split"] == 0, "train", "test")

        # system: map to "old"/"new"
        df["_system_label"] = df["system"].map(
            {False: "old", True: "new", 0: "old", 1: "new"}
        ).astype(str)

        # health / repeat: map to "false"/"true"
        df["_health_label"] = (
            df["health"]
            .map({False: "false", True: "true", 0: "false", 1: "true"})
            .astype(str)
        )
        df["_repeat_label"] = (
            df["repeat"]
            .map({False: "false", True: "true", 0: "false", 1: "true"})
            .astype(str)
        )

        # global sample index for customdata
        df["sample_idx"] = df.index.astype(int)

        self.df = df

        # ---- axis mapping for this stage ----
        axis_names = ["UMAP1", "UMAP2"] + [f"PC{i}" for i in range(1, 7)]
        self.axis_col_map = {name: f"{stage}_{name}" for name in axis_names}

        missing_axes = [
            col 
            for col in self.axis_col_map.values() 
            if col not in df.columns
        ]
        if missing_axes:
            raise ValueError(
                f"Missing columns for stage '{stage}': {missing_axes}"
            )

        # precompute global axis ranges
        self.axis_ranges = {}
        for name, col in self.axis_col_map.items():
            vals = df[col].to_numpy()
            vmin, vmax = vals.min(), vals.max()
            span = vmax - vmin
            pad = 0.005 * (span if span > 0 else 1.0)
            self.axis_ranges[name] = [vmin - pad, vmax + pad]

        # waveform ranges (BP & optical)
        (
            self.wave_range_bp,
            self.wave_range_opt,
            self.N_T,
            self.GLOBAL_T,
        ) = self._compute_waveform_ranges(self.x, self.y)   # type: ignore

        # plot style
        self.color_seq = plotly.express.colors.qualitative.Plotly
        self.symbol = ["circle"]

        # columns available for color-by
        self.base_color_cols = [
            "subject", "patient", "health", "system", "repeat",
            "condition", "split", "diastole", "systole", "T",
        ]

        self._click_handler_refs = []

        # build UI
        self.ui = self._build_dashboard()
        IPython.display.display(self.ui)

    # ============================================================
    # dashboard
    # ============================================================
    def _build_dashboard(self):
        controls = self._create_controls()
        out = ipywidgets.Output()
        suppress_update = False

        def update_plot(*_):
            nonlocal suppress_update
            if suppress_update:
                return

            # Only clear this Output widget
            out.clear_output(wait=True)
            with out:
                df_all = self.df

                # ===================================================
                # 1) Boolean-like filters: health / system / repeat / split
                # ===================================================
                df_bool = df_all

                health_f = controls["health_filter"].value  
                # "all", "false", "true"
                if health_f != "all":
                    df_bool = df_bool[df_bool["_health_label"] == health_f]

                system_f = controls["system_filter"].value  
                # "all", "old", "new"
                if system_f != "all":
                    df_bool = df_bool[df_bool["_system_label"] == system_f]

                repeat_f = controls["repeat_filter"].value  
                # "all", "false", "true"
                if repeat_f != "all":
                    df_bool = df_bool[df_bool["_repeat_label"] == repeat_f]

                split_f = controls["split_filter"].value  
                # "all", "train", "test"
                if split_f != "all":
                    df_bool = df_bool[df_bool["_split_group"] == split_f]

                # ===================================================
                # 2) Update subject/condition dropdowns based on df_bool
                # ===================================================
                suppress_update = True
                try:
                    # subject
                    new_subjects = sorted(df_bool["subject"].unique())
                    subj_widget = controls["subject_select"]
                    new_subj_options = [(s, s) for s in new_subjects]

                    old_subj_sel = list(subj_widget.value)
                    new_subj_sel = [
                        s for s in old_subj_sel if s in new_subjects
                    ]
                    if not new_subj_sel:
                        new_subj_sel = new_subjects

                    subj_widget.options = new_subj_options
                    subj_widget.value = tuple(new_subj_sel)

                    # condition
                    new_conds = sorted(df_bool["condition"].unique())
                    cond_widget = controls["condition_select"]
                    new_cond_options = [
                        (int(c), int(c)) for c in new_conds
                    ]

                    old_cond_sel = list(cond_widget.value)
                    new_cond_sel = [
                        c for c in old_cond_sel if c in new_conds
                    ]
                    if not new_cond_sel:
                        new_cond_sel = new_conds

                    cond_widget.options = new_cond_options
                    cond_widget.value = tuple(new_cond_sel)
                finally:
                    suppress_update = False

                # ===================================================
                # 3) Subject/condition selections further refine the 
                # highlighted set
                # ===================================================
                df_high = df_bool
                subj_sel = list(controls["subject_select"].value)
                if subj_sel:
                    df_high = df_high[df_high["subject"].isin(subj_sel)]

                cond_sel = list(controls["condition_select"].value)
                if cond_sel:
                    df_high = df_high[df_high["condition"].isin(cond_sel)]

                if df_high.empty:
                    mask_highlight = np.zeros(len(df_all), dtype=bool)
                else:
                    mask_highlight = df_all.index.isin(df_high.index)

                # ===================================================
                # 4) 
                # ===================================================
                x_axis_name = controls["x_axis_dropdown"].value
                y_axis_name = controls["y_axis_dropdown"].value
                x_col = self.axis_col_map[x_axis_name]
                y_col = self.axis_col_map[y_axis_name]

                x_vals_all = df_all[x_col].to_numpy()
                y_vals_all = df_all[y_col].to_numpy()

                # ===================================================
                # 5) customdata + hover
                # ===================================================
                sample_idx_arr = df_all["sample_idx"].to_numpy().astype(int)
                custom_columns = [sample_idx_arr]
                hover_lines = ["sample_idx = %{customdata[0]}"]

                meta_cols_for_hover = [
                    "subject", "patient", "condition",
                    "T", "diastole", "systole",
                    "health", "system", "repeat", "_split_group",
                ]
                col_pos = 1
                for col in meta_cols_for_hover:
                    vals = df_all[col].to_numpy()
                    custom_columns.append(vals)
                    hover_lines.append(
                        f"{col} = %{{customdata[{col_pos}]}}"
                    )
                    col_pos += 1

                hover_template = \
                    "<br>".join(hover_lines) + "<extra></extra>"
                customdata_all = np.column_stack(custom_columns)

                # ===================================================
                # 6) 
                # ===================================================
                color_by = controls["color_dropdown"].value
                point_size = controls["size_input"].value
                point_opacity = controls["opacity_input"].value

                continuous_color = False
                labels_full = None
                vals_full = None

                if color_by == "subject":
                    labels_full = df_all["subject"].astype(str)
                elif color_by == "patient":
                    labels_full = df_all["patient"].astype(str)
                elif color_by == "health":
                    labels_full = df_all["_health_label"]
                elif color_by == "system":
                    labels_full = df_all["_system_label"]
                elif color_by == "repeat":
                    labels_full = df_all["_repeat_label"]
                elif color_by == "condition":
                    labels_full = df_all["condition"].astype(str)
                elif color_by == "split":
                    labels_full = df_all["_split_group"].astype(str)
                elif color_by in ["diastole", "systole", "T"]:
                    continuous_color = True
                    vals = df_all[color_by].to_numpy().astype(float)
                    if color_by == "T":
                        # Normalize T to [0,1] within each subject
                        subj = df_all["subject"].to_numpy()
                        T_max = df_all.groupby(
                            "subject"
                        )["T"].transform("max").to_numpy()
                        T_max[T_max <= 0] = 1.0
                        vals = vals / T_max
                    vals_full = vals
                else:
                    labels_full = pd.Series(
                        ["all"] * len(df_all), index=df_all.index
                    )
                    color_by = "all"

                idx_high = np.where(mask_highlight)[0]
                labels_high = None
                vals_high = None
                if idx_high.size > 0:
                    if continuous_color and (vals_full is not None):
                        vals_high = vals_full[idx_high]
                    elif (not continuous_color) and \
                    (labels_full is not None):
                        labels_high = labels_full.iloc[idx_high]

                # ===================================================
                # ===================================================
                base_fig = plotly.subplots.make_subplots(
                    rows=2, cols=2,
                    specs=[
                        [{"rowspan": 2}, {"type": "xy"}],
                        [None,          {"type": "xy"}],
                    ],
                    column_widths=[0.5, 0.5],
                    row_heights=[0.5, 0.5],
                    shared_xaxes=False,
                    horizontal_spacing=0.02,
                    vertical_spacing=0.06,
                )
                fig = plotly.graph_objects.FigureWidget(base_fig)

                #  Left: gray background (all samples) ----------------
                fig.add_trace(
                    plotly.graph_objects.Scatter(
                        x=x_vals_all,
                        y=y_vals_all,
                        mode="markers",
                        marker=dict(
                            color="lightgray",
                            size=point_size,
                            opacity=0.15,
                        ),
                        customdata=customdata_all,
                        hovertemplate=hover_template,
                        showlegend=False,
                    ),
                    row=1, col=1,
                )

                #  Left: highlighted set ----------------
                if idx_high.size > 0:
                    x_high = x_vals_all[idx_high]
                    y_high = y_vals_all[idx_high]
                    custom_high = customdata_all[idx_high]

                    if continuous_color and (vals_high is not None):
                        # Continuous colors (no colorbar)
                        fig.add_trace(
                            plotly.graph_objects.Scatter(
                                x=x_high,
                                y=y_high,
                                mode="markers",
                                marker=dict(
                                    color=vals_high.tolist(),
                                    colorscale="Viridis",
                                    showscale=False,
                                    size=point_size,
                                    opacity=point_opacity,
                                ),
                                customdata=custom_high,
                                hovertemplate=hover_template,
                                showlegend=False,
                            ),
                            row=1, col=1,
                        )
                    elif (not continuous_color) and \
                    (labels_high is not None):
                        labels_high = labels_high.astype(str)
                        color_vals = sorted(labels_high.unique())
                        color_map = {
                            v: self.color_seq[i % len(self.color_seq)]
                            for i, v in enumerate(color_vals)
                        }
                        for v in color_vals:
                            m = (labels_high == v).to_numpy()
                            if not m.any():
                                continue
                            fig.add_trace(
                                plotly.graph_objects.Scatter(
                                    x=x_high[m],
                                    y=y_high[m],
                                    mode="markers",
                                    marker=dict(
                                        color=color_map[v],
                                        symbol=self.symbol[0],
                                        size=point_size,
                                        opacity=point_opacity,
                                    ),
                                    customdata=custom_high[m],
                                    hovertemplate=hover_template,
                                    showlegend=False,
                                ),
                                row=1, col=1,
                            )

                # Fix embedding axis ranges and keep equal aspect
                x_range = self.axis_ranges[x_axis_name]
                y_range = self.axis_ranges[y_axis_name]
                combined_min = min(x_range[0], y_range[0])
                combined_max = max(x_range[1], y_range[1])
                center = 0.5 * (combined_min + combined_max)
                half_span = 0.5 * max(combined_max - combined_min, 1e-6)
                equal_range = [center - half_span, center + half_span]

                fig.update_xaxes(
                    title_text=f"{self.stage} {x_axis_name}",
                    range=equal_range,
                    row=1, col=1,
                )
                fig.update_yaxes(
                    title_text=f"{self.stage} {y_axis_name}",
                    range=equal_range,
                    scaleanchor="x",
                    scaleratio=1,
                    row=1, col=1,
                )

                # Right: placeholder waveform traces ----------------
                t = self.GLOBAL_T

                # Top: BP
                idx_bp = len(fig.data)
                fig.add_trace(
                    plotly.graph_objects.Scatter(
                        x=t,
                        y=np.full(self.N_T, np.nan),
                        mode="lines",
                        name="BP",
                        line=dict(color="#1f77b4", width=2),
                        showlegend=False,
                    ),
                    row=1, col=2,
                )

                # Bottom: 3 optical channels
                idx_ch0 = len(fig.data)
                opt_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
                for ch in range(3):
                    fig.add_trace(
                        plotly.graph_objects.Scatter(
                            x=t,
                            y=np.full(self.N_T, np.nan),
                            mode="lines",
                            name=f"ch{ch}",
                            line=dict(color=opt_colors[ch], width=2),
                            showlegend=False,
                        ),
                        row=2, col=2,
                    )

                wave_trace_indices = {
                    "bp": idx_bp,
                    "ch0": idx_ch0,
                    "ch1": idx_ch0 + 1,
                    "ch2": idx_ch0 + 2,
                }

                fig.update_yaxes(row=1, col=2, range=self.wave_range_bp)
                fig.update_yaxes(row=2, col=2, range=self.wave_range_opt)

                fig.update_layout(
                    height=608,
                    width=1400,
                    font=dict(size=12),
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                )

                # click handler: update waveforms
                def handle_click(trace, points, _):
                    if not points.point_inds:
                        return
                    idx0 = points.point_inds[0]
                    cd_row = trace.customdata[idx0]
                    s_idx = int(cd_row[0])   # customdata[0] = sample_idx

                    if s_idx < 0 or s_idx >= self.x.shape[0]:
                        return

                    bp_wave = self.y[s_idx]
                    opt_waves = self.x[s_idx, 0:3, :]

                    with fig.batch_update():
                        fig.data[wave_trace_indices["bp"]].y  = bp_wave
                        fig.data[
                            wave_trace_indices["ch0"]].y = opt_waves[0]
                        fig.data[
                            wave_trace_indices["ch1"]].y = opt_waves[1]
                        fig.data[
                            wave_trace_indices["ch2"]].y = opt_waves[2]

                # Keep handler reference to avoid garbage collection
                self._click_handler_refs.append(handle_click)

                # All scatter plots (gray + highlighted) support clicks
                for tr in fig.data:
                    if (
                        getattr(tr, "mode", None) == "markers" and
                        getattr(tr, "customdata", None) is not None
                    ):
                        tr.on_click(handle_click)

                IPython.display.display(fig)

        # Bind controls
        for w in controls["widgets_list"]:
            w.observe(update_plot, names="value")

        ui = ipywidgets.VBox([controls["top_controls"], out])
        update_plot()
        return ui

    # ============================================================
    # waveform ranges
    # ============================================================
    @staticmethod
    def _compute_waveform_ranges(
        x_np: np.ndarray,  # (N, 3, T)
        y_np: np.ndarray,  # (N, T)
    ):
        # BP range
        vmin_bp = float(y_np.min())
        vmax_bp = float(y_np.max())

        # optical range
        vmin_opt = float(x_np.min())
        vmax_opt = float(x_np.max())

        pad_bp = 0.05 * (vmax_bp - vmin_bp + 1e-8)
        pad_opt = 0.05 * (vmax_opt - vmin_opt + 1e-8)

        wave_range_bp  = [vmin_bp  - pad_bp,  vmax_bp  + pad_bp]
        wave_range_opt = [vmin_opt - pad_opt, vmax_opt + pad_opt]

        N_T = x_np.shape[2]
        GLOBAL_T = np.arange(N_T)

        return wave_range_bp, wave_range_opt, N_T, GLOBAL_T

    # ============================================================
    # controls
    # ============================================================
    def _create_controls(self) -> dict:
        UNIFIED_WIDTH = "220px"
        CONTROL_STYLE = {"description_width": "60px"}

        # axis dropdowns
        axis_options = \
            ["UMAP1", "UMAP2"] + [f"PC{i}" for i in range(1, 7)]

        x_axis_dropdown = ipywidgets.Dropdown(
            options=axis_options,
            value="UMAP1",
            description="x",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH),
            style=CONTROL_STYLE,
        )
        y_axis_dropdown = ipywidgets.Dropdown(
            options=axis_options,
            value="UMAP2",
            description="y",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH),
            style=CONTROL_STYLE,
        )

        # color dropdown
        color_dropdown = ipywidgets.Dropdown(
            options=self.base_color_cols,
            value="condition",
            description="color",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH),
            style=CONTROL_STYLE,
        )

        # boolean-like filters
        health_filter = ipywidgets.Dropdown(
            options=[
                ("all", "all"), ("false", "false"), ("true", "true")
            ],
            value="all",
            description="health",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH),
            style=CONTROL_STYLE,
        )
        system_filter = ipywidgets.Dropdown(
            options=[("all", "all"), ("old", "old"), ("new", "new")],
            value="all",
            description="system",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH),
            style=CONTROL_STYLE,
        )
        repeat_filter = ipywidgets.Dropdown(
            options=[
                ("all", "all"), 
                ("false", "false"), 
                ("true", "true")
            ],
            value="all",
            description="repeat",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH),
            style=CONTROL_STYLE,
        )
        split_filter = ipywidgets.Dropdown(
            options=[
                ("all", "all"), ("train", "train"), ("test", "test")
            ],
            value="all",
            description="split",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH),
            style=CONTROL_STYLE,
        )

        # point size / opacity
        size_input = ipywidgets.BoundedIntText(
            value=5, min=1, max=15, step=1,
            description="size",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH),
            style=CONTROL_STYLE,
        )
        opacity_input = ipywidgets.BoundedFloatText(
            value=0.7, min=0.1, max=1.0, step=0.05,
            description="opacity",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH),
            style=CONTROL_STYLE,
        )

        # subject/condition multi-select (all selected initially)
        subjects = sorted(self.df["subject"].unique())
        conds = sorted(self.df["condition"].unique())

        subject_select = ipywidgets.SelectMultiple(
            options=[(s, s) for s in subjects],
            value=tuple(subjects),
            description="subject",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH, height="130px"),
            style=CONTROL_STYLE,
        )
        condition_select = ipywidgets.SelectMultiple(
            options=[(int(c), int(c)) for c in conds],
            value=tuple(conds),
            description="cond",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH, height="130px"),
            style=CONTROL_STYLE,
        )

        top_controls = ipywidgets.HBox([
            ipywidgets.VBox([x_axis_dropdown, y_axis_dropdown]),
            ipywidgets.VBox([color_dropdown, size_input, opacity_input]),
            ipywidgets.VBox([
                health_filter, system_filter, 
                repeat_filter, split_filter
            ]),
            subject_select,
            condition_select,
        ])

        widgets_list = [
            x_axis_dropdown, y_axis_dropdown,
            color_dropdown,
            health_filter, system_filter, 
            repeat_filter, split_filter,
            size_input, opacity_input,
            subject_select, condition_select,
        ]

        return {
            "x_axis_dropdown": x_axis_dropdown,
            "y_axis_dropdown": y_axis_dropdown,
            "color_dropdown": color_dropdown,
            "health_filter": health_filter,
            "system_filter": system_filter,
            "repeat_filter": repeat_filter,
            "split_filter": split_filter,
            "size_input": size_input,
            "opacity_input": opacity_input,
            "subject_select": subject_select,
            "condition_select": condition_select,
            "widgets_list": widgets_list,
            "top_controls": top_controls,
        }


class VisualizationPrediction():
    def __init__(self, result: torch.Tensor, profile: pd.DataFrame):
        # data
        self.result = result.detach().cpu().numpy()
        self.profile = profile
        # plot settings
        self.color = plotly.express.colors.qualitative.Plotly
        self.symbol = [
            "circle", "square", "diamond", "x", "cross", "triangle-up"
        ]
        # columns used for filtering / hover
        self.filter_cols = [
            "subject", "health", "system", "repeat", "condition", "split"
        ]
        self.hover_cols  = [
            "subject", "health", "system", "repeat", "condition", "split"
        ]
        self.multi_filter_cols = ["subject", "condition"]
        self.bool_like_cols    = ["health", "system", "repeat", "split"]
        # build dashboard
        self.ui = self._build_dashboard()
        IPython.display.display(self.ui)

    def _build_dashboard(self):
        wave_range_row1, wave_range_row2, N_T, GLOBAL_T = (
            self._compute_waveform_ranges(self.result)
        )

        numeric_cols = self.profile.select_dtypes(
            include=["float64", "float32"]
        ).columns.tolist()
        controls = self._create_controls(self.profile, numeric_cols)
        plot_output = ipywidgets.Output()

        suppress_update = False
        wave_trace_indices = {}
        sample_idx_pos = None

        # ----------------------------
        # inner callback
        # ----------------------------
        def update_plot(*args):
            nonlocal suppress_update, sample_idx_pos, wave_trace_indices
            if suppress_update:
                return

            with plot_output:
                IPython.display.clear_output(wait=True)

                # --- 1. boolean filters ---
                df_bool = self.profile
                for col, w in controls["bool_filter_radios"].items():
                    val = w.value
                    if val == "__ALL__":
                        continue
                    df_bool = df_bool[df_bool[col] == val]
                if df_bool.empty:
                    print("No samples match the current filters.")
                    return

                # --- 2. update multi filters ---
                suppress_update = True
                try:
                    for col, w in controls["multi_filter_widgets"].items():
                        new_vals = sorted(df_bool[col].unique())
                        new_options = [(str(v), v) for v in new_vals]

                        old_selected = list(w.value)
                        new_selected = tuple(
                            v for v in old_selected if v in new_vals
                        )
                        if len(new_selected) == 0:
                            new_selected = tuple(new_vals)

                        w.options = new_options
                        w.value = new_selected
                finally:
                    suppress_update = False

                # --- 3. multi filters ---
                sub_df = df_bool
                for col, w in controls["multi_filter_widgets"].items():
                    selected = list(w.value)
                    if selected:
                        sub_df = sub_df[sub_df[col].isin(selected)]
                if sub_df.empty:
                    print("No samples match current filters.")
                    return

                sub_df = sub_df.copy()
                sub_df["sample_idx"] = sub_df.index

                # --- prepare scatter data ---
                x1 = controls["x1_dropdown"].value
                y1 = controls["y1_dropdown"].value
                x2 = controls["x2_dropdown"].value
                y2 = controls["y2_dropdown"].value
                has_x2 = (x2 is not None) and (y2 is not None)

                color_col = controls["color_dropdown"].value
                point_size = controls["size_input"].value
                point_opacity = controls["opacity_input"].value

                df_list = []

                df1 = sub_df.copy()
                df1["_x"] = df1[x1]
                df1["_y"] = df1[y1]
                df1["_pair"] = "pair1"
                df_list.append(df1)

                if has_x2:
                    df2 = sub_df.copy()
                    df2["_x"] = df2[x2]
                    df2["_y"] = df2[y2]
                    df2["_pair"] = "pair2"
                    df_list.append(df2)

                df_long = pd.concat(df_list, ignore_index=True)

                # color
                if color_col:
                    if color_col in ["health", "repeat"]:
                        df_long["_color_label"] = (
                            df_long[color_col]
                            .map({
                                False: "false", True: "true",
                                "False": "false", "True": "true",
                                0: "false", 1: "true"
                            })
                            .fillna(df_long[color_col])
                            .astype(str)
                        )
                    else:
                        df_long["_color_label"] = \
                            df_long[color_col].astype(str)
                    color_vals = sorted(df_long["_color_label"].unique())
                else:
                    df_long["_color_label"] = "all"
                    color_vals = ["all"]

                color_map = {
                    v: self.color[i % len(self.color)]
                    for i, v in enumerate(color_vals)
                }

                pair_vals = sorted(df_long["_pair"].unique())
                symbol_map = {
                    p: self.symbol[i] for i, p in enumerate(pair_vals)
                }
                pair_label_map = {"pair1": "x1 vs y1", "pair2": "x2 vs y2"}

                # ---- build customdata for hover ----
                sample_idx_series = df_long["sample_idx"].astype(int)
                sample_idx_arr = sample_idx_series.to_numpy()

                sub_indexed = sub_df.set_index("sample_idx")
                x1_for_row = sub_indexed.loc[sample_idx_arr, x1].to_numpy()
                y1_for_row = sub_indexed.loc[sample_idx_arr, y1].to_numpy()

                custom_columns = [x1_for_row, y1_for_row]
                lines = ["x1 = %{customdata[0]}", "y1 = %{customdata[1]}"]
                next_idx = 2

                if has_x2:
                    x2_for_row = sub_indexed.loc[sample_idx_arr, x2].to_numpy()
                    y2_for_row = sub_indexed.loc[sample_idx_arr, y2].to_numpy()
                    custom_columns.append(x2_for_row)
                    custom_columns.append(y2_for_row)
                    lines.append(f"x2 = %{{customdata[{next_idx}]}}")
                    lines.append(f"y2 = %{{customdata[{next_idx+1}]}}")
                    next_idx += 2

                sample_idx_pos = next_idx
                custom_columns.append(sample_idx_arr)
                lines.append(f"s = %{{customdata[{next_idx}]}}")
                next_idx += 1

                for col in self.hover_cols:
                    if col in ["health", "repeat"]:
                        vals = (
                            df_long[col]
                            .map({
                                False: "false", True: "true",
                                "False": "false", "True": "true",
                                0: "false", 1: "true"
                            })
                            .fillna(df_long[col])
                            .astype(str)
                            .to_numpy()
                        )
                    else:
                        vals = df_long[col].astype(str).to_numpy()
                    custom_columns.append(vals)
                    lines.append(f"{col} = %{{customdata[{next_idx}]}}")
                    next_idx += 1

                hover_template = "<br>".join(lines) + "<extra></extra>"
                customdata = np.column_stack(custom_columns)

                # ---- subplot layout ----
                base_fig = plotly.subplots.make_subplots(
                    rows=2, cols=2,
                    specs=[
                        [{"rowspan": 2}, {"type": "xy"}],
                        [None, {"type": "xy"}]
                    ],
                    column_widths=[0.5, 0.5],
                    row_heights=[0.5, 0.5],
                    shared_xaxes=True,
                    horizontal_spacing=0.035,
                    vertical_spacing=0.035,
                )
                fig = plotly.graph_objects.FigureWidget(base_fig)

                # ---- scatter traces ----
                if color_col:
                    for c in color_vals:
                        for p in pair_vals:
                            mask = (
                                (df_long["_color_label"] == c)
                                & (df_long["_pair"] == p)
                            )
                            if not mask.any():
                                continue
                            sub = df_long[mask]
                            fig.add_trace(
                                plotly.graph_objects.Scatter(
                                    x=sub["_x"],
                                    y=sub["_y"],
                                    mode="markers",
                                    marker=dict(
                                        color=color_map[c],
                                        symbol=symbol_map.get(
                                            p, self.symbol[0]
                                        ),
                                        size=point_size,
                                        opacity=point_opacity,
                                    ),
                                    customdata=customdata[mask.to_numpy()],
                                    hovertemplate=hover_template,
                                    showlegend=False,
                                ), row=1, col=1
                            )
                else:
                    for p in pair_vals:
                        mask = df_long["_pair"] == p
                        sub = df_long[mask]
                        fig.add_trace(
                            plotly.graph_objects.Scatter(
                                x=sub["_x"], y=sub["_y"],
                                mode="markers",
                                marker=dict(
                                    color="blue",
                                    symbol=symbol_map.get(
                                        p, self.symbol[0]
                                    ),
                                    size=point_size,
                                    opacity=point_opacity,
                                ),
                                customdata=customdata[mask.to_numpy()],
                                hovertemplate=hover_template,
                                showlegend=False,
                            ), row=1, col=1
                        )

                # ---- waveform placeholders ----
                t = GLOBAL_T
                color_bp_true = "#1f77b4"
                color_bp_pred = "#ff7f0e"
                empty_wave_custom = np.zeros((N_T, 3), dtype=float)

                start_idx = len(fig.data)
                idx_bp_before_true = start_idx
                fig.add_trace(
                    plotly.graph_objects.Scatter(
                        x=t,
                        y=np.full(N_T, np.nan),
                        mode="lines",
                        name="BP True",
                        line=dict(color=color_bp_true, width=2),
                        legendgroup="waveform",
                        legendgrouptitle_text="waveform",
                        showlegend=True,
                        customdata=empty_wave_custom,
                        hovertemplate=(
                            "T = %{x}<br>"
                            "BP True = %{customdata[0]:.2f}<br>"
                            "BP Pred = %{customdata[1]:.2f}<br>"
                            "|Pred-True| = %{customdata[2]:.2f}<extra></extra>"
                        ),
                    ), row=1, col=2
                )
                idx_bp_before_pred = start_idx + 1
                fig.add_trace(
                    plotly.graph_objects.Scatter(
                        x=t,
                        y=np.full(N_T, np.nan),
                        mode="lines",
                        name="BP Pred",
                        line=dict(color=color_bp_pred, width=2),
                        legendgroup="waveform",
                        showlegend=True,
                        customdata=empty_wave_custom,
                        hovertemplate=(
                            "T = %{x}<br>"
                            "BP True = %{customdata[0]:.2f}<br>"
                            "BP Pred = %{customdata[1]:.2f}<br>"
                            "|Pred-True| = %{customdata[2]:.2f}<extra></extra>"
                        ),
                    ), row=1, col=2
                )

                # ch0–2 after normalization
                idx_ch0 = len(fig.data)
                for ch in range(3):
                    fig.add_trace(
                        plotly.graph_objects.Scatter(
                            x=t,
                            y=np.full(N_T, np.nan),
                            mode="lines",
                            name=f"ch{ch}",
                            opacity=0.5,
                            line=dict(width=2.0),
                            legendgroup="waveform",
                            showlegend=True,
                        ), row=2, col=2
                    )

                idx_bp_after_true = idx_ch0 + 3
                fig.add_trace(
                    plotly.graph_objects.Scatter(
                        x=t, y=np.full(N_T, np.nan),
                        mode="lines", name="BP True",
                        line=dict(color=color_bp_true, width=2),
                        legendgroup="waveform",
                        showlegend=False,
                    ), row=2, col=2
                )
                idx_bp_after_pred = idx_ch0 + 4
                fig.add_trace(
                    plotly.graph_objects.Scatter(
                        x=t, y=np.full(N_T, np.nan),
                        mode="lines", name="BP Pred",
                        line=dict(color=color_bp_pred, width=2),
                        legendgroup="waveform",
                        showlegend=False,
                    ), row=2, col=2
                )

                wave_trace_indices = {
                    "bp_before_true": idx_bp_before_true,
                    "bp_before_pred": idx_bp_before_pred,
                    "ch0": idx_ch0,
                    "ch1": idx_ch0 + 1,
                    "ch2": idx_ch0 + 2,
                    "bp_after_true": idx_bp_after_true,
                    "bp_after_pred": idx_bp_after_pred,
                }

                # ---- legend entries ----
                for i, p in enumerate(["pair1", "pair2"]):
                    fig.add_trace(
                        plotly.graph_objects.Scatter(
                            x=[None], y=[None],
                            mode="markers",
                            marker=dict(
                                color="black",
                                size=12,
                                symbol=self.symbol[i],
                            ),
                            name=pair_label_map[p],
                            showlegend=True,
                            legendgroup="scatter",
                            legendgrouptitle_text=\
                                "scatter" if i == 0 else None,
                        )
                    )
                for c in color_vals:
                    fig.add_trace(
                        plotly.graph_objects.Scatter(
                            x=[None], y=[None],
                            mode="markers",
                            marker=dict(
                                color=color_map[c], size=12
                            ),
                            name=f"{color_col} = {c}",
                            showlegend=True,
                            legendgroup="scatter",
                        )
                    )

                # axes
                fig.update_yaxes(row=1, col=2, range=wave_range_row1)
                fig.update_yaxes(row=2, col=2, range=wave_range_row2)
                fig.update_xaxes(showticklabels=False, row=1, col=2)
                fig.update_layout(
                    height=520, width=1200,
                    font=dict(size=12),
                    margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(orientation="v"),
                )

                # click handler for waveform update
                def handle_click(trace, points, _):
                    if not points.point_inds:
                        return
                    idx0 = points.point_inds[0]
                    cd_row = trace.customdata[idx0]
                    s_idx = int(cd_row[sample_idx_pos])
                    if (s_idx < 0) or (s_idx >= self.result.shape[0]):
                        return

                    w = self.result[s_idx]
                    if (w.ndim != 2) or (w.shape[0] < 7):
                        return

                    bp_true = w[5]
                    bp_pred = w[6]
                    bp_diff = np.abs(bp_true - bp_pred)

                    mae_max = abs(
                        float(bp_true.max()) - float(bp_pred.max())
                    )
                    mae_min = abs(
                        float(bp_true.min()) - float(bp_pred.min())
                    )

                    wave_custom = np.stack(
                        [bp_true, bp_pred, bp_diff], axis=1
                    )

                    with fig.batch_update():
                        # before
                        fig.data[
                            wave_trace_indices["bp_before_true"]
                        ].y = bp_true
                        fig.data[
                            wave_trace_indices["bp_before_pred"]
                        ].y = bp_pred
                        fig.data[
                            wave_trace_indices["bp_before_true"]
                        ].customdata = wave_custom
                        fig.data[
                            wave_trace_indices["bp_before_pred"]
                        ].customdata = wave_custom

                        # after normalized: ch0-2, BP
                        fig.data[wave_trace_indices["ch0"]].y = w[0]
                        fig.data[wave_trace_indices["ch1"]].y = w[1]
                        fig.data[wave_trace_indices["ch2"]].y = w[2]
                        fig.data[
                            wave_trace_indices["bp_after_true"]
                        ].y = w[3]
                        fig.data[
                            wave_trace_indices["bp_after_pred"]
                        ].y = w[4]

                        fig.update_layout(annotations=[dict(
                            xref="x2 domain", yref="y2 domain",
                            x=1.0, y=0.0,
                            text=(
                                "MAE of (min, max) = " + 
                                f"({mae_min:.2f}, {mae_max:.2f})"
                            ),
                            showarrow=False,
                            bgcolor="rgba(255,255,255,0.7)",
                            borderwidth=1,
                            font=dict(size=12),
                        )])

                # connect click callback to scatter traces
                for tr in fig.data: 
                    if (
                        getattr(tr, "mode", None) == "markers" and 
                        getattr(tr, "customdata", None) is not None
                    ): tr.on_click(handle_click)

                IPython.display.display(fig)

        # ---- bind update callbacks ----
        for w in controls["widgets_list"]:
            w.observe(update_plot, names="value")

        ui = ipywidgets.VBox([controls["left_col"], plot_output])
        update_plot()

        return ui

    @staticmethod
    def _compute_waveform_ranges(waveforms: np.ndarray) -> tuple[
        list[float], list[float], int, np.ndarray
    ]:
        # Channels 5-6: (BP before normalization)
        wave_min_row1 = float(waveforms[:, 5:7, :].min())
        wave_max_row1 = float(waveforms[:, 5:7, :].max())

        # Channels 0–4 after normalization
        wave_min_row2 = float(waveforms[:, 0:5, :].min())
        wave_max_row2 = float(waveforms[:, 0:5, :].max())

        pad1 = 0.05 * (wave_max_row1 - wave_min_row1 + 1e-8)
        pad2 = 0.05 * (wave_max_row2 - wave_min_row2 + 1e-8)

        wave_range_row1 = [wave_min_row1 - pad1, wave_max_row1 + pad1]
        wave_range_row2 = [wave_min_row2 - pad2, wave_max_row2 + pad2]

        N_T = waveforms.shape[2]
        GLOBAL_T = np.arange(N_T)

        return wave_range_row1, wave_range_row2, N_T, GLOBAL_T

    def _create_controls(
        self, profile: pd.DataFrame, numeric_cols: list[str]
    ) -> dict:
        """
        Create all interactive widgets.

        Returns dict of widgets and UI layout.
        """
        UNIFIED_WIDTH = "220px"
        CONTROL_STYLE = {"description_width": "70px"}

        # X/Y dropdowns
        x1_dropdown = ipywidgets.Dropdown(
            options=numeric_cols, value="PredMinBP", description="x1",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH), 
            style=CONTROL_STYLE,
        )
        y1_dropdown = ipywidgets.Dropdown(
            options=numeric_cols, value="(P-T)MinBP", description="y1",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH), 
            style=CONTROL_STYLE,
        )
        x2_dropdown = ipywidgets.Dropdown(
            options=[None] + numeric_cols, value=None, description="x2",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH), 
            style=CONTROL_STYLE,
        )
        y2_dropdown = ipywidgets.Dropdown(
            options=[None] + numeric_cols, value=None, description="y2",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH), 
            style=CONTROL_STYLE,
        )

        color_dropdown = ipywidgets.Dropdown(
            options=self.filter_cols, value="condition", 
            description="color",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH), 
            style=CONTROL_STYLE,
        )

        size_input = ipywidgets.BoundedIntText(
            value=4, min=1, max=15, step=1, 
            description="size",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH), 
            style=CONTROL_STYLE,
        )
        opacity_input = ipywidgets.BoundedFloatText(
            value=0.7, min=0.1, max=1.0, step=0.05, 
            description="opacity",
            layout=ipywidgets.Layout(width=UNIFIED_WIDTH), 
            style=CONTROL_STYLE,
        )

        # Multi-selection filters
        multi_filter_widgets = {}
        for col in self.multi_filter_cols:
            unique_vals = sorted(profile[col].unique())
            options = [(str(v), v) for v in unique_vals]

            sm = ipywidgets.SelectMultiple(
                options=options,
                value=tuple(v for _, v in options),
                description=col,
                layout=ipywidgets.Layout(
                    width=UNIFIED_WIDTH, height="125px"
                ),
                style=CONTROL_STYLE,
            )
            multi_filter_widgets[col] = sm

        bool_filter_radios = {}
        for col in self.bool_like_cols:
            unique_vals = sorted(profile[col].unique())
            options = (
                [("all", "__ALL__")] +
                [(str(v).lower(), v) for v in unique_vals]
            ) if col in ["health", "repeat"] else (
                [("all", "__ALL__")] +
                [(str(v), v) for v in unique_vals]
            )
            radio = ipywidgets.Dropdown(
                options=options,
                value="test" if col == "split" else "__ALL__",
                description=col,
                layout=ipywidgets.Layout(width=UNIFIED_WIDTH),
                style=CONTROL_STYLE,
            )
            bool_filter_radios[col] = radio

        # left column layout
        left_col = ipywidgets.HBox([
            ipywidgets.VBox([
                x1_dropdown, y1_dropdown,
                x2_dropdown, y2_dropdown,
            ]),
            ipywidgets.VBox([color_dropdown, size_input, opacity_input]),
            multi_filter_widgets["subject"],
            multi_filter_widgets["condition"],
            ipywidgets.VBox([
                bool_filter_radios["health"],
                bool_filter_radios["system"],
                bool_filter_radios["repeat"],
                bool_filter_radios["split"],
            ]),
        ])

        widgets_list = [
            x1_dropdown, y1_dropdown, x2_dropdown, y2_dropdown,
            color_dropdown, size_input, opacity_input
        ] + list(
            multi_filter_widgets.values()
        ) + list(bool_filter_radios.values())

        return {
            "x1_dropdown": x1_dropdown,
            "y1_dropdown": y1_dropdown,
            "x2_dropdown": x2_dropdown,
            "y2_dropdown": y2_dropdown,
            "color_dropdown": color_dropdown,
            "size_input": size_input,
            "opacity_input": opacity_input,
            "multi_filter_widgets": multi_filter_widgets,
            "bool_filter_radios": bool_filter_radios,
            "widgets_list": widgets_list,
            "left_col": left_col,
        }
