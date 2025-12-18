import os

import numpy as np
import pandas as pd

import plotly.express as px
import streamlit as st


st.set_page_config(layout="wide")
st.sidebar.title("SCOS-BP")
tab_df, tab_plot = st.tabs(["dataframe", "plot"], default="plot")


with tab_df:
    # upload file
    uploader = st.file_uploader(
        " ", type=["csv", "parquet"], label_visibility="collapsed",
    )
    if uploader is not None: file_load_path = uploader.name
    else: file_load_path = "data/evaluation/pretrain-t/profile.csv.parquet"
    # load file
    ext = os.path.splitext(file_load_path)[1]
    reader = {
        ".csv": pd.read_csv,
        ".parquet": pd.read_parquet,
    }.get(ext.lower())
    if reader is None: st.stop()
    df = reader(file_load_path)
    # preview
    st.data_editor(df)
    # process
    df["split"] = df["split"].map({0: "train", 1: "test", 2: "test"})
    df["_filter"] = True
    float_cols = df.select_dtypes(include=["float"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)


st.sidebar.header("filter")
with st.sidebar.expander("subject", expanded=False):
    option = {
        "subject": df["subject"].unique().tolist(),                 # str
        "group": df["group"].unique().tolist(),                     # str
        "health": df["health"].unique().tolist(),                   # bool
        "system": df["system"].unique().tolist(),                   # bool
        "age": sorted(df["age"].unique().tolist()),                 # int
    }
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "select all", key="subject_select_all", 
            use_container_width=True
        ): st.session_state["filter_subject"] = option["subject"]
    with col2:
        if st.button(
            "clear", key="subject_clear", 
            use_container_width=True
        ): st.session_state["filter_subject"] = []
    filter_subject = st.multiselect(
        "subject", options=option["subject"], default=option["subject"],
        key="filter_subject",
    )
    filter_group = st.pills(
        "group", options=option["group"], default=option["group"],
        selection_mode="multi",
    )
    plot_facet_x = st.pills(
        "health", options=option["health"], default=option["health"],
        selection_mode="multi",
    )
    filter_system = st.pills(
        "system", options=option["system"], default=option["system"], 
        selection_mode="multi",
    )
    filter_age = st.select_slider(
        "age", options=option["age"],
        value=(min(option["age"]), max(option["age"])),
    )
    df.loc[~df["subject"].isin(filter_subject), "_filter"] = False
    df.loc[~df["group"].isin(filter_group), "_filter"] = False
    df.loc[~df["health"].isin(plot_facet_x), "_filter"] = False
    df.loc[~df["system"].isin(filter_system), "_filter"] = False
    df.loc[
        (df["age"] < filter_age[0]) | (df["age"] > filter_age[1]), "_filter"
    ] = False
with st.sidebar.expander("measurement", expanded=False):
    option = {
        "measurement": df["measurement"].unique().tolist(),         # str
        "repeat": df["repeat"].unique().tolist(),                   # bool
        "arm": df["arm"].unique().tolist(),                         # bool
    }
    col3, col4 = st.columns(2)
    with col3:
        if st.button(
            "select all", key="measurement_select_all", 
            use_container_width=True, 
        ): st.session_state["filter_measurement"] = option["measurement"]
    with col4:
        if st.button(
            "clear", key="measurement_clear", 
            use_container_width=True
        ): st.session_state["filter_measurement"] = []
    filter_measurement = st.multiselect(
        "measurement", options=option["measurement"], 
        default=option["measurement"],
        key="filter_measurement",
    )
    filter_repeat = st.pills(
        "repeat", options=option["repeat"], default=option["repeat"],
        selection_mode="multi",
    )
    filter_arm = st.pills(
        "arm", options=option["arm"], default=option["arm"],
        selection_mode="multi",
    )
    df.loc[~df["measurement"].isin(filter_measurement), "_filter"] = False
    df.loc[~df["repeat"].isin(filter_repeat), "_filter"] = False
    df.loc[~df["arm"].isin(filter_arm), "_filter"] = False
with st.sidebar.expander("sample", expanded=False):
    option = {
        "condition": sorted(df["condition"].unique().tolist()),     # int
        "split": sorted(df["split"].unique().tolist()),             # int
    }
    filter_condition = st.pills(
        "condition", options=option["condition"], default=option["condition"],
        selection_mode="multi",
    )
    filter_split = st.pills(
        "split", options=option["split"], default=option["split"],
        selection_mode="multi",
    )
    df.loc[~df["condition"].isin(filter_condition), "_filter"] = False
    df.loc[~df["split"].isin(filter_split), "_filter"] = False


df_true = df[df["_filter"]]


st.sidebar.header("plotly")
with st.sidebar.expander("panel", expanded=False):
    plot_width = st.number_input(
        "width", min_value=200, value=1100, step=50
    )
    plot_height = st.number_input(
        "height", min_value=200, value=650, step=50
    )
    plot_facet_x = st.pills(
        "facetting x",
        options=[
            c for c in df_true.select_dtypes(exclude=["float"]).columns
            if df_true[c].nunique(dropna=True) <= 6
            if not c.startswith("_")
        ],
        default="split",
        selection_mode="single",
    )
    plot_facet_y = st.pills(
        "facetting y",
        options=[
            c for c in df_true.select_dtypes(exclude=["float"]).columns
            if df_true[c].nunique(dropna=True) <= 6
            if not c.startswith("_")
        ],
        selection_mode="single",
    )
    plot_marginal_x = st.pills(
        "marginal x", options=['box', 'histogram', 'rug', 'violin'],
        default='histogram',
        selection_mode="single",
    )
    plot_marginal_y = st.pills(
        "marginal y", options=['box', 'histogram', 'rug', 'violin'],
        default='histogram',
        selection_mode="single",
    )
with st.sidebar.expander("axis", expanded=False):
    selector_x = st.pills(
        "x", key="selector_x", options=float_cols, default="umap1",
        selection_mode="single", 
    )
    selector_y = st.pills(
        "y", key="selector_y", options=float_cols, default="umap2",
        selection_mode="single", 
    )
with st.sidebar.expander("appearance", expanded=False):
    plot_size = st.slider(
        "size", min_value=0.5, max_value=10.0, value=2.5, step=0.50
    )
    plot_opacity = st.slider(
        "opacity", min_value=0.1, max_value=1.0, value=0.6, step=0.05
    )
    plot_color = st.pills(
        "color",
        options=[
            c for c in df_true.columns
            if not c.startswith("_")
        ],
        default="measurement",
        selection_mode="single",
    )
    plot_color_map_discrete = st.pills(
        "color map discrete",
        options=[
            "Plotly", "D3", "G10", "T10", "Alphabet", "Dark24", "Light24",
            "Set1", "Pastel1", "Dark2", "Set2", "Pastel2", "Set3",
            "Antique", "Bold", "Pastel", "Prism", "Safe", "Vivid",
        ],
        default="Plotly",
        selection_mode="single",
    )
    plot_color_scale_continuous = st.pills(
        "color scale continuous",
        options=[
            "Plotly3",
            "Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Turbo",
            "Rainbow", "Jet", "Hot", "Blackbody",
            "Electric", "YlGnBu", "YlOrRd", "RdBu", "Electric", "Bluered",
            "YlGn", "Greens", "Greys", "Blues", "Reds",
        ],
        default="RdBu",
        selection_mode="single",
    )



with tab_plot:
    fig = px.scatter(
        df_true,
        x=selector_x,
        y=selector_y,
        color=plot_color,
        color_discrete_sequence=\
            px.colors.qualitative.__getattribute__(     # type: ignore
                plot_color_map_discrete
            ),
        color_continuous_scale=\
            px.colors.sequential.__getattribute__(      # type: ignore
                plot_color_scale_continuous
            ),
        facet_col=plot_facet_x,
        facet_row=plot_facet_y,
        marginal_x=plot_marginal_x,
        marginal_y=plot_marginal_y,
        opacity=plot_opacity,
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.for_each_trace(
        lambda t: t.update(marker=dict(size=plot_size))
        if t.type in ("scatter", "scattergl") else None
    )
    st.plotly_chart(fig, height=plot_height, width=plot_width)
