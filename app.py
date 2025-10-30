import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import webbrowser

df = pd.read_csv("runners_no_missing_splits_3000.csv", encoding="gbk")
split_cols = [c for c in df.columns if c.startswith("split_")]

for col in split_cols + ["finish_time"]:
    df[col] = pd.to_timedelta(df[col], errors="coerce").dt.total_seconds()
df["pace_std_sec"]    = df[split_cols].std(axis=1, skipna=True)
df["pace_std_min"]    = df["pace_std_sec"] / 60.0
df["finish_time_min"] = df["finish_time"] / 60.0

df["age_group"] = df["category"].astype(str).str.extract(r"(\d+-\d+)")
df["gender"]    = df["gender_pos"].astype(str).str.extract(r"(Male|Female)")
df["country"]   = df["representative"].astype(str).str.replace(r"\|", "", regex=True)
df["bib"] = df["name"].astype(str).str.extract(r"#(\d+)")[0].astype("Int64")

mask_time    = df["finish_time_min"].notna()
mask_gender  = mask_time & df["gender"].notna()
mask_country = mask_time & df["country"].notna()

overall_rk = df["finish_time_min"].rank(method="dense", ascending=True)
df["overall_rank"] = overall_rk.where(mask_time).round().astype("Int64")

gender_rk = (
    df.groupby("gender", dropna=False)["finish_time_min"]
      .rank(method="dense", ascending=True)
)
df["gender_rank"] = gender_rk.where(mask_gender).round().astype("Int64")

country_rk = (
    df.groupby("country", dropna=False)["finish_time_min"]
      .rank(method="dense", ascending=True)
)
df["country_rank"] = country_rk.where(mask_country).round().astype("Int64")

seg_dist = {
    "split_5km": 5.0, "split_10km": 5.0, "split_15km": 5.0, "split_20km": 5.0, "split_21.1km": 1.1,
    "split_25km": 3.9, "split_30km": 5.0, "split_35km": 5.0, "split_40km": 5.0, "split_42.195km": 2.195,
}
first_half_cols  = ["split_5km","split_10km","split_15km","split_20km","split_21.1km"]
second_half_cols = ["split_25km","split_30km","split_35km","split_40km","split_42.195km"]

FIRST_HALF_KM  = sum(seg_dist[c] for c in first_half_cols)
SECOND_HALF_KM = sum(seg_dist[c] for c in second_half_cols)
FULL_KM        = FIRST_HALF_KM + SECOND_HALF_KM

df["first_half_sec"]  = df[first_half_cols].sum(axis=1, skipna=True)
df["second_half_sec"] = df[second_half_cols].sum(axis=1, skipna=True)

seg_pace_cols = []
for c in split_cols:
    km = seg_dist.get(c, np.nan)
    pace_col = f"{c}_sec_per_km"
    df[pace_col] = df[c] / km
    seg_pace_cols.append(pace_col)

df["pace_std_min_km"] = df[seg_pace_cols].std(axis=1, skipna=True) / 60.0
q_km = df[seg_pace_cols].quantile([0.75, 0.25], axis=1)
df["pace_iqr_min_km"] = (q_km.loc[0.75] - q_km.loc[0.25]) / 60.0

def pace_sec_to_str(sec_per_km: float) -> str:
    """把秒/公里转成 'M:SS /km'，含稳健进位处理"""
    if pd.isna(sec_per_km) or sec_per_km <= 0:
        return ""
    total = int(round(sec_per_km))
    m, s = divmod(total, 60)
    return f"{m}:{s:02d} /km"

def sec_to_hms(sec: float) -> str:
    if pd.isna(sec): return ""
    sec = int(round(sec)); h = sec // 3600; m = (sec % 3600) // 60; s = sec % 60
    return f"{h}:{m:02d}:{s:02d}"

age_options      = sorted(df["age_group"].dropna().unique())
country_options  = sorted(df["country"].dropna().unique())
age_dropdown     = [{"label": ag, "value": ag} for ag in age_options]
country_dropdown = [{"label": c, "value": c} for c in country_options]

# DASH PART
app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Marathon Runner Performance Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Div([ html.Label("Age Group"), dcc.Dropdown(id="age_filter", options=age_dropdown, multi=True) ], style={"width": "30%"}),
        html.Div([
            html.Label("Gender"),
            dcc.RadioItems(
                id="gender_filter",
                options=[{"label":"All","value":"All"},{"label":"Male","value":"Male"},{"label":"Female","value":"Female"}],
                value="All", inline=True
            )
        ], style={"width": "30%"}),
        html.Div([ html.Label("Country"), dcc.Dropdown(id="country_filter", options=country_dropdown, multi=True) ], style={"width": "30%"}),
    ], style={"display":"flex","justifyContent":"space-between","marginBottom":"20px"}),

    dcc.Tabs(id="tabs", value="tab1", children=[
        dcc.Tab(label="Pace Stability",            value="tab1"),
        dcc.Tab(label="Finish Time by Age Group",  value="tab2"),
        dcc.Tab(label="Finish Time Distribution",  value="tab3"),
        dcc.Tab(label="Bib Search",                value="tab4"),
        dcc.Tab(label="Marathon Story (by bib)",   value="tab5"),
    ]),
    html.Div(id="tab-content")
])

def apply_filters(base_df, age_val, gender_val, country_val):
    dff = base_df.copy()
    if age_val:
        dff = dff[dff["age_group"].isin(age_val)]
    if gender_val and gender_val != "All":
        dff = dff[dff["gender"] == gender_val]
    if country_val:
        dff = dff[dff["country"].isin(country_val)]
    return dff

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("age_filter", "value"),
    Input("gender_filter", "value"),
    Input("country_filter", "value")
)
def render_tab(tab, age_val, gender_val, country_val):
    base = df
    dff  = apply_filters(base, age_val, gender_val, country_val)

    if dff.empty and tab in ("tab1", "tab2", "tab3"):
        return html.Div("No data matches selection.")

    if tab == "tab1":
        fig = px.scatter(
            dff, x="pace_std_min_km", y="pace_iqr_min_km",
            color="country", symbol="age_group",
            title="Pace Stability (based on per-km pace): Std vs IQR",
            labels={
                "pace_std_min_km":"Pace Std Dev (min per km)",
                "pace_iqr_min_km":"Pace IQR (min per km)"
            },
            template="simple_white",
        )
        fig.update_traces(marker=dict(size=6, opacity=0.7))
        fig.update_layout(margin=dict(l=50, r=30, t=60, b=50))
        fig.update_xaxes(zeroline=False, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
        fig.update_yaxes(zeroline=False, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
        return dcc.Graph(figure=fig)

    elif tab == "tab2":
        dff = dff.copy()
        dff["finish_time_min"] = pd.to_numeric(dff["finish_time_min"], errors="coerce")

        parts = []
        m = dff[dff["gender"] == "Male"].copy()
        if not m.empty: m["gender_shown"] = "Male"; parts.append(m)
        f = dff[dff["gender"] == "Female"].copy()
        if not f.empty: f["gender_shown"] = "Female"; parts.append(f)
        o = dff.copy(); o["gender_shown"] = "Overall"; parts.append(o)
        dfg = pd.concat(parts, ignore_index=True) if parts else dff.assign(gender_shown="Overall")

        def q1(x): x=pd.to_numeric(x,errors="coerce").dropna(); return np.percentile(x,25) if len(x) else np.nan
        def q3(x): x=pd.to_numeric(x,errors="coerce").dropna(); return np.percentile(x,75) if len(x) else np.nan

        agg = (
            dfg.groupby(["age_group","gender_shown"], dropna=True)["finish_time_min"]
               .agg(median="median", q1=q1, q3=q3, count="size")
               .reset_index()
        )
        agg["iqr"] = agg["q3"] - agg["q1"]
        agg["err"] = agg["iqr"] / 2.0

        if agg.empty:
            return html.Div("No data for the current selection.", style={"padding":"12px","color":"#555"})

        color_map = {"Overall":"#7f8c8d", "Male":"#1f77b4", "Female":"#e377c2"}
        try:
            age_sorted = sorted(agg["age_group"].dropna().unique(), key=lambda s: int(str(s).split("-")[0]))
        except Exception:
            age_sorted = list(agg["age_group"].dropna().unique())

        bars = []
        for g in ["Overall","Male","Female"]:
            sub = agg[agg["gender_shown"] == g]
            if sub.empty: continue
            custom = np.c_[sub["iqr"].values, sub["count"].values]
            bars.append(
                go.Bar(
                    name=g, x=sub["age_group"], y=sub["median"],
                    error_y=dict(type="data", array=sub["err"], visible=True),
                    marker_color=color_map.get(g),
                    customdata=custom,
                    hovertemplate=(
                        "Age: %{x}<br>Median: %{y:.1f} min<br>"
                        "IQR: %{customdata[0]:.1f} min<br>"
                        "n=%{customdata[1]}<extra>%{fullData.name}</extra>"
                    ),
                )
            )

        fig = go.Figure(bars)
        fig.update_layout(
            barmode="group",
            template="simple_white",
            title="Finish Time by Age Group (Median ± IQR/2)",
            xaxis_title="Age Group",
            yaxis_title="Finish Time (min)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=60, r=30, t=60, b=60),
        )
        fig.update_xaxes(categoryorder="array", categoryarray=age_sorted)
        return dcc.Graph(figure=fig)

    elif tab == "tab3":
        fig = px.histogram(
            dff, x="finish_time_min", color="country", nbins=50,
            histnorm="percent", barmode="overlay", opacity=0.55,
            title="Overall Finish Time Distribution",
            template="simple_white",
        )

        ft = dff["finish_time_min"].dropna()
        if len(ft) >= 3:
            p50, p80, p95 = np.percentile(ft, [50, 80, 95])

            xmin = float(np.nanmin(ft))
            xmax = float(np.nanmax(ft))
            pad  = max((xmax - xmin) * 0.02, 0.1)
            fig.update_xaxes(range=[xmin - pad, xmax + pad])

            def add_percentile_shape(x, color, dash):
                fig.add_shape(
                    type="line",
                    x0=x, x1=x, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(color=color, width=2, dash=dash),
                    layer="above"
                )

            add_percentile_shape(p50, "green",  "dash")
            add_percentile_shape(p80, "orange", "dot")
            add_percentile_shape(p95, "red",    "dot")

            fig.add_trace(go.Scatter(
                x=[xmin - 10], y=[0], mode="lines",
                line=dict(color="green", dash="dash"), name=f"50th percentile: {p50:.1f} min",
                showlegend=True, hoverinfo="skip", visible="legendonly"
            ))
            fig.add_trace(go.Scatter(
                x=[xmin - 10], y=[0], mode="lines",
                line=dict(color="orange", dash="dot"), name=f"80th percentile: {p80:.1f} min",
                showlegend=True, hoverinfo="skip", visible="legendonly"
            ))
            fig.add_trace(go.Scatter(
                x=[xmin - 10], y=[0], mode="lines",
                line=dict(color="red", dash="dot"), name=f"95th percentile: {p95:.1f} min",
                showlegend=True, hoverinfo="skip", visible="legendonly"
            ))

        fig.update_layout(
            margin=dict(l=220, b=100, r=40, t=50),
            legend=dict(
                x=-0.35, y=-0.25, orientation="v",
                bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(0,0,0,0.15)", borderwidth=1
            )
        )
        fig.update_yaxes(title="Percentage Density")
        fig.update_xaxes(title="Finish Time (minutes)")
        return dcc.Graph(figure=fig)

    elif tab == "tab4":
        full = base.copy()
        column_defs = [{"name": i, "id": i} for i in
                       ["bib","name","overall_rank","gender","country","age_group","finish_time_min"]
                       if i in full.columns]
        result = full[column_defs[0]["id"]].to_frame().join([full[c["id"]] for c in column_defs[1:]])
        return html.Div([
            html.Div(f"Showing {len(result)} rows / Total {len(full)} rows", style={"margin":"6px 0", "color":"#555"}),
            dash_table.DataTable(
                columns=column_defs,
                data=result.to_dict("records"),
                page_size=100,
                style_table={"overflowX":"auto"},
                style_cell={"textAlign":"center"},
                style_header={"fontWeight":"bold"},
                export_format="csv",
                export_headers="display",
            )
        ])

    elif tab == "tab5":
        return html.Div([
            html.Div([
                html.Label("Enter bib:"),
                dcc.Input(id="bib-input", type="number", min=1, step=1,
                          debounce=False, placeholder="e.g., 4292"),
            ], style={"padding":"8px 0"}),
            html.Div(id="story-output", style={"marginTop":"10px"})
        ], style={"maxWidth":"720px"})

    return html.Div("Unknown tab")

@app.callback(
    Output("story-output", "children"),
    Input("bib-input", "value")
)
def make_story(bib_value):
    if bib_value is None or bib_value == "":
        return html.Div("Enter a bib to generate the story.", style={"color":"#555"})

    try:
        bib_int = int(bib_value)
    except Exception:
        return html.Div("Please enter a valid integer bib.", style={"color":"#a00"})

    candidate = df[pd.to_numeric(df["bib"], errors="coerce") == bib_int]
    if candidate.empty and "pos" in df.columns:
        candidate = df[pd.to_numeric(df["pos"], errors="coerce") == bib_int]
    if candidate.empty:
        return html.Div(f"No runner found with bib={bib_int}.", style={"color":"#a00"})

    row = candidate.iloc[0]

    overall_pace_sec = (row["finish_time"] / FULL_KM) if pd.notna(row["finish_time"]) else np.nan
    first_pace_sec   = (row["first_half_sec"] / FIRST_HALF_KM) if pd.notna(row["first_half_sec"]) else np.nan
    second_pace_sec  = (row["second_half_sec"] / SECOND_HALF_KM) if pd.notna(row["second_half_sec"]) else np.nan
    neg_split = (second_pace_sec < first_pace_sec) if (pd.notna(first_pace_sec) and pd.notna(second_pace_sec)) else False

    story_rows = [
        {"Metric":"Bib",              "Value": int(row.get("bib")) if pd.notna(row.get("bib")) else ""},
        {"Metric":"Name",             "Value": row.get("name","")},
        {"Metric":"Country",          "Value": row.get("country","")},
        {"Metric":"Gender",           "Value": row.get("gender","")},
        {"Metric":"Age Group",        "Value": row.get("age_group","")},
        {"Metric":"Overall Rank",     "Value": int(row.get("overall_rank")) if pd.notna(row.get("overall_rank")) else ""},
        {"Metric":"Gender Rank",      "Value": int(row.get("gender_rank"))  if pd.notna(row.get("gender_rank")) else ""},
        {"Metric":"Country Rank",     "Value": int(row.get("country_rank")) if pd.notna(row.get("country_rank")) else ""},
        {"Metric":"Finish Time",      "Value": sec_to_hms(row.get("finish_time", np.nan))},
        {"Metric":"First Half Pace",  "Value": pace_sec_to_str(first_pace_sec)},
        {"Metric":"Second Half Pace", "Value": pace_sec_to_str(second_pace_sec)},
        {"Metric":"Overall Pace",     "Value": pace_sec_to_str(overall_pace_sec)},
        {"Metric":"Negative Split?",  "Value": "Yes" if neg_split else "No"},
    ]

    table = dash_table.DataTable(
        columns=[{"name":"Metric","id":"Metric"},{"name":"Value","id":"Value"}],
        data=story_rows,
        style_cell={"textAlign":"left","padding":"8px"},
        style_header={"fontWeight":"bold"},
        style_table={"maxWidth":"720px"},
    )

    narrative_text = (
        f"Bib #{int(row.get('bib')) if pd.notna(row.get('bib')) else '—'} — "
        f"{row.get('name','Runner')} finished in {sec_to_hms(row.get('finish_time', np.nan))}, "
        f"ranked #{int(row.get('overall_rank')) if pd.notna(row.get('overall_rank')) else '—'} overall, "
        f"#{int(row.get('gender_rank')) if pd.notna(row.get('gender_rank')) else '—'} in {row.get('gender','—')}, "
        f"and #{int(row.get('country_rank')) if pd.notna(row.get('country_rank')) else '—'} in {row.get('country','—')}. "
        f"{'Negative split achieved.' if neg_split else 'No negative split.'}"
    )
    narrative = html.Div(narrative_text, style={"margin":"8px 0","color":"#333"})

    return html.Div([narrative, table])

if __name__ == "__main__":
    port = 8051
    webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(port=port, debug=True)

# Start the server
if __name__ == '__main__':
    app.run_server(debug=True)
