import streamlit as st
import plotly.graph_objects as go

def render_sort(steps, current_step, highlight_indices=[]):
    arr = steps[current_step]
    n = len(arr)

    neon_colors = [
        f'hsl({int(i * 360 / n)}, 100%, 60%)'
        for i in range(n)
    ]

    fig = go.Figure()

    # Transparent bars with bright neon outline only
    fig.add_trace(go.Bar(
        y=arr,
        marker=dict(
            color='rgba(0,0,0,0)',        # fully transparent fill
            line=dict(
                color=neon_colors,        # neon colored outline
                width=2.5
            ),
        ),
        hovertemplate='Value: %{y}<extra></extra>'
    ))

    # Add subtle neon glow behind each bar
    for i, val in enumerate(arr):
        color_hsl = int(i * 360 / n)
        fig.add_shape(
            type="rect",
            x0=i - 0.4,
            x1=i + 0.4,
            y0=0,
            y1=val,
            line=dict(width=0),
           fillcolor=f'hsla({color_hsl}, 100%, 60%, 0.30)'
        )

    fig.update_layout(
        plot_bgcolor='#0d0d0d',
        paper_bgcolor='#0d0d0d',
        font=dict(color='#ffffff'),
        showlegend=False,
        title=dict(
            text=f'⚡ Step {current_step + 1} of {len(steps)}',
            font=dict(color='#7C3AED', size=18),
            x=0.5
        ),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            showline=False
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            showline=False
        ),
        bargap=0.15,
        margin=dict(l=20, r=20, t=60, b=20),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)