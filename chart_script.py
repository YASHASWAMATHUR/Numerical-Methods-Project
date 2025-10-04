import plotly.graph_objects as go
import json

# Load the data
data = {
    "dose_cGy": [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0, 2100.0, 2200.0],
    "target_volume_percent": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 90.9, 90.9, 81.8, 81.8, 72.7, 63.6, 54.5, 36.4, 27.3, 18.2, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 0.0],
    "oar_volume_percent": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 90.0, 70.0, 40.0, 20.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

# Create figure
fig = go.Figure()

# Add target volume trace
fig.add_trace(go.Scatter(
    x=data["dose_cGy"],
    y=data["target_volume_percent"],
    mode='lines',
    name='Target',
    line=dict(color='#1FB8CD', width=3),
    cliponaxis=False
))

# Add OAR volume trace
fig.add_trace(go.Scatter(
    x=data["dose_cGy"],
    y=data["oar_volume_percent"],
    mode='lines',
    name='OAR',
    line=dict(color='#DB4545', width=3),
    cliponaxis=False
))

# Update layout
fig.update_layout(
    title='DVH: Target vs OAR Dose Distribution',
    xaxis_title='Dose (cGy)',
    yaxis_title='Volume %',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update y-axis range to show full percentage scale
fig.update_yaxes(range=[0, 105])

# Save the chart
fig.write_image('dvh_chart.png')