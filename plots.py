# Visualization
import plotly.express as px # for data visualization
import matplotlib.pyplot as plt # for showing handwritten digits


def showPlot(X_trans,y,algorithm, color):
  # x=X_trans[:,0]
  # y=X_trans[:,1]
  # Create a scatter plot
  fig = px.scatter(None, x=X_trans, y=y, 
                  labels={
                      "x": "Dimension 1",
                      "y": "Dimension 2",
                  },
                  opacity=1, color=color)

  # Change chart background color
  fig.update_layout(dict(plot_bgcolor = 'white'))

  # Update axes lines
  fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                  zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                  showline=True, linewidth=1, linecolor='black')

  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                  zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                  showline=True, linewidth=1, linecolor='black')

  # Set figure title
  fig.update_layout(title_text=algorithm)

  # Update marker size
  fig.update_traces(marker=dict(size=3))

  fig.show()