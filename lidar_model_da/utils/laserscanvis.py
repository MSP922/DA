import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from matplotlib import pyplot as plt

class LaserScanVis:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, scan, hr_scan, scan_names, offset=0):
    self.scan = scan
    self.hr_scan = hr_scan

    self.scan_names = scan_names
    self.offset = offset
    self.total = len(self.scan_names)

    self.reset()
    self.update_scan()

  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities

    self.canvas = SceneCanvas(keys='interactive', show=True)
    self.canvas.events.key_press.connect(self.key_press)
    self.canvas.events.draw.connect(self.draw)
    self.grid = self.canvas.central_widget.add_grid()

    self.scan_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.canvas.scene)
    self.grid.add_widget(self.scan_view, 0, 0)
    self.scan_vis = visuals.Markers()
    self.scan_view.camera = 'turntable'
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)

    ###################################################################
    self.hr_canvas = SceneCanvas(keys='interactive', show=True)
    self.hr_canvas.events.key_press.connect(self.key_press)
    self.hr_canvas.events.draw.connect(self.draw)
    self.hr_grid = self.hr_canvas.central_widget.add_grid()

    self.hr_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.hr_canvas.scene)
    self.hr_grid.add_widget(self.hr_view, 0, 0)
    self.hr_vis = visuals.Markers()
    self.hr_view.camera = 'turntable'
    self.hr_view.add(self.hr_vis)
    visuals.XYZAxis(parent=self.hr_view.scene)

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def update_scan(self):
    # first open data
    self.scan.open_scan(self.scan_names[self.offset])
    self.hr_scan.open_scan(self.scan_names[self.offset])

    # then change names
    self.canvas.title = "input (16L) " + str(self.offset)
    self.hr_canvas.title = "Up-sampled result (64L) " + str(self.offset)

    # then do all the point cloud stuff

    # plot scan
    power = 16
    # print()
    range_data = np.copy(self.scan.unproj_range)
    range_data = range_data**(1 / power)
    viridis_range = ((range_data - range_data.min()) /
                     (range_data.max() - range_data.min()) *
                     255).astype(np.uint8)
    viridis_map = self.get_mpl_colormap("viridis")
    viridis_colors = viridis_map[viridis_range]
    self.scan_vis.set_data(self.scan.points,
                           face_color=viridis_colors[..., ::-1],
                           edge_color=viridis_colors[..., ::-1],
                           size=1)

    hr_range_data = np.copy(self.hr_scan.unproj_range)
    hr_range_data = hr_range_data**(1 / power)
    hr_viridis_range = ((hr_range_data - hr_range_data.min()) /
                     (hr_range_data.max() - hr_range_data.min()) *
                     255).astype(np.uint8)
    hr_viridis_colors = viridis_map[hr_viridis_range]
    self.hr_vis.set_data(self.hr_scan.points,
                           face_color=hr_viridis_colors[..., ::-1],
                           edge_color=hr_viridis_colors[..., ::-1],
                           size=1)

  # interface
  def key_press(self, event):
    self.canvas.events.key_press.block()
    self.hr_canvas.events.key_press.block()
    if event.key == 'N':
      self.offset += 1
      if self.offset >= self.total:
        self.offset = 0
      self.update_scan()
    elif event.key == 'B':
      self.offset -= 1
      if self.offset < 0:
        self.offset = self.total - 1
      self.update_scan()
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()
    if self.hr_canvas.events.key_press.blocked():
      self.hr_canvas.events.key_press.unblock()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    self.hr_canvas.close()
    vispy.app.quit()

  def run(self):
    vispy.app.run()
