from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

def animate_colormap(data, axs_labels=None, times=None, c_label=None, interval=50, sz=5, cm='bwr'):
    """
    Create an animation of a 2D colormap over time.

    This function takes a 3D numpy array representing time-dependent 2D data and 
    generates an animated colormap. The animation can be rendered in a Jupyter 
    notebook using the :class:`IPython.display.HTML` object.

    :param data: 
        A 3D array with shape (t, L, M), where t is the number of timesteps, 
        and L and M represent the dimensions of each data slice (e.g., rows and columns).
    :type data: numpy.ndarray
    :param axs_labels: 
        A list of length 2 containing strings for the x-axis and y-axis labels, 
        in the 0th and 1st positions respectively. Default is None.
    :type axs_labels: list of str, optional
    :param times: 
        A 1D array of length t, where each value represents the time corresponding 
        to each timestep. Default is None.
    :type times: numpy.ndarray, optional
    :param c_label: 
        A label for the colormap (color bar). Default is None.
    :type c_label: str, optional
    :param interval: 
        The delay in milliseconds between frames in the animation. Default is 50.
    :type interval: int, optional
    :param sz: 
        The size of the plot figure. Default is 5.
    :type sz: int, optional
    :param cm: 
        The colormap to use for the animation. Default is 'bwr'.
    :type cm: str, optional

    :return: 
        An animation object that can be rendered in a Jupyter notebook using 
        :class:`IPython.display.HTML`.
    :rtype: matplotlib.animation.FuncAnimation

    :examples:
        >>> import numpy as np
        >>> from IPython.display import HTML
        >>> data = np.random.random((50, 100, 100))  # Example 3D array (50 timesteps, 100x100 grid)
        >>> ani = animate_colormap(data, axs_labels=["X-axis", "Y-axis"], times=np.arange(50))
        >>> HTML(ani.to_jshtml())
    """
    def init():
        img.set_data(data[0])
        vmin = np.amin(data[0])
        vmax = np.amax(data[0])
        img.set_clim(vmin, vmax)
        if times is not None:
            ax.set(title=f"Time = {times[0]}")
        return (img,)

    def update(i):
        img.set_data(data[i])
        vmin = np.amin(data[i])
        vmax = np.amax(data[i])
        img.set_clim(vmin, vmax)
        if times is not None:
            ax.set(title=f"Time = {times[i]}")
        return (img,)

    fig, ax = plt.subplots(1, 1, figsize=(sz, sz))
    img = ax.imshow(data[0], cmap=cm, vmin=np.amin(data[0]), vmax=np.amax(data[0]))
    if axs_labels is not None:
        ax.set_xlabel(axs_labels[0])
        ax.set_ylabel(axs_labels[1])
    fig.colorbar(img, ax=ax, orientation="horizontal", label=c_label, pad=0.2)
    ani = animation.FuncAnimation(fig, update, frames=len(data), init_func=init, interval=interval, blit=True)
    plt.close()
    return ani

def read_data(file_path):
    data = []
    with open(file_path, "r") as f:
        list_of_text = f.readlines()
        for curr_line in list_of_text:
            curr_data = [float(i) for i in curr_line.split(",")[:-1]]
            data.append(curr_data)

    data = np.array(data)

    return data