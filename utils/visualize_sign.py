import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def get_sign_landmarks(cls:int, instance:int, dataset:dict):
    """ Get the landmarks of a video specified using its class and instance number."""

    # video path
    # vid_path = f'./clips/{cls}/{instance}.MOV'
    vid_path = os.path.join('./clips', str(cls))
    vid_path = os.path.join(vid_path, f'{instance}.MOV')

    # get index of data instance from dataset
    data_idx = dataset['path'].index(vid_path)
    print(f'Index of Class = {cls}, Instance = {instance}: {data_idx}')

    # extract data of video using index
    return dataset['data'][data_idx]

def display_sign(data, cls:str=None, save_animation:bool=False):
    """ Display the sign via its landmarks. """

    # --- Prepare figure and properties
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')

    # setting the axes properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    txt_title = ax.set_title(f'')
    scatter_plot = ax.scatter([], [], [], c='r', edgecolors='k')
    ax.view_init(elev=120, azim=90)

    # animation to draw each frame
    def animate(i):
        scatter_plot._offsets3d = (data[i,:,0], data[i,:,1], data[i,:,2])
        txt_title.set_text(f'{str(cls).upper()} \n Frame: {i:2d}')
        return (scatter_plot, txt_title)

    # create animation object
    anim = FuncAnimation(fig, animate, frames=data.shape[0], interval=100, blit=True)

    if save_animation:
        anim.save(f'animations/{cls}.mp4')
    
    return anim
