from numpy import linspace,zeros_like,ones
from random import randint
from bin3D import AdjustPackingGame
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import listdir,makedirs
from os.path import exists
size = 10


def Discrete(ax, size, mask):
    x = linspace(0, size, 1000)
    vertical = zeros_like(x)
    points = []
    for item in range(len(mask)):
        if mask[item] == 1:
            points.append([item//size, item%size, 0])
    for point in points:
        ax.scatter(point[0], point[1],point[2], s = 1, color = 'gray',zorder = 2)
    for i in range(size+1):
        ax.plot3D(vertical+i, x, vertical, linewidth= '0.5', linestyle='--', color="gray",zorder = 1)
        ax.plot3D(x, vertical+i, vertical, linewidth= '0.5', linestyle='--', color="gray",zorder = 1)


def Chessboard(size, mask):
    global bx
    x = linspace(0, size, 1000)
    vertical = zeros_like(x)
    points = []
    bx.axis('off')
    for item in range(len(mask)):
        if mask[item] == 1:
            points.append([item//size, item%size])
    for point in points:
        bx.scatter(point[0], point[1],color = 'gray',s = 70,zorder = 2)
        bx.scatter(point[0], point[1],color = 'white',s = 10,zorder = 3)

    bx.plot(vertical + 0, x, color="gray", linestyle='-', zorder=1)
    bx.plot(x, vertical + 0, color="gray", linestyle='-', zorder=1)

    for i in range(1,size+1):
        bx.plot(vertical+i, x, color="gray",linestyle = '--', zorder = 1)
        bx.plot(x, vertical+i, color="gray",linestyle = '--', zorder = 1)

    bx.quiver(10, 0, 12, 0, linestyle='-', linewidth=1.5, color='gray')
    bx.quiver(0, 10, 0,12, linestyle='-', linewidth=1.5, color='gray',)
    bx.text(10,0-0.8,'x',fontsize = 20,color = 'gray')
    bx.text(0-0.8,10,'y',fontsize = 20,color = 'gray')
    for i in range(10):
        bx.text(i,0-0.7,'{}'.format(i),color = 'gray',fontsize = 15)
        bx.text(0-0.7,i,'{}'.format(i),color = 'gray',fontsize = 15)

def save_trajs(game):
    boxes, last_box = game.space.boxes, game.temp_box
    if not exists("trajs"):
        makedirs("./trajs")
    index = len(listdir("./trajs"))
    f = open("./trajs/traj{}.txt".format(index),"w")
    f.write("The traj index is:{}\n".format(game.box_creator.index))
    f.write("The dataset is:" + game.data_name + '\n')
    for box in boxes:
        f.write("{}, {}, {}, {}, {}, {}\n".format(box.x, box.y, box.z, box.lx, box.ly, box.lz))
    f.write("{}, {}, {}".format(last_box[0], last_box[1], last_box[2]))
    f.close()

def plot_container(game):
    game.container.color = 'gray'
    game.container.plot_linear_cube(ax,"",linestyle = '--')


    ax.quiver(0, 0, 0, size * 1.2, 0, 0, linestyle='-', linewidth=1.5, color='r', arrow_length_ratio=.05)
    ax.quiver(0, 0, 0, 0, size * 1.2, 0, linestyle='-', linewidth=1.5, color='g', arrow_length_ratio=.08)
    ax.quiver(0, 0, 0, 0, 0, size * 1.2, linestyle='-', linewidth=1.5, color='b', arrow_length_ratio=.08)
    ax.text3D(size * 1.3, 0, -1, 'x', fontsize=15, color='r')
    ax.text3D(0, size * 1.3, 0 + 0.5, 'y', fontsize=15, color='g')
    ax.text3D(0 - 0.5, 0 - 0.5, size * 1.3, 'z', fontsize=15, color='b')
    # infor_around_container(game.container,ax)


def infor_around_container(box,ax):
    box.set_color(choose_color())
    vertex = box.vertex

    ax.text3D((vertex[0][0]+vertex[1][0])/2, vertex[0][1], vertex[0][2]-1.5, 'L', fontsize=13,
              verticalalignment="center", color = 'gray',
              horizontalalignment="center", zorder=0)
    ax.text3D(vertex[1][0], (vertex[1][1]+vertex[2][1])/2, vertex[2][2]-1.5, 'W', fontsize=13,
              verticalalignment="center", color = 'gray',
              horizontalalignment="center", zorder=0)
    ax.text3D(vertex[3][0]+0.5, vertex[3][1]+0.5, (vertex[2][2]+vertex[6][2])/2, 'H', fontsize=13,
              verticalalignment="center", color = 'gray',
              horizontalalignment="center", zorder=0)

def choose_color():
    color = ['gold', 'springgreen','pink','aquamarine','cyan']
    index = randint(0,len(color)-1)
    return color[index]

def draw_3D_box(game,container = [10,10,10]):
    global ax
    ax.grid(False)
    ax.axis('off')
    plot_container(game)
    boxes = game.space.boxes
    try_box = game.space.try_box

    for i in range(len(boxes)):
        boxes[i].set_color(choose_color())
        boxes[i].plot_opaque_cube(ax, str(i))
    if try_box is not None:
        try_box.set_color('green')
        try_box.plot_opaque_cube(ax, str(len(boxes)+1))

    mask = ones(size**2)
    Discrete(ax,size,mask)

    ax.set_xlim3d(0, container[0] * 1.5)
    ax.set_ylim3d(0, container[0] * 1.5)
    ax.set_zlim3d(0, container[0] * 1.5)
    ax.set_xlabel("x-label", color='r')
    ax.set_ylabel("y-label", color='g')
    ax.set_zlabel("z-label", color='b')

def on_press(event):
    global game, ax, bx
    action = [round(event.xdata), round(event.ydata)]
    action = int(action[0]*size + action[1])
    print("the currunt action is:", [round(event.xdata), round(event.ydata)] )
    if event.button==3:
        game.try_step([action])
    elif event.button==1:
        observation, reward, done, info = game.step([action])
        if done:
            print("Game Over!")
            save_trajs(game)
            game.reset()
    mask = game.get_possible_position()
    mask = mask.reshape(size ** 2)
    plt.figure(1)
    plt.cla()
    plt.figure(2)
    plt.cla()
    Chessboard(size, mask)
    draw_3D_box(game)
    ax.figure.canvas.draw_idle()
    bx.figure.canvas.draw_idle()

def on_motion(event):
    global game, ax, bx
    action = [round(event.xdata), round(event.ydata)]
    action = int(action[0]*size + action[1])
    print("the currunt action is:", [round(event.xdata), round(event.ydata)] )
    game.try_step([action])
    plt.figure(1)
    plt.cla()
    draw_3D_box(game)
    ax.figure.canvas.draw_idle()


game = AdjustPackingGame(adjust_grid=size)
game.reset()
fig1 = plt.figure(1)
fig2 = plt.figure(2,figsize = (5,5))
ax = fig1.add_subplot(111, projection='3d')
ax.grid(False)
ax.axis('off')
bx = fig2.add_subplot(111)
mask = game.get_possible_position()
mask = mask.reshape(size**2)
Chessboard(size, mask)
bx.figure.canvas.mpl_connect('button_press_event', on_press)
bx.figure.canvas.mpl_connect('motion_notify_event', on_motion)
plt.show()
