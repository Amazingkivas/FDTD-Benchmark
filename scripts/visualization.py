import os
import glob
import subprocess
import platform
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

components_num = {"Ex": '1', "Ey": '2', "Ez": '3', "Bx": '4', "By": '5', "Bz": '6'}

def get_animation(component):
    folder_path = f'OutFiles_{components_num[component]}'
    file_names_base = sorted(glob.glob(folder_path + '/*.csv'))
    file_names = sorted(file_names_base, key=lambda x: int(x.split(os.path.sep)[-1].split('.')[0]))

    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        data = pd.read_csv(file_names[frame], sep=';')
        data = data.apply(lambda x: x.str.replace(',', '.').astype(float) if x.dtype == 'object' else x)
        ax.imshow(data, cmap='viridis', vmin=-0.07, vmax=0.07, interpolation='none') # RdBu
        ax.set_title('Frame {}'.format(frame + 1))
    
    if not os.path.exists('animations'):
        os.makedirs('animations')
    
    ani = FuncAnimation(fig, update, frames=len(file_names), interval=50)
    ani.save(f'animations/animation_{component}.gif', writer='imagemagick')


def get_heatmap(component, iteration):
    data = pd.read_csv(f'OutFiles_{components_num[component]}/{iteration}.csv', sep=';')

    data = data.applymap(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)

    plt.imshow(data, cmap='viridis', vmin=-0.07, vmax=0.07, interpolation='none')
    plt.colorbar()

    if not os.path.exists('heatmap'):
        os.makedirs('heatmap')

    heatmap_path = os.path.join('heatmap', f'heatmap_iteration_{iteration}_component_{component}.png')
    plt.savefig(heatmap_path)


def execute_cpp(cpp_executable, grid_size, iters_num):
    args = [cpp_executable, str(grid_size), str(iters_num), str(int(True))]
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError:
        print('Error when starting the C++ project')
    except FileNotFoundError:
        print(f'{cpp_executable} not found')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_cpp', action='store_true', help="Run C++ application before generating data")
    parser.add_argument('--function', choices=['heatmap', 'animation'], help="Select function to execute")
    parser.add_argument('--iteration', type=int, default=0, help="Iteration number for heatmap")
    parser.add_argument('component', type=str, help="Component for analysis")

    group = parser.add_argument_group('conditional arguments')
    group.add_argument('--grid_size', type=int, help="Grid size")
    group.add_argument('--iters_num', type=int, help="Number of iterations")
    
    group.add_argument('--executable', choices=['sample', 'kokkos_sample'], 
                      help="Choose which C++ executable to run (required if --run_cpp is specified)")

    args = parser.parse_args()

    if args.run_cpp:
        if not (args.grid_size and args.iters_num):
            parser.error("The --run_cpp flag requires --grid_size and --iters_num arguments")

        if not args.executable:
            parser.error("The --executable argument is required when --run_cpp is specified.")

        system = platform.system()
        if system == 'Windows':
            cpp_executable = f'src/Release/{args.executable}.exe'
        else:
            cpp_executable = f'src/Release/{args.executable}'

        execute_cpp(cpp_executable, args.grid_size, args.iters_num)

    if args.function == 'heatmap':
        get_heatmap(args.component, args.iteration)
    elif args.function == 'animation':
        get_animation(args.component)
        